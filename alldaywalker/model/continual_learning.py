import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import json
import os
import numpy as np
from PIL import Image

class EWCLoss:
    """Elastic Weight Consolidation loss"""
    def __init__(self, model, device='cuda', mode='online'):
        self.model = model
        self.device = device
        self.mode = mode
        self.fisher_dict = {}
        self.optimal_params_dict = {}
        self.model_dtype = next(model.parameters()).dtype
        
        # Save trained U3 and U4 parameters
        self.trained_U3_params = {}  # {scene_idx: saved_U3_row}
        self.trained_U4_params = {}  # {env_idx: saved_U4_row}
        self.trained_routes = set()  # Save trained (scene_idx, env_idx) pairs
        
    def _prepare_batch(self, batch):
        """Prepare batch data, ensuring it's on the correct device"""
        prepared_batch = {}
        
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # Move tensor to correct device
                prepared_batch[k] = v.to(self.device)
            elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                # Handle tensor lists
                prepared_batch[k] = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in v]
            else:
                # Keep as is
                prepared_batch[k] = v
        
        return prepared_batch
    
    def save_task_specific_params(self, scene_idx, env_idx):
        """Save U3 and U4 parameters after current task training"""
        from streamvln.model.tucker_lora_layers import Tucker4DLoRALinear
        
        for name, module in self.model.named_modules():
            if isinstance(module, Tucker4DLoRALinear):
                # Save specified row of U3
                if scene_idx not in self.trained_U3_params:
                    self.trained_U3_params[scene_idx] = module.lora_layer.U3[scene_idx].detach().clone()
                
                # Save specified row of U4
                if env_idx not in self.trained_U4_params:
                    self.trained_U4_params[env_idx] = module.lora_layer.U4[env_idx].detach().clone()
        
        self.trained_routes.add((scene_idx, env_idx))
    
    def compute_stability_loss(self, stability_weight=1.0):
        """Compute stability loss for U3 and U4"""
        stability_loss = 0.0
        from streamvln.model.tucker_lora_layers import Tucker4DLoRALinear
        
        for name, module in self.model.named_modules():
            if isinstance(module, Tucker4DLoRALinear):
                # U3 stability loss
                for scene_idx, saved_param in self.trained_U3_params.items():
                    current_param = module.lora_layer.U3[scene_idx]
                    stability_loss += torch.norm(current_param - saved_param.to(current_param.device), p=2) ** 2
                
                # U4 stability loss
                for env_idx, saved_param in self.trained_U4_params.items():
                    current_param = module.lora_layer.U4[env_idx]
                    stability_loss += torch.norm(current_param - saved_param.to(current_param.device), p=2) ** 2
        
        return stability_loss * stability_weight
    
    def save_fisher_matrix(self, path):
        """Save Fisher matrix and U3/U4 parameters to file"""
        save_dict = {
            'fisher': {k: v.cpu() for k, v in self.fisher_dict.items()},
            'optimal_params': {k: v.cpu() for k, v in self.optimal_params_dict.items()},
            'trained_U3_params': {k: v.cpu() for k, v in self.trained_U3_params.items()},
            'trained_U4_params': {k: v.cpu() for k, v in self.trained_U4_params.items()},
            'trained_routes': list(self.trained_routes)
        }
        torch.save(save_dict, path)
        print(f"Fisher matrix and task-specific params saved to {path}")
    
    def load_fisher_matrix(self, path):
        """Load Fisher matrix and U3/U4 parameters from file"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            self.fisher_dict = {k: v.to(self.device) for k, v in checkpoint['fisher'].items()}
            self.optimal_params_dict = {k: v.to(self.device) for k, v in checkpoint['optimal_params'].items()}
            
            # Load U3/U4 parameters
            if 'trained_U3_params' in checkpoint:
                self.trained_U3_params = {k: v.to(self.device) for k, v in checkpoint['trained_U3_params'].items()}
            if 'trained_U4_params' in checkpoint:
                self.trained_U4_params = {k: v.to(self.device) for k, v in checkpoint['trained_U4_params'].items()}
            if 'trained_routes' in checkpoint:
                self.trained_routes = set(checkpoint['trained_routes'])
            
            print(f"Fisher matrix and task-specific params loaded from {path}")
    
    def compute_fisher_matrix(self, dataloader, num_samples=20):
        """Compute Fisher information matrix - only for shared parameters G, U1, U2"""
        import torch.distributed as dist
        
        self.fisher_dict = {}
        self.optimal_params_dict = {}
        
        # Save original training state
        was_training = self.model.training
        original_grad_enabled = torch.is_grad_enabled()
        
        # Temporarily disable gradient checkpointing
        original_gradient_checkpointing = None
        if hasattr(self.model, 'gradient_checkpointing'):
            original_gradient_checkpointing = self.model.gradient_checkpointing
            self.model.gradient_checkpointing = False
        
        # Set to evaluation mode but enable gradients
        self.model.eval()
        torch.set_grad_enabled(True)
        
        # Check if dataloader is empty
        if dataloader is None or len(dataloader) == 0:
            print("Warning: Empty dataloader for Fisher matrix computation")
            return
        
        # Get rank information
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Initialize local Fisher matrix
        local_fisher = {}
        
        # First collect all parameters that need Fisher computation and ensure requires_grad=True
        target_params = []
        param_names = []
        
        for name, param in self.model.named_parameters():
            clean_name = name.replace('module.', '') if 'module.' in name else name
            
            # Check if it's Tucker layer's G, U1, U2 parameters (shared parameters)
            if '.lora_layer.G' in clean_name or \
            '.lora_layer.U1' in clean_name or \
            '.lora_layer.U2' in clean_name:
                # Ensure it's not U3 or U4
                if '.lora_layer.U3' not in clean_name and '.lora_layer.U4' not in clean_name:
                    # Force set requires_grad=True
                    param.requires_grad = True
                    
                    target_params.append(param)
                    param_names.append(name)
                    
                    # Initialize Fisher matrix
                    local_fisher[name] = torch.zeros_like(param, device='cpu', dtype=torch.float32)
                    self.optimal_params_dict[name] = param.data.clone().cpu().float()
                    
                    if rank == 0:
                        print(f"  Found shared parameter: {name}, shape: {param.shape}")
        
        # If no parameters found, print more detailed debug info
        if len(local_fisher) == 0:
            print(f"Warning: No shared Tucker parameters (G, U1, U2) found for Fisher matrix computation on rank {rank}")
            return
        
        print(f"Found {len(local_fisher)} shared Tucker parameters (G, U1, U2) for Fisher matrix on rank {rank}")
        
        # Reduce samples per rank
        samples_per_rank = min(5, num_samples // world_size, len(dataloader))
        local_sample_count = 0
        
        # Ensure model is in correct state
        self.model.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            if local_sample_count >= samples_per_rank:
                break
            
            if batch is None:
                continue
                
            batch = self._prepare_batch(batch)
            
            if 'input_ids' not in batch or batch['input_ids'] is None:
                continue
                
            batch_size = batch['input_ids'].shape[0]
            
            # Only process first sample to reduce memory usage
            single_batch = self._create_single_batch(batch, 0)
            
            try:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Ensure all target parameters have requires_grad=True
                for param in target_params:
                    param.requires_grad = True
                
                # Forward pass
                outputs = self.model(**single_batch)
                
                if outputs is None:
                    continue
                
                # Get loss
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    loss = outputs[0]
                else:
                    continue
                
                if loss is None or not loss.requires_grad:
                    continue
                
                # Scale loss to avoid numerical issues
                loss = loss * 0.01
                
                # Compute gradients (only for target parameters)
                grads = torch.autograd.grad(
                    loss,
                    target_params,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True
                )
                
                # Accumulate Fisher information
                for param_name, grad in zip(param_names, grads):
                    if grad is not None:
                        # Immediately convert to CPU and accumulate
                        grad_sq = (grad.data.float() ** 2).cpu()
                        local_fisher[param_name] += grad_sq
                        del grad, grad_sq
                
                local_sample_count += 1
                if rank == 0 and local_sample_count % 1 == 0:
                    print(f"Rank {rank}: Processed sample {local_sample_count}/{samples_per_rank}")
                
                # Clear gradients
                self.model.zero_grad()
                torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at sample {local_sample_count} on rank {rank}, cleaning up...")
                    torch.cuda.empty_cache()
                    self.model.zero_grad()
                    continue
                else:
                    print(f"Runtime error: {e}")
                    continue
            except Exception as e:
                print(f"Unexpected error in Fisher computation: {e}")
                continue
        
        # Restore original settings
        if original_gradient_checkpointing is not None:
            self.model.gradient_checkpointing = original_gradient_checkpointing
        
        torch.set_grad_enabled(original_grad_enabled)
        
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        
        # Check if any samples were successfully processed
        if local_sample_count == 0:
            print(f"Warning: No samples were successfully processed for Fisher matrix on rank {rank}")
            # Create small non-zero Fisher matrix as fallback
            for name in local_fisher:
                local_fisher[name] = torch.ones_like(local_fisher[name]) * 1e-6
            local_sample_count = 1  # Avoid division by 0
        
        # Normalize local Fisher
        for name in local_fisher:
            local_fisher[name] /= max(local_sample_count, 1)
        
        # Distributed aggregation of Fisher matrix
        if dist.is_initialized():
            for name, fisher in local_fisher.items():
                fisher_tensor = fisher.to(self.device)
                dist.all_reduce(fisher_tensor, op=dist.ReduceOp.SUM)
                fisher_tensor /= world_size
                self.fisher_dict[name] = fisher_tensor
        else:
            for name, fisher in local_fisher.items():
                self.fisher_dict[name] = fisher.to(self.device)
        
        if rank == 0:
            print(f"Computed Fisher matrix for {len(self.fisher_dict)} shared parameters (G, U1, U2) with {local_sample_count * world_size} samples")
            # Print some statistics
            for name in list(self.fisher_dict.keys())[:3]:
                fisher = self.fisher_dict[name]
                print(f"  {name}: mean={fisher.mean().item():.2e}, "
                    f"max={fisher.max().item():.2e}, "
                    f"min={fisher.min().item():.2e}")
        
        torch.cuda.empty_cache()


    def _create_single_batch(self, batch, idx):
        """Helper function to create single-sample batch"""
        single_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                single_batch[k] = v[idx:idx+1]
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                single_batch[k] = [v[idx]] if idx < len(v) else [v[0]]
            else:
                single_batch[k] = v
        return single_batch


    def update_fisher_matrix(self, dataloader, alpha=0.9):
        """Online update Fisher information matrix - fix DeepSpeed compatibility"""
        if self.mode != 'online':
            return self.compute_fisher_matrix(dataloader)
        
        import torch.distributed as dist
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Initialize new Fisher matrix
        new_fisher = {}
        new_optimal = {}
        
        # Save original state
        was_training = self.model.training
        original_grad_enabled = torch.is_grad_enabled()
        
        # Temporarily disable gradient checkpointing
        original_gradient_checkpointing = None
        if hasattr(self.model, 'gradient_checkpointing'):
            original_gradient_checkpointing = self.model.gradient_checkpointing
            self.model.gradient_checkpointing = False
        
        # Set to evaluation mode but enable gradients
        self.model.eval()
        torch.set_grad_enabled(True)
        
        for name, param in self.model.named_parameters():
            # Fix: use same parameter identification logic
            if param.requires_grad and 'lora_layer' in name:
                param_suffix = name.split('.')[-1]
                if param_suffix in ['G', 'U1', 'U2']:
                    param.requires_grad_(True)
                    new_fisher[name] = torch.zeros_like(param, device='cpu', dtype=torch.float32)
                    new_optimal[name] = param.data.clone().cpu().float()
        
        # Calculate samples per rank
        max_samples = min(50, len(dataloader))
        samples_per_rank = max_samples // world_size
        sample_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= samples_per_rank:
                break

            if batch is None:
                continue
            
            batch = self._prepare_batch(batch)

            if 'input_ids' not in batch or batch['input_ids'] is None:
                continue
            
            batch_size = batch['input_ids'].shape[0]
            
            for i in range(batch_size):
                if sample_count >= samples_per_rank:
                    break
                
                single_batch = self._create_single_batch(batch, i)
                
                try:
                    if sample_count % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    self.model.zero_grad()
                    
                    # Collect target parameters
                    target_params = []
                    for name, param in self.model.named_parameters():
                        if name in new_fisher:
                            param.requires_grad_(True)
                            target_params.append(param)
                    
                    outputs = self.model(**single_batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    
                    if loss is not None and loss.requires_grad and len(target_params) > 0:
                        # Manually compute gradients
                        grads = torch.autograd.grad(
                            loss,
                            target_params,
                            create_graph=False,
                            retain_graph=False,
                            allow_unused=True
                        )
                        
                        param_names = [n for n, p in self.model.named_parameters() if n in new_fisher]
                        grad_idx = 0
                        for name in param_names:
                            if grad_idx < len(grads) and grads[grad_idx] is not None:
                                new_fisher[name] += (grads[grad_idx].data.float() ** 2).cpu()
                                grad_idx += 1
                            
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM during Fisher update at sample {sample_count} on rank {rank}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"Error: {e}")
                        continue
                
                sample_count += 1
        
        # Restore original settings
        if original_gradient_checkpointing is not None:
            self.model.gradient_checkpointing = original_gradient_checkpointing
        
        torch.set_grad_enabled(original_grad_enabled)
        
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        
        # Normalize
        for name in new_fisher:
            new_fisher[name] /= max(sample_count, 1)
        
        # Distributed aggregation
        if dist.is_initialized():
            for name in new_fisher:
                fisher_tensor = new_fisher[name].to(self.device)
                dist.all_reduce(fisher_tensor, op=dist.ReduceOp.SUM)
                fisher_tensor /= world_size
                
                if name in self.fisher_dict:
                    self.fisher_dict[name] = alpha * self.fisher_dict[name] + (1 - alpha) * fisher_tensor
                else:
                    self.fisher_dict[name] = fisher_tensor
                
                self.optimal_params_dict[name] = new_optimal[name].to(self.device)
        else:
            for name in new_fisher:
                new_fisher[name] = new_fisher[name].to(self.device)
                
                if name in self.fisher_dict:
                    self.fisher_dict[name] = alpha * self.fisher_dict[name] + (1 - alpha) * new_fisher[name]
                else:
                    self.fisher_dict[name] = new_fisher[name]
                
                self.optimal_params_dict[name] = new_optimal[name].to(self.device)
        
        if rank == 0:
            print(f"Updated Fisher matrix (online) for {len(self.fisher_dict)} parameters with {sample_count * world_size} samples")
        
        torch.cuda.empty_cache()
    
    def compute_ewc_loss(self):
        """Compute EWC regularization loss - only for shared parameters"""
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            # Only compute EWC loss for parameters in Fisher dict (G, U1, U2)
            if name in self.fisher_dict:
                # Ensure on same device
                fisher = self.fisher_dict[name].to(param.device)
                optimal = self.optimal_params_dict[name].to(param.device)
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        # Add debug info
        if len(self.fisher_dict) == 0:
            print("Warning: Fisher dict is empty in compute_ewc_loss")
        
        return ewc_loss
    
    def save_fisher_matrix(self, path):
        """Save Fisher matrix to file"""
        # Move all content to CPU before saving
        save_dict = {
            'fisher': {k: v.cpu() for k, v in self.fisher_dict.items()},
            'optimal_params': {k: v.cpu() for k, v in self.optimal_params_dict.items()}
        }
        torch.save(save_dict, path)
        print(f"Fisher matrix saved to {path}")
    
    def load_fisher_matrix(self, path):
        """Load Fisher matrix from file"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            self.fisher_dict = {k: v.to(self.device) for k, v in checkpoint['fisher'].items()}
            self.optimal_params_dict = {k: v.to(self.device) for k, v in checkpoint['optimal_params'].items()}
            print(f"Fisher matrix loaded from {path}")


class RouteManager:
    """Hard routing manager - scene and environment selection based on image features"""
    def __init__(self, scene_num=5, env_num=4, device='cuda'):
        self.scene_num = scene_num
        self.env_num = env_num
        self.device = device
        
        # Predefined task ID to route mapping (as fallback)
        self.task_routes = {}
        for task_id in range(20):
            self.task_routes[task_id] = {
                'scene_idx': task_id % scene_num,
                'env_idx': task_id % env_num,
                'task_name': f'task_{task_id}'
            }
        
        # Try to import hard route selector
        self.hard_router = None
        try:
            from streamvln.model.hard_router import HardRouteSelector
            self.hard_router = HardRouteSelector(
                scene_num=scene_num,
                env_num=env_num,
                device=device
            )
            print("HardRouteSelector initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize HardRouteSelector: {e}")
            print("Falling back to predefined route mapping")
    
    def get_route_from_image(self, image):
        """Get route based on image features"""
        if self.hard_router is not None and image is not None:
            try:
                # Use hard route selector
                scene_idx, env_idx = self.hard_router.hard_route(image)
                return scene_idx, env_idx
            except Exception as e:
                print(f"Error in hard routing: {e}")
        
        # Fallback: return default route
        return 0, 0
    
    def get_route_from_task_id(self, task_id):
        """Get predefined route based on task ID (used during training)"""
        if task_id in self.task_routes:
            route = self.task_routes[task_id]
            return route['scene_idx'], route['env_idx']
        else:
            # Default route
            return task_id //(self.scene_num), task_id % self.env_num
    
    def save_config(self, path):
        """Save routing configuration"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.task_routes, f, indent=2)
    
    def load_config(self, path):
        """Load routing configuration"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                config = json.load(f)
                self.task_routes = {int(k): v for k, v in config.items()}


class ContinualLearningTrainer:
    """Continual learning trainer"""
    def __init__(self, model, training_args, device='cuda'):
        self.model = model
        self.training_args = training_args
        self.device = device
        self.route_manager = RouteManager(
            scene_num=training_args.tucker_scene_num,
            env_num=training_args.tucker_env_num,
            device=device
        )
        # EWC loss management
        self.ewc_loss = None        
        if training_args.continual_learning:
            self.ewc_loss = EWCLoss(model, device=device, mode=training_args.ewc_mode)
        
        # Load routing configuration
        if training_args.route_config_file:
            if os.path.exists(training_args.route_config_file):
                self.route_manager.load_config(training_args.route_config_file)
            else:
                # Create default routing configuration
                self._create_default_route_config()
                self.route_manager.save_config(training_args.route_config_file)
    
    def set_task_route(self, task_id, sample_image=None):
        """Set current task route"""
        if sample_image is not None and self.training_args.hard_route_selection:
            # Inference time: route selection based on image features
            scene_idx, env_idx = self.route_manager.get_route_from_image(sample_image)
        else:
            # Training time: use predefined task routes
            scene_idx, env_idx = self.route_manager.get_route_from_task_id(task_id)
        
        # Set routes for all Tucker layers in the model
        from streamvln.model.tucker_lora_layers import Tucker4DLoRALinear
        for name, module in self.model.named_modules():
            if isinstance(module, Tucker4DLoRALinear):
                module.set_route(scene_idx, env_idx, task_id)
        
        print(f"Task {task_id}: Scene {scene_idx}, Environment {env_idx}")
        return scene_idx, env_idx
    
    def apply_gradient_masks(self):
        """Apply gradient masks to all Tucker layers"""
        from streamvln.model.tucker_lora_layers import Tucker4DLoRALayer
        for name, module in self.model.named_modules():
            if isinstance(module, Tucker4DLoRALayer):
                module.apply_gradient_mask()
    
    def compute_total_loss(self, task_loss):
        """Compute total loss including EWC and orthogonal constraints"""
        total_loss = task_loss
        
        # Add EWC loss (only constrains shared parameters)
        if self.ewc_loss is not None and len(self.ewc_loss.fisher_dict) > 0:
            ewc_penalty = self.ewc_loss.compute_ewc_loss()
            total_loss = total_loss + self.training_args.ewc_lambda * ewc_penalty
        
        # Add orthogonal constraint loss (constrains U3 and U4)
        if self.training_args.use_orthogonal_reg:
            ortho_loss = 0.0
            from streamvln.model.tucker_lora_layers import Tucker4DLoRALayer
            for name, module in self.model.named_modules():
                if isinstance(module, Tucker4DLoRALayer):
                    ortho_loss += module.compute_orthogonal_loss()
            total_loss = total_loss + ortho_loss
        
        # Add stability loss
        if self.ewc_loss is not None and len(self.ewc_loss.trained_routes) > 0:
            stability_weight = getattr(self.training_args, 'stability_weight', 1.0)
            stability_loss = self.ewc_loss.compute_stability_loss(stability_weight)
            total_loss = total_loss + stability_loss
        
        return total_loss
    
    def update_fisher_after_task(self, dataloader):
        """Update Fisher information matrix after task completion"""
        if self.ewc_loss is not None:
            print("Updating Fisher Information Matrix...")
            self.ewc_loss.update_fisher_matrix(dataloader)
            
            # Save Fisher matrix
            if self.training_args.fisher_path:
                task_id = self.training_args.current_task_id
                fisher_path = os.path.join(
                    self.training_args.fisher_path, 
                    f"fisher_task_{task_id}.pt"
                )
                os.makedirs(os.path.dirname(fisher_path), exist_ok=True)
                self.ewc_loss.save_fisher_matrix(fisher_path)

    def _create_default_route_config(self):
        """Create default task-route mapping"""
        print("Creating default route configuration...")
        for task_id in range(self.training_args.num_tasks):
            # Use more reasonable allocation strategy
            scene_idx = task_id % 5    # Cycle through 5 environments
            env_idx = task_id % 4      # Cycle through 4 environments
            
            # Ensure within range
            scene_idx = min(scene_idx, self.training_args.tucker_scene_num - 1)
            
            self.route_manager.task_routes[task_id] = {
                'scene_idx': scene_idx,
                'env_idx': env_idx,
                'task_name': f'task_{task_id}',
                'data_path': f'task/Task_{task_id + 1}'
            }
    

    def freeze_non_active_parameters(self, scene_idx, env_idx):
        """Set active parameters and gradient masks"""
        from streamvln.model.tucker_lora_layers import Tucker4DLoRALayer
        
        for name, module in self.model.named_modules():
            if isinstance(module, Tucker4DLoRALayer):
                # First clean old hooks
                if hasattr(module, 'clear_hooks'):
                    module.clear_hooks()
                
                # Ensure all Tucker parameters are trainable
                module.G.requires_grad = True
                module.U1.requires_grad = True  
                module.U2.requires_grad = True
                module.U3.requires_grad = True
                module.U4.requires_grad = True
                
                # Set gradient masks (only affects specific rows of U3 and U4)
                module.U3_mask.fill_(False)
                module.U4_mask.fill_(False)
                module.U3_mask[scene_idx] = True
                module.U4_mask[env_idx] = True
                
                # Ensure mask is on correct device
                module.U3_mask = module.U3_mask.to(module.U3.device)
                module.U4_mask = module.U4_mask.to(module.U4.device)
                
                # Register new gradient hooks
                module.register_backward_hook()