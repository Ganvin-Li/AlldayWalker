import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import numpy as np

class Tucker4DLoRALayer(nn.Module):
    """4D Tucker decomposition LoRA layer with hard routing support for scenes and environments"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scene_num: int = 5,      # Number of scenes
        env_num: int = 4,        # Number of environments
        ranks: Tuple[int, int, int, int] = (16, 16, 8, 8),  # 4D ranks
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        init_scale: float = 0.02,
        dtype: torch.dtype = None,
        device: torch.device = None,
        use_orthogonal_reg: bool = True,  # Orthogonal constraint
        ortho_reg_weight: float = 0.01,   # Orthogonal constraint weight
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scene_num = scene_num
        self.env_num = env_num
        self.r1, self.r2, self.r3, self.r4 = ranks
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r1
        self.use_orthogonal_reg = use_orthogonal_reg
        self.ortho_reg_weight = ortho_reg_weight
        
        # Set dtype and device
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device if device is not None else torch.device('cpu')
        
        # 4D Tucker decomposition parameters
        # G: Core tensor (r1, r2, r3, r4)
        self.G = nn.Parameter(torch.empty(self.r1, self.r2, self.r3, self.r4, dtype=self.dtype, device=self.device))
        # U1: Output projection matrix (out_features, r1)
        self.U1 = nn.Parameter(torch.empty(out_features, self.r1, dtype=self.dtype, device=self.device))
        # U2: Input projection matrix (in_features, r2)  
        self.U2 = nn.Parameter(torch.empty(in_features, self.r2, dtype=self.dtype, device=self.device))
        # U3: Scene matrix (scene_num, r3)
        self.U3 = nn.Parameter(torch.empty(scene_num, self.r3, dtype=self.dtype, device=self.device))
        # U4: Environment matrix (env_num, r4)
        self.U4 = nn.Parameter(torch.empty(env_num, self.r4, dtype=self.dtype, device=self.device))
        
        # Scene and environment masks to freeze unused rows
        self.register_buffer('U3_mask', torch.ones(scene_num, dtype=torch.bool, device=self.device))
        self.register_buffer('U4_mask', torch.ones(env_num, dtype=torch.bool, device=self.device))
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Task routing record (for continual learning)
        self.task_routes = {}  # {task_id: (scene_idx, env_idx)}
        
        # Initialize parameters
        self.reset_parameters(init_scale)

        # Currently active routing indices
        self.active_scene_idx = None
        self.active_env_idx = None
        
    def reset_parameters(self, scale=0.02):
        """Initialize Tucker decomposition parameters
        - G, U1, U2: Kaiming initialization
        - U3, U4: Zero initialization
        """
        # Kaiming initialization for G, U1, U2
        # For G core tensor, use Kaiming uniform initialization
        fan_in = self.r2 * self.r3 * self.r4  # Product of input dimensions
        fan_out = self.r1 * self.r3 * self.r4  # Product of output dimensions
        
        # Kaiming initialization standard deviation calculation
        kaiming_std = math.sqrt(2.0 / (fan_in + fan_out))
        
        # G uses Kaiming initialization
        nn.init.normal_(self.G, mean=0.0, std=kaiming_std)
        
        # U1 Kaiming initialization (out_features, r1)
        fan_in_u1 = self.r1
        fan_out_u1 = self.out_features
        kaiming_std_u1 = math.sqrt(2.0 / (fan_in_u1 + fan_out_u1))
        nn.init.normal_(self.U1, mean=0.0, std=kaiming_std_u1)
        
        # U2 Kaiming initialization (in_features, r2)
        fan_in_u2 = self.in_features
        fan_out_u2 = self.r2
        kaiming_std_u2 = math.sqrt(2.0 / (fan_in_u2 + fan_out_u2))
        nn.init.normal_(self.U2, mean=0.0, std=kaiming_std_u2)
        
        # U3 and U4 use near-zero initialization
        nn.init.normal_(self.U3, mean=0.0, std=scale * 0.1)
        nn.init.normal_(self.U4, mean=0.0, std=scale * 0.1)
    
    def set_active_route(self, scene_idx: int, env_idx: int, task_id: Optional[int] = None):
        """Set currently active route (for freezing other rows during training)"""
        # Record currently active indices
        self.active_scene_idx = scene_idx
        self.active_env_idx = env_idx
        
        # Reset all masks
        self.U3_mask.fill_(False)
        self.U4_mask.fill_(False)
        
        # Only activate current route
        self.U3_mask[scene_idx] = True
        self.U4_mask[env_idx] = True
        
        # Record task route
        if task_id is not None:
            self.task_routes[task_id] = (scene_idx, env_idx)
    
    def zero_inactive_gradients(self):
        """Zero gradients of inactive rows (called before optimizer step)"""
        if self.U3.grad is not None and self.active_scene_idx is not None:
            # Create gradient mask
            with torch.no_grad():
                # Keep active row gradients, zero others
                for i in range(self.scene_num):
                    if i != self.active_scene_idx:
                        self.U3.grad.data[i].zero_()
        
        if self.U4.grad is not None and self.active_env_idx is not None:
            # Create gradient mask
            with torch.no_grad():
                # Keep active row gradients, zero others
                for i in range(self.env_num):
                    if i != self.active_env_idx:
                        self.U4.grad.data[i].zero_()
    
    def compute_orthogonal_loss(self) -> torch.Tensor:
        """Compute orthogonal constraint loss for U3 and U4"""
        ortho_loss = 0.0
        
        if self.use_orthogonal_reg:
            # Only compute orthogonal constraint for active rows
            active_U3_indices = self.U3_mask.nonzero(as_tuple=True)[0]
            active_U4_indices = self.U4_mask.nonzero(as_tuple=True)[0]
            
            # U3 orthogonal constraint (only for active rows)
            if len(active_U3_indices) > 1:
                active_U3 = self.U3[active_U3_indices]
                U3_gram = torch.mm(active_U3.t(), active_U3)
                U3_eye = torch.eye(self.r3, device=self.device, dtype=self.dtype)
                ortho_loss += torch.norm(U3_gram - U3_eye, p='fro')
            
            # U4 orthogonal constraint (only for active rows)
            if len(active_U4_indices) > 1:
                active_U4 = self.U4[active_U4_indices]
                U4_gram = torch.mm(active_U4.t(), active_U4)
                U4_eye = torch.eye(self.r4, device=self.device, dtype=self.dtype)
                ortho_loss += torch.norm(U4_gram - U4_eye, p='fro')
        
        return ortho_loss * self.ortho_reg_weight
    
    def forward(self, x: torch.Tensor, scene_idx: int = None, env_idx: int = None) -> torch.Tensor:
        """
        4D Tucker-LoRA forward propagation
        Args:
            x: Input tensor (batch_size, seq_len, in_features) or (batch_size, in_features)
            scene_idx: Scene index
            env_idx: Environment index
        Returns:
            LoRA output
        """
        # If no indices provided, use default values or currently active indices
        if scene_idx is None:
            scene_idx = self.active_scene_idx if self.active_scene_idx is not None else 0
        if env_idx is None:
            env_idx = self.active_env_idx if self.active_env_idx is not None else 0
            
        # Add index range check
        scene_idx = min(max(0, scene_idx), self.scene_num - 1)
        env_idx = min(max(0, env_idx), self.env_num - 1)

        batch_dims = x.shape[:-1]
        x_2d = x.view(-1, self.in_features)
        
        # Ensure all tensors have consistent types - critical fix
        # Convert all parameters to input dtype
        x_dtype = x.dtype
        x_device = x.device
        
        # Convert parameters to correct dtype and device
        U2 = self.U2.to(dtype=x_dtype, device=x_device)
        U4 = self.U4.to(dtype=x_dtype, device=x_device)
        U3 = self.U3.to(dtype=x_dtype, device=x_device)
        G = self.G.to(dtype=x_dtype, device=x_device)
        U1 = self.U1.to(dtype=x_dtype, device=x_device)
        
        # Apply dropout
        x_2d = self.lora_dropout(x_2d)
        
        # 4D Tucker computation
        # Step 1: Contract along mode-4 (environment)
        G_env = torch.einsum('ijkl,l->ijk', G, U4[env_idx])
        
        # Step 2: Contract along mode-3 (scene)
        G_scene_env = torch.einsum('ijk,k->ij', G_env, U3[scene_idx])
        
        # Step 3: Input projection
        v = x_2d @ U2
        
        # Step 4: Core transformation
        u = v @ G_scene_env.t()
        
        # Step 5: Output projection
        output = u @ U1.t()
        
        # Apply scaling
        output = output * self.scaling
        
        # Reshape to match input dimensions
        output = output.view(*batch_dims, self.out_features)
        
        return output


class Tucker4DLoRALinear(nn.Module):
    """Linear layer with 4D Tucker-LoRA"""
    def __init__(
    self,
    base_layer: nn.Linear,
    scene_num: int = 5,
    env_num: int = 4,
    ranks: Tuple[int, int, int, int] = (16, 16, 8, 8),
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    init_scale: float = 0.02,
    use_orthogonal_reg: bool = True,
    ortho_reg_weight: float = 0.01,
):
        super().__init__()
        self.base_layer = base_layer
        
        # Get dtype and device from base layer
        base_dtype = base_layer.weight.dtype
        base_device = base_layer.weight.device
        
        # Create Tucker-LoRA layer with matching dtype and device
        self.lora_layer = Tucker4DLoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            scene_num=scene_num,
            env_num=env_num,
            ranks=ranks,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_scale=init_scale,
            dtype=base_dtype,
            device=base_device,
            use_orthogonal_reg=use_orthogonal_reg,
            ortho_reg_weight=ortho_reg_weight,
        )
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Ensure Tucker-LoRA parameters are trainable
        for param in self.lora_layer.parameters():
            param.requires_grad = True
        
        # Currently active route
        self.current_scene_idx = None
        self.current_env_idx = None
    
    def set_route(self, scene_idx: int, env_idx: int, task_id: Optional[int] = None):
        """Set current route"""
        self.current_scene_idx = scene_idx
        self.current_env_idx = env_idx
        self.lora_layer.set_active_route(scene_idx, env_idx, task_id)
    
    @property
    def in_features(self):
        return self.base_layer.in_features
    
    @property
    def out_features(self):
        return self.base_layer.out_features
    
    @property
    def weight(self):
        return self.base_layer.weight
    
    @property
    def bias(self):
        return self.base_layer.bias
            
    def forward(self, x: torch.Tensor, scene_idx: Optional[int] = None, env_idx: Optional[int] = None) -> torch.Tensor:
        """
        Forward propagation
        Args:
            x: Input tensor
            scene_idx: Scene index (if None, use currently set index)
            env_idx: Environment index (if None, use currently set index)
        """
        # Use provided indices or currently set indices
        if scene_idx is None:
            scene_idx = self.current_scene_idx if self.current_scene_idx is not None else 0
        if env_idx is None:
            env_idx = self.current_env_idx if self.current_env_idx is not None else 0
        
        # Base forward pass
        base_output = self.base_layer(x)
        
        # Tucker-LoRA forward pass
        lora_output = self.lora_layer(x, scene_idx, env_idx)
        
        # Combine outputs
        return base_output + lora_output.to(base_output.dtype)