import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 先导入habitat-sim，再导入自定义噪声模型
import habitat_sim
from habitat_extensions import AtmosphericScatteringNoiseModel, AtmosphericScatteringPresets
from habitat_extensions import LowLightNoiseModel, LowLightPresets
from habitat_extensions import OverexposureNoiseModel, OverexposurePresets

# 验证注册并手动注册
try:
    if "AtmosphericScatteringNoiseModel" not in habitat_sim.registry._get_noise_model_registry():
        habitat_sim.registry.register_noise_model(AtmosphericScatteringNoiseModel)
    if "LowLightNoiseModel" not in habitat_sim.registry._get_noise_model_registry():
        habitat_sim.registry.register_noise_model(LowLightNoiseModel)
    if "OverexposureNoiseModel" not in habitat_sim.registry._get_noise_model_registry():
        habitat_sim.registry.register_noise_model(OverexposureNoiseModel)
except:
    pass

import re
import tqdm
import torch
import copy
import json
import random
import argparse
import itertools
import quaternion
import transformers
import numpy as np

from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from depth_camera_filtering import filter_depth
from transformers.image_utils import to_numpy_array

import habitat
from habitat import logger, Env
# from habitat_extensions import measures
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from model.stream_video_vln import StreamVLNForCausalLM
from utils.utils import dict_to_cuda
from utils.dist import *
from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN

import torch.nn as nn
from pathlib import Path

class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        tokenizer: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        # 添加Tucker-LoRA专家权重支持
        self.expert_weights = None
        if hasattr(args, 'expert_weights') and args.expert_weights:
            self.expert_weights = torch.tensor([
                float(w) for w in args.expert_weights.split(',')
            ], device=self.device)
            print(f"Using expert weights: {self.expert_weights}")

        with habitat.config.read_write(self.config):
            # self.config.habitat.task.measurements.success.success_distance=3.0
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        self.image_processor = model.get_vision_tower().image_processor
        self.model = model
        self.tokenizer = tokenizer
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })
        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]
        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history

        # 添加4D Tucker-LoRA硬路由支持
        self.use_hard_routing = getattr(args, 'use_hard_routing', False)
        self.scene_idx = getattr(args, 'scene_idx', None)
        self.env_idx = getattr(args, 'env_idx', None)
        
        # 如果使用硬路由，设置模型路由
        if self.use_hard_routing and self.scene_idx is not None and self.env_idx is not None:
            self.set_model_hard_route(model, self.scene_idx, self.env_idx)
            print(f"Hard routing set: Scene {self.scene_idx}, Environment {self.env_idx}")

    def set_model_hard_route(self, model, scene_idx, env_idx):
        """Set hard routing for all Tucker4DLoRALinear layers in the model"""
        from streamvln.model.tucker_lora_layers import Tucker4DLoRALinear
        
        # 验证索引有效性
        if hasattr(model, 'tucker_scene_num') and hasattr(model, 'tucker_env_num'):
            if scene_idx >= model.tucker_scene_num:
                raise ValueError(f"scene_idx {scene_idx} exceeds maximum {model.tucker_scene_num-1}")
            if env_idx >= model.tucker_env_num:
                raise ValueError(f"env_idx {env_idx} exceeds maximum {model.tucker_env_num-1}")
        
        for name, module in model.named_modules():
            if isinstance(module, Tucker4DLoRALinear):
                # 设置活跃路由
                module.set_route(scene_idx, env_idx)
                
                # 确保只有指定的U3和U4行是活跃的
                module.lora_layer.active_scene_idx = scene_idx
                module.lora_layer.active_env_idx = env_idx
                
                # 更新mask - 这里假设U3对应场景，U4对应环境
                module.lora_layer.U3_mask = torch.zeros(model.tucker_scene_num, dtype=torch.bool)
                module.lora_layer.U4_mask = torch.zeros(model.tucker_env_num, dtype=torch.bool)
                module.lora_layer.U3_mask[scene_idx] = True
                module.lora_layer.U4_mask[env_idx] = True
        
        print(f"Model hard routing configured for Scene {scene_idx}, Env {env_idx}")
        print(f"U3_mask: only index {scene_idx} is active")
        print(f"U4_mask: only index {env_idx} is active")
        
    
    def preprocess_depth_image(self, depth_image, do_depth_scale=True, depth_scale=1000):
        target_height = self.image_processor.crop_size['height']  # 384
        target_width  = self.image_processor.crop_size['width']  # 384
        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)
        
        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale
    
        return img, (target_width, target_height)
    
    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [ 0.0, fy, cy, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]
        ])
        return intrinsic_matrix
    
    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):  # (V, 4, 4) (resize_shape) (h, w)
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]  # (1, 4, 4) or (B, 4, 4)
        
        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic
    
    def get_axis_align_matrix(self):
        # ma = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ma = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()
        return ma
    
    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env

    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        intrinsic_matrix = self.get_intrinsic_matrix(self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])
                        
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start",episode_instruction)
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue
                self.model.reset_for_env(idx)
                env.current_episode = episode
                observations = env.reset()
                
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg'))
                
                vis_frames = []
                step_id = 0
                
                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}'), exist_ok=True)
                initial_height = env.sim.get_agent_state().position[1]

                rgb_list = []
                depth_list = []
                depth_images_list = []
                pose_list = []
                intrinsic_list = []
                time_ids = []
                action_seq = []
                past_key_values = None
                output_ids = None
                while not env.episode_over:
                    self.model.eval()
                    time_ids.append(step_id)
                    rgb = observations["rgb"]
                    depth = observations["depth"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000

                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height # Habitat GPS makes west negative, so flip y
                    camera_position = np.array([x, -y, self._camera_height + height])
                    robot_xy = camera_position[:2]
                    tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
                    
                    rotation = agent_state.rotation
                    translation = agent_state.position
                    rotation_matrix = quaternion.as_rotation_matrix(rotation)
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = rotation_matrix
                    transformation_matrix[:3, 3] = translation
                    
                    image = Image.fromarray(rgb).convert('RGB')
                    image_size = image.size
                    # image = self.image_processor.preprocess(images=image, do_rescale=True, do_normalize=True, return_tensors='pt')['pixel_values'][0]
                    image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
                    depth_image, resize_shape = self.preprocess_depth_image(Image.fromarray(depth.astype(np.uint16), mode='I;16'), do_depth_scale=True)
                    
                    intrinsic = self.preprocess_instrinsic(intrinsic_matrix, image_size, resize_shape)
                    intrinsic = torch.from_numpy(intrinsic).float()
    
                    rgb_list.append(image)
                    depth_list.append(torch.from_numpy(depth_image).float())
                    pose_list.append(torch.from_numpy(tf_camera_to_episodic) @ self.get_axis_align_matrix())
                    intrinsic_list.append(intrinsic)
                    
                    info = env.get_metrics()
                    if info['top_down_map'] is not None:
                        frame = observations_to_image({'rgb':observations['rgb']}, info)
                        vis_frames.append(frame)
                    # import ipdb; ipdb.set_trace()
                    if len(action_seq) == 0:
                        if output_ids is None:
                            sources = copy.deepcopy(self.conversation)
                            sources[0]["value"] = sources[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                            if step_id != 0 :
                                sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
                            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', episode.instruction.instruction_text)
                            add_system = True
                            print(step_id, sources[0]["value"])
                        else:
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            add_system = False
                            
                        input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
                        if output_ids is not None:
                            input_ids = torch.cat([output_ids,input_ids.to(output_ids.device)], dim=1)

                        images = rgb_list[-1:]
                        depths = depth_list[-1:]
                        poses = pose_list[-1:]
                        intrinsics = intrinsic_list[-1:]
                        # import ipdb; ipdb.set_trace()
                        if step_id != 0 and step_id % self.num_frames == 0:
                            if self.num_history is None:
                                history_ids = slice(0, time_ids[0], self.num_future_steps)
                            else:
                                history_ids = slice(0, time_ids[0], (time_ids[0] // self.num_history))
                            images = rgb_list[history_ids] + images
                            depths = depth_list[history_ids] + depths
                            poses = pose_list[history_ids] + poses
                            intrinsics = intrinsic_list[history_ids] + intrinsics
                                
                        input_dict = {'images':torch.stack(images).unsqueeze(0), 'depths':torch.stack(depths).unsqueeze(0), \
                                        'poses':torch.stack(poses).unsqueeze(0), 'intrinsics':torch.stack(intrinsics).unsqueeze(0), 'inputs':input_ids, 'env_id':idx, 'time_ids':[time_ids],'task_type':[0]}
                            
                        input_dict = dict_to_cuda(input_dict, self.device)
                        
                        for key, value in input_dict.items():
                            if key in ['images', 'depths', 'poses', 'intrinsics']:
                                input_dict[key] = input_dict[key].to(torch.bfloat16)
                        
                        outputs = self.model.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=past_key_values)
                        
                        output_ids = outputs.sequences
                        past_key_values = outputs.past_key_values
                        llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                        print(llm_outputs, flush=True)
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)
                        if len(action_seq) == 0: ## if generated llm without Specific values
                            action_seq = [0]
                    action = action_seq.pop(0)
                    
                    observations = env.step(action)
                    step_id += 1
                    if step_id % self.num_frames == 0:
                        self.model.reset_for_env(idx)
                        output_ids = None
                        past_key_values = None
                        time_ids = []
                        
                process_bar.update(1)
                # episode_id += 1
                metrics = env.get_metrics()
                if self.save_video:
                    images_to_video(
                        vis_frames, os.path.join(self.output_path, f'vis_{self.epoch}'), f'{scene_id}_{episode_id}', fps=6, quality=9
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])

                # 临时添加
                if 'oracle_success' not in metrics:
                    metrics['oracle_success'] = metrics.get('success', 0)
                if 'oracle_navigation_error' not in metrics:
                    metrics['oracle_navigation_error'] = 0.0
                    
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }
                
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)     

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)



    def preprocess_qwen(self, sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.",add_system: bool = False):
        # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        roles = {"human": "user", "gpt": "assistant"}
        # import ipdb; ipdb.set_trace()
        # Add image tokens to tokenizer as a special tokens
        # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
        tokenizer = copy.deepcopy(tokenizer)
        # When there is actually an image, we add the image tokens as a special token
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
        unmask_tokens_idx =  [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids

        # Reset Qwen chat templates so that it won't include system message every time we apply
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # _system = tokenizer("system").input_ids + nl_tokens
        # _user = tokenizer("user").input_ids + nl_tokens
        # _assistant = tokenizer("assistant").input_ids + nl_tokens

        # Apply prompt templates
        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else: 
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                # Skip the first one if it is not from human
                source = source[1:]

            input_id, target = [], []

            # import ipdb; ipdb.set_trace()
            # New version, use apply chat template
            # Build system message for each sentence
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            for conv in source:
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role =  roles.get(role, role)
                
                conv = [{"role" : role, "content" : content}]
                # import ipdb; ipdb.set_trace()
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id
            

            # assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
                    
            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids,  conversations # tensor(bs x seq_len)

def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output
   
def load_model_and_tokenizer(args):
    """根据参数加载模型和tokenizer，支持Tucker-LoRA"""
    import json
    
    # 检查各种LoRA配置
    tucker_config_path = os.path.join(args.model_path, "tucker_lora_config.json")
    separated_config_path = os.path.join(args.model_path, "lora_separated_config.json")
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    
    is_tucker_lora = os.path.exists(tucker_config_path)
    is_separated_lora = os.path.exists(separated_config_path) and not is_tucker_lora
    is_standard_lora = os.path.exists(adapter_config_path) and not is_separated_lora and not is_tucker_lora
    
    print(f"Model path: {args.model_path}")
    print(f"Base model path: {args.base_model_path if hasattr(args, 'base_model_path') else 'Not specified'}")
    print(f"Is Tucker-LoRA model: {is_tucker_lora}")
    print(f"Is standard LoRA model: {is_standard_lora}")
    print(f"Is separated LoRA model: {is_separated_lora}")
    
    if is_tucker_lora:
        print("=" * 50)
        print("LOADING TUCKER-LORA MODEL")
        print("=" * 50)
        
        # Load Tucker-LoRA configuration
        with open(tucker_config_path, 'r') as f:
            tucker_config = json.load(f)
        
        # Get base model path
        if hasattr(args, 'base_model_path') and args.base_model_path:
            base_model_path = args.base_model_path
        else:
            base_model_path = tucker_config.get('base_model_path')
        
        if not base_model_path:
            raise ValueError("Base model path not specified. Please provide --base_model_path")
        
        print(f"Loading base model from: {base_model_path}")
        
        # Load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            model_max_length=getattr(args, 'model_max_length', 2048),
            padding_side="right"
        )
        
        # Load base model config
        config = transformers.AutoConfig.from_pretrained(base_model_path)
        
        # Load base model
        print("Loading base model...")
        model = StreamVLNForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None  # 我们会手动管理设备
        )
        
        # Load Tucker-LoRA weights
        print("Loading Tucker-LoRA adapter...")
        tucker_dir = tucker_config.get('tucker_dir', os.path.join(args.model_path, 'tucker_lora'))
        
        # 支持命令行覆盖Tucker目录
        if hasattr(args, 'tucker_dir') and args.tucker_dir:
            tucker_dir = args.tucker_dir
            print(f"Using command-line specified Tucker directory: {tucker_dir}")
        
        model = load_tucker_lora_model(model, tucker_config_path, tucker_dir)
        
        print("Tucker-LoRA model loaded successfully!")
        model.eval()
        
        # 设置Tucker-LoRA标志
        model.is_tucker_lora = True
        
        return model, tokenizer, config, True
    
    elif is_separated_lora:
        print("=" * 50)
        print("LOADING SEPARATED LORA MODEL")
        print("=" * 50)
        
        # 原有的分离LoRA加载逻辑保持不变
        import json
        with open(separated_config_path, 'r') as f:
            separated_config = json.load(f)
        
        if hasattr(args, 'lora_A_dir') and args.lora_A_dir:
            lora_A_dir = args.lora_A_dir
            print(f"Using command-line specified LoRA A directory: {lora_A_dir}")
        else:
            lora_A_dir = separated_config['separated_storage']['lora_A_dir']
            print(f"Using config-specified LoRA A directory: {lora_A_dir}")
            
        if hasattr(args, 'lora_B_dir') and args.lora_B_dir:
            lora_B_dir = args.lora_B_dir
            print(f"Using command-line specified LoRA B directory: {lora_B_dir}")
        else:
            lora_B_dir = separated_config['separated_storage']['lora_B_dir']
            print(f"Using config-specified LoRA B directory: {lora_B_dir}")
        
        if not os.path.exists(lora_A_dir):
            raise ValueError(f"LoRA A directory not found: {lora_A_dir}")
        if not os.path.exists(lora_B_dir):
            raise ValueError(f"LoRA B directory not found: {lora_B_dir}")
        
        if hasattr(args, 'base_model_path') and args.base_model_path:
            base_model_path = args.base_model_path
        else:
            base_model_path = separated_config['base_model_path']
        
        print(f"Loading base model from: {base_model_path}")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            model_max_length=getattr(args, 'model_max_length', 2048),
            padding_side="right"
        )
        
        config = transformers.AutoConfig.from_pretrained(base_model_path)
        
        print("Loading base model...")
        model = StreamVLNForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None
        )
        
        print("Initializing custom LoRA structure...")
        model = load_explicit_lora_model(model, separated_config_path, lora_A_dir, lora_B_dir)
        
        print("Custom LoRA model loaded successfully!")
        model.eval()
        
        return model, tokenizer, config, True
    
    elif is_standard_lora:
        print("=" * 50)
        print("LOADING STANDARD LORA MODEL")
        print("=" * 50)
        
        # 标准LoRA加载流程
        import json
        adapter_config_file = os.path.join(args.model_path, "adapter_config.json")
        with open(adapter_config_file, 'r') as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get('base_model_name_or_path')
        
        if hasattr(args, 'base_model_path') and args.base_model_path:
            actual_base_path = args.base_model_path
            print(f"Using provided base model path: {actual_base_path}")
        else:
            actual_base_path = base_model_path
            print(f"Using base model path from LoRA config: {actual_base_path}")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            actual_base_path,
            model_max_length=getattr(args, 'model_max_length', 2048),
            padding_side="right"
        )
        
        config = transformers.AutoConfig.from_pretrained(actual_base_path)
        
        print("Loading base model...")
        model = StreamVLNForCausalLM.from_pretrained(
            actual_base_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None
        )
        
        print("Loading standard LoRA adapter with custom implementation...")
        lora_A_dir = args.model_path
        lora_B_dir = args.model_path
        model = load_explicit_lora_model(model, adapter_config_file, lora_A_dir, lora_B_dir)
        
        print("Standard LoRA adapter loaded successfully!")
        model.eval()
        
        return model, tokenizer, config, True
    
    else:
        print("=" * 50)
        print("LOADING BASE MODEL")
        print("=" * 50)
        
        # 加载基础模型
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_path,
            model_max_length=getattr(args, 'model_max_length', 2048),
            padding_side="right"
        )
        
        config = transformers.AutoConfig.from_pretrained(args.model_path)
        
        model = StreamVLNForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=False,
            local_files_only=True
        )
        
        print("Base model loaded successfully!")
        model.eval()
        
        return model, tokenizer, config, False

def load_tucker_lora_model(model, tucker_config_path, tucker_weights_dir):
    """Load Tucker-LoRA weights into model - 支持4D版本"""
    import json
    
    # Load configuration
    with open(tucker_config_path, 'r') as f:
        config = json.load(f)
    
    # 检测是否是4D版本
    is_4d_version = 'tucker_4d_config' in config or 'scene_num' in config.get('tucker_lora_config', {})
    
    if is_4d_version:
        # 4D Tucker-LoRA
        from streamvln.model.tucker_lora_layers import Tucker4DLoRALinear
        print("Loading 4D Tucker-LoRA model")
        
        # 获取配置 - 兼容不同的配置结构
        if 'tucker_4d_config' in config:
            tucker_config = config['tucker_4d_config']
        else:
            tucker_config = config.get('tucker_lora_config', config)
        
        # 获取目标模块 - 提供默认值
        if 'target_modules' in tucker_config:
            if isinstance(tucker_config['target_modules'], str):
                target_modules = tucker_config['target_modules'].split(',')
            else:
                target_modules = tucker_config['target_modules']
        else:
            # 默认的目标模块列表
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                            'gate_proj', 'up_proj', 'down_proj']
            print(f"Using default target modules: {target_modules}")
        
        # 获取4D特定参数
        scene_num = tucker_config.get('scene_num', 5)
        env_num = tucker_config.get('env_num', 4) 
        ranks_4d = tucker_config.get('ranks_4d', tucker_config.get('ranks', [16, 16, 8, 8]))
        if isinstance(ranks_4d, str):
            ranks_4d = [int(r) for r in ranks_4d.split(',')]
        
        lora_alpha = tucker_config.get('alpha', tucker_config.get('lora_alpha', 32))
        dropout = tucker_config.get('dropout', tucker_config.get('lora_dropout', 0))
        init_scale = tucker_config.get('init_scale', 0.02)
        use_orthogonal_reg = tucker_config.get('use_orthogonal_reg', True)
        ortho_reg_weight = tucker_config.get('ortho_reg_weight', 0.01)
        
        # 获取模型数据类型
        model_dtype = next(model.parameters()).dtype
        print(f"Model dtype: {model_dtype}")
        print(f"4D Tucker config - Scene: {scene_num}, Env: {env_num}, Ranks: {ranks_4d}")
        
        # 替换目标层
        replaced_count = 0
        replaced_modules = []
        
        for name, module in list(model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                # 检查是否是目标模块
                if any(target in name for target in target_modules):
                    *parent_path, layer_name = name.split('.')
                    parent = model
                    for p in parent_path:
                        parent = getattr(parent, p)
                    
                    # Create 4D Tucker-LoRA layer
                    tucker_lora_linear = Tucker4DLoRALinear(
                        module,
                        scene_num=scene_num,
                        env_num=env_num,
                        ranks=tuple(ranks_4d),
                        lora_alpha=lora_alpha,
                        lora_dropout=dropout,
                        init_scale=init_scale,
                        use_orthogonal_reg=use_orthogonal_reg,
                        ortho_reg_weight=ortho_reg_weight,
                    )
                    
                    # 确保权重使用正确的dtype
                    tucker_lora_linear.lora_layer.U1.data = tucker_lora_linear.lora_layer.U1.data.to(model_dtype)
                    tucker_lora_linear.lora_layer.U2.data = tucker_lora_linear.lora_layer.U2.data.to(model_dtype)
                    tucker_lora_linear.lora_layer.U3.data = tucker_lora_linear.lora_layer.U3.data.to(model_dtype)
                    tucker_lora_linear.lora_layer.U4.data = tucker_lora_linear.lora_layer.U4.data.to(model_dtype)
                    tucker_lora_linear.lora_layer.G.data = tucker_lora_linear.lora_layer.G.data.to(model_dtype)
                    
                    setattr(parent, layer_name, tucker_lora_linear)
                    replaced_count += 1
                    replaced_modules.append(name)
        
    else:
        # 标准3D Tucker-LoRA (保留原有代码以兼容)
        from streamvln.model.tucker_lora_layers import TuckerLoRALinear
        print("Loading 3D Tucker-LoRA model")
        
        tucker_config = config['tucker_lora_config']
        target_modules = tucker_config['target_modules'].split(',')
        
        model_dtype = next(model.parameters()).dtype
        print(f"Model dtype: {model_dtype}")
        
        replaced_count = 0
        replaced_modules = []
        
        for name, module in list(model.named_modules()):
            if isinstance(module, torch.nn.Linear):
                if any(target in name for target in target_modules):
                    *parent_path, layer_name = name.split('.')
                    parent = model
                    for p in parent_path:
                        parent = getattr(parent, p)
                    
                    tucker_lora_linear = TuckerLoRALinear(
                        module,
                        num_experts=tucker_config['num_experts'],
                        ranks=tuple(tucker_config['ranks']),
                        lora_alpha=tucker_config['alpha'],
                        lora_dropout=tucker_config.get('dropout', 0),
                        init_scale=tucker_config.get('init_scale', 0.02),
                    )
                    
                    tucker_lora_linear.lora_layer.U1.data = tucker_lora_linear.lora_layer.U1.data.to(model_dtype)
                    tucker_lora_linear.lora_layer.U2.data = tucker_lora_linear.lora_layer.U2.data.to(model_dtype)
                    tucker_lora_linear.lora_layer.U3.data = tucker_lora_linear.lora_layer.U3.data.to(model_dtype)
                    tucker_lora_linear.lora_layer.G.data = tucker_lora_linear.lora_layer.G.data.to(model_dtype)
                    
                    setattr(parent, layer_name, tucker_lora_linear)
                    replaced_count += 1
                    replaced_modules.append(name)
    
    print(f"Replaced {replaced_count} layers with {'4D' if is_4d_version else '3D'} Tucker-LoRA")
    print(f"Replaced modules: {replaced_modules[:5]}..." if len(replaced_modules) > 5 else replaced_modules)
    
    # Load weights - 多种路径尝试
    tucker_weights_path = None
    possible_paths = [
        os.path.join(tucker_weights_dir, "tucker_lora_weights.bin"),
        os.path.join(tucker_weights_dir, "tucker_lora_weights.pt"),
        os.path.join(os.path.dirname(tucker_config_path), "tucker_lora", "tucker_lora_weights.bin"),
        os.path.join(os.path.dirname(tucker_config_path), "tucker_lora", "tucker_lora_weights.pt"),
        os.path.join(os.path.dirname(tucker_config_path), "tucker_lora_weights.bin"),
        os.path.join(os.path.dirname(tucker_config_path), "tucker_lora_weights.pt"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            tucker_weights_path = path
            break
    
    if tucker_weights_path is None:
        print(f"Warning: Tucker-LoRA weights not found. Tried paths: {possible_paths}")
        print("Model will use randomly initialized Tucker-LoRA weights")
        return model
    
    print(f"Loading Tucker-LoRA weights from: {tucker_weights_path}")
    tucker_weights = torch.load(tucker_weights_path, map_location='cpu')
    
    # Apply weights
    loaded_params = 0
    missing_keys = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_layer'):
            # 尝试多种可能的键格式
            param_names = ['U1', 'U2', 'G'] + (['U3', 'U4'] if is_4d_version else ['U3'])
            
            for param_name in param_names:
                possible_keys = [
                    f"{name}.tucker_lora.{param_name}",
                    f"{name}.lora_layer.{param_name}",
                    f"{name}.{param_name}",
                ]
                
                loaded = False
                for key in possible_keys:
                    if key in tucker_weights:
                        param_tensor = tucker_weights[key].to(
                            device=getattr(module.lora_layer, param_name).device,
                            dtype=model_dtype
                        )
                        setattr(module.lora_layer, param_name, 
                               torch.nn.Parameter(param_tensor))
                        loaded = True
                        loaded_params += 1
                        break
                
                if not loaded and param_name in ['U1', 'U2', 'G']:  # 只对必需参数报警
                    missing_keys.append(f"{name}.{param_name}")
    
    print(f"Loaded {loaded_params} Tucker-LoRA parameter groups")
    if missing_keys:
        print(f"Warning: Missing required keys: {missing_keys[:10]}...")
    
    # 设置4D Tucker-LoRA标志
    if is_4d_version:
        model.is_tucker_4d = True
        model.tucker_scene_num = scene_num
        model.tucker_env_num = env_num
    
    return model

    
def load_explicit_lora_model(model, lora_config_path, lora_A_dir, lora_B_dir):
    """Load explicit LoRA weights into model"""
    from streamvln.model.lora_layers import LoRALinear
    
    # Load configuration
    with open(lora_config_path, 'r') as f:
        config = json.load(f)
    
    lora_config = config['lora_config']
    target_modules = lora_config['target_modules'].split(',')
    
    # 获取模型的目标数据类型
    model_dtype = next(model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")
    
    # Apply LoRA structure to model
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            if any(target in name for target in target_modules):
                *parent_path, layer_name = name.split('.')
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)
                
                # Create LoRA layer
                lora_linear = LoRALinear(
                    module,
                    r=lora_config['r'],
                    lora_alpha=lora_config['alpha'],
                    lora_dropout=lora_config.get('dropout', 0)
                )
                
                # 确保LoRA权重使用正确的数据类型
                lora_linear.lora_layer.lora_A.data = lora_linear.lora_layer.lora_A.data.to(model_dtype)
                lora_linear.lora_layer.lora_B.data = lora_linear.lora_layer.lora_B.data.to(model_dtype)
                
                setattr(parent, layer_name, lora_linear)
    
    # Load weights
    lora_A_weights = torch.load(os.path.join(lora_A_dir, "adapter_model_A.bin"), map_location='cpu')
    lora_B_weights = torch.load(os.path.join(lora_B_dir, "adapter_model_B.bin"), map_location='cpu')
    
    # Apply weights with proper dtype handling
    for name, module in model.named_modules():
        if hasattr(module, 'lora_layer'):
            for key, value in lora_A_weights.items():
                if name in key and 'lora_A' in key:
                    # 确保权重使用正确的数据类型和设备
                    module.lora_layer.lora_A.data = value.to(
                        device=module.lora_layer.lora_A.device, 
                        dtype=model_dtype
                    )
            for key, value in lora_B_weights.items():
                if name in key and 'lora_B' in key:
                    # 确保权重使用正确的数据类型和设备
                    module.lora_layer.lora_B.data = value.to(
                        device=module.lora_layer.lora_B.device, 
                        dtype=model_dtype
                    )
    
    print(f"LoRA weights loaded with dtype: {model_dtype}")
    return model

def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_seen')
    parser.add_argument("--output_path", type=str, default='./results/val_seen/streamvln')
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help="Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='Path to the base model (for LoRA models)')
    parser.add_argument('--lora_A_dir', type=str, default=None,
                       help='Directory containing LoRA matrix A weights')
    parser.add_argument('--lora_B_dir', type=str, default=None,
                       help='Directory containing LoRA matrix B weights')
    
    # 添加Tucker-LoRA特定参数
    parser.add_argument('--tucker_dir', type=str, default=None,
                       help='Directory containing Tucker-LoRA weights')
    parser.add_argument('--expert_weights', type=str, default=None,
                       help='Expert weights for Tucker-LoRA MOE (comma-separated, e.g., "0.4,0.3,0.2,0.1")')
    
    # 添加4D Tucker-LoRA相关参数
    parser.add_argument('--use_hard_routing', action='store_true', default=False,
                        help='Use hard routing for 4D Tucker-LoRA')
    parser.add_argument('--scene_idx', type=int, default=None,
                        help='Scene index for hard routing (0-4)')
    parser.add_argument('--env_idx', type=int, default=None,
                        help='Environment index for hard routing (0-3)')
    parser.add_argument('--auto_route_by_task', type=int, default=None,
                        help='Automatically set route based on task ID')
    
    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank

    # 自动路由设置（基于任务ID）
    if args.auto_route_by_task is not None:
        task_id = args.auto_route_by_task
        args.use_hard_routing = True
        args.scene_idx = task_id % 5  # 5个场景
        args.env_idx = task_id % 4    # 4个环境
        print(f"Auto-routing for task {task_id}: Scene {args.scene_idx}, Env {args.env_idx}")


    # 使用新的加载函数
    model, tokenizer, config, is_lora_model = load_model_and_tokenizer(args)
    
    # 移动到设备
    model.to('cuda')
    model.model.num_history = args.num_history
    model.requires_grad_(False)
    model.to(local_rank)
    
    # 验证模型状态
    if is_lora_model:
        print("Verifying LoRA/Tucker-LoRA model...")
        print(f"Model type: {type(model)}")
        
        # 检查是否是Tucker-LoRA
        if hasattr(model, 'is_tucker_lora') and model.is_tucker_lora:
            print("Tucker-LoRA model confirmed")
            # 统计Tucker-LoRA参数
            tucker_params = 0
            for name, module in model.named_modules():
                if hasattr(module, 'lora_layer'):
                    tucker_params += module.lora_layer.U1.numel()
                    tucker_params += module.lora_layer.U2.numel()
                    tucker_params += module.lora_layer.U3.numel()
                    tucker_params += module.lora_layer.G.numel()
            print(f"Total Tucker-LoRA parameters: {tucker_params:,}")
        
        print("=" * 50)
    
    evaluate(model, tokenizer, args)


def evaluate(model, tokenizer, args):
    model.eval()
    
    world_size = get_world_size()
    model.reset(world_size)
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        epoch=0,
        args=args
    )
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank()) 
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    result_all = {
                    "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
                    "spls_all": (sum(spls_all)/len(spls_all)).item(),
                    "oss_all": (sum(oss_all)/len(oss_all)).item(),
                    "ones_all": (sum(ones_all)/len(ones_all)).item(),
                    'length': len(sucs_all)
                }
    
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()