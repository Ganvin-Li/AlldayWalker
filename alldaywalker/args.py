from dataclasses import dataclass, field
from threading import local
from typing import Dict, Optional, Sequence, List
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_size: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

    num_future_steps: Optional[int] = field(default=1)
    num_frames: Optional[int] = field(default=32)
    num_history: Optional[int] = field(default=None)
    data_augmentation: Optional[bool] = field(default=False)
    transform_train: Optional[str] = field(default=None)
    image_size: Optional[int] = field(default=384)
    remove_init_turns: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_target_modules: str = "q_proj,v_proj"
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    group_by_task: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})

    # LoRA相关参数
    use_lora: bool = field(default=False, metadata={"help": "是否使用LoRA微调"})
    lora_r: int = field(default=64, metadata={"help": "LoRA的rank值"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA的alpha值"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA的dropout率"})
    lora_target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", metadata={"help": "LoRA目标模块"})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias设置"})
    
    # 预训练权重加载
    pretrained_checkpoint_path: Optional[str] = field(default="model_zoo/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln/", metadata={"help": "预训练检查点路径"})
    
    # LoRA分离存储路径
    lora_A_dir: Optional[str] = field(default=None,metadata={"help": "Directory to save LoRA matrix A weights. If None, will use output_dir/lora_A"})
    lora_B_dir: Optional[str] = field(default=None,metadata={"help": "Directory to save LoRA matrix B weights. If None, will use output_dir/lora_B"})
    
    # Tucker-LoRA特定参数
    tucker_num_experts: int = field(default=4, metadata={"help": "Number of experts for Tucker-LoRA MOE"})
    tucker_ranks: str = field(default="16,16,8", metadata={"help": "Tucker decomposition ranks (r1,r2,r3)"})
    tucker_init_scale: float = field(default=0.02, metadata={"help": "Initialization scale for Tucker factors"})
    tucker_use_internal_gating: bool = field(default=False, metadata={"help": "Use internal gating network for expert selection"})
    
    # 预训练权重加载
    pretrained_checkpoint_path: Optional[str] = field(default=None, metadata={"help": "预训练检查点路径"})
    
    # Tucker-LoRA分离存储路径
    tucker_dir: Optional[str] = field(default=None, metadata={"help": "Directory to save Tucker-LoRA weights. If None, will use output_dir/tucker_lora"})

    # 四维Tucker-LoRA参数
    use_tucker_4d: bool = field(default=False, metadata={"help": "使用四维Tucker分解"})
    tucker_scene_num: int = field(default=5, metadata={"help": "场景数量"})
    tucker_env_num: int = field(default=4, metadata={"help": "环境数量"})
    tucker_ranks_4d: str = field(default="16,16,8,8", metadata={"help": "四维Tucker ranks (r1,r2,r3,r4)"})
    
    # 持续学习参数
    continual_learning: bool = field(default=False, metadata={"help": "启用持续学习模式"})
    num_tasks: int = field(default=20, metadata={"help": "持续学习任务数量"})
    current_task_id: int = field(default=0, metadata={"help": "当前任务ID"})
    ewc_lambda: float = field(default=5000.0, metadata={"help": "EWC正则化系数"})
    ewc_mode: str = field(default="online", metadata={"help": "EWC模式: online或standard"})
    
    # 路由配置
    route_config_file: Optional[str] = field(default=None, metadata={"help": "路由配置文件路径"})
    hard_route_selection: bool = field(default=True, metadata={"help": "使用硬路由选择"})
    
    # 正交约束
    use_orthogonal_reg: bool = field(default=True, metadata={"help": "对U3和U4使用正交约束"})
    ortho_reg_weight: float = field(default=0.01, metadata={"help": "正交约束权重"})
    
    # 稳定性约束参数
    stability_weight: float = field(default=1.0, metadata={"help": "Weight for U3/U4 stability loss to prevent drastic changes in trained rows"})
    
    # Fisher信息矩阵保存路径
    fisher_path: Optional[str] = field(default=None, metadata={"help": "Fisher信息矩阵保存路径"})
    old_params_path: Optional[str] = field(default=None, metadata={"help": "旧参数保存路径"})