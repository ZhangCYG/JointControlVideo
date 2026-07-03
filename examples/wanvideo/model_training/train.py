import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset, LMDBH5UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from torch.nn.parameter import Parameter
from safetensors import safe_open
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_hand_controler_from_prior(hand_controler, prior_path):
    """Load a prior 42-joint HandConditioningModule checkpoint into a module with a
    different num_joints; joint_id_embed.weight is re-initialized because its shape changes."""
    from safetensors.torch import load_file as load_safetensors
    import pathlib

    suffix = pathlib.Path(prior_path).suffix.lower()
    if suffix == ".safetensors":
        prior_sd = load_safetensors(prior_path, device="cpu")
    else:
        prior_sd = torch.load(prior_path, map_location="cpu")
        if "state_dict" in prior_sd:
            prior_sd = prior_sd["state_dict"]

    stripped = {k.removeprefix("hand_controler."): v for k, v in prior_sd.items()}
    missing, unexpected = hand_controler.load_state_dict(stripped, strict=False)

    print(f"Loaded prior hand_controler weights from {prior_path}")
    if missing:
        print(f"  Missing keys (re-initialized): {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    return hand_controler


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        num_joints=42,
        prior_hand_checkpoint=None,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)

        if num_joints != 42:
            from diffsynth.models.wan_video_joint_control import HandConditioningModule
            print(f"Replacing hand_controler: num_joints 42 -> {num_joints}")
            self.pipe.hand_controler = HandConditioningModule(
                num_joints=num_joints, embed_dim=64, output_dim=16,
                canvas_height=480, canvas_width=832, downsample_rate=8, heatmap_sigma=2.0,
            ).to(torch.float).to(device)

        if prior_hand_checkpoint is not None:
            if self.pipe.hand_controler is None:
                from diffsynth.models.wan_video_joint_control import HandConditioningModule
                self.pipe.hand_controler = HandConditioningModule(
                    num_joints=num_joints, embed_dim=64, output_dim=16,
                    canvas_height=480, canvas_width=832, downsample_rate=8, heatmap_sigma=2.0,
                ).to(torch.float).to(device)
            load_hand_controler_from_prior(self.pipe.hand_controler, prior_hand_checkpoint)

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        if lora_checkpoint is not None:
            ckpt = self.load_state_dict_from_safetensors(lora_checkpoint, torch_dtype=torch.bfloat16)
            self.replace_modules_in_pipe()
            self.apply_custom_state_dict(ckpt)
        else:
            self._replace_embedding_conv_in()

    def load_state_dict_from_safetensors(self, file_path, torch_dtype=None, device="cpu"):
        state_dict = {}
        with safe_open(file_path, framework="pt", device=str(device)) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
                if torch_dtype is not None:
                    state_dict[k] = state_dict[k].to(torch_dtype)
        return state_dict

    def replace_modules_in_pipe(self):
        """
        Performs the architectural modification by replacing existing modules 
        with new, expanded modules, as requested by the user.
        """
        print("\n--- Performing Architecture Replacement (Channel +16) ---")

        original_conv = self.pipe.dit.patch_embedding
        _n_convin_out_channel = original_conv.out_channels
        _n_convin_in_channel = original_conv.in_channels + 16
        new_conv_in = torch.nn.Conv3d(
            _n_convin_in_channel, _n_convin_out_channel, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding
        )

        self.pipe.dit.patch_embedding = new_conv_in.to(torch.bfloat16).to(self.pipe.device)
        print(f"Patched Conv3d: In-Channels changed from {original_conv.in_channels} to {new_conv_in.in_channels}")

    def apply_custom_state_dict(self, state_dict):
        """
        Applies the state_dict by performing a direct copy for exact matches,
        or partial loading with zero-initialization for channel mismatches.
        Specifically handles mapping source weights (e.g., 52 channels) to 
        target weights (e.g., 68 channels).
        """
        target_module = self.pipe.dit
        ALLOWED_KEYS = {
            'patch_embedding.bias',
            'patch_embedding.weight',
        }

        print(f"\n--- Starting Custom Checkpoint Loading (Partial/Exact) ---")
        print(f"Checkpoint contains {len(state_dict)} keys. Only processing: {list(ALLOWED_KEYS)}")

        updated_count = 0
        mismatched_count = 0
        ignored_count = 0

        for key, value in state_dict.items():
            if key not in ALLOWED_KEYS:
                ignored_count += 1
                continue

            try:
                module_path = key.rsplit('.', 1)[0]
                param_name = key.rsplit('.', 1)[-1]
                current_module = target_module
                for sub_module_name in module_path.split('.'):
                    current_module = getattr(current_module, sub_module_name)

                param: Parameter = getattr(current_module, param_name)
                
                target_shape = tuple(param.data.shape)
                source_shape = tuple(value.shape)

                if target_shape == source_shape:
                    value = value.to(param.data.dtype).to(param.data.device) 
                    param.data.copy_(value)
                    updated_count += 1
                    print(f"-> Successfully updated (Exact Match): {key}")

                elif (len(target_shape) == len(source_shape) and 
                    target_shape[0] == source_shape[0] and 
                    target_shape[2:] == source_shape[2:]):
                    
                    src_channels = source_shape[1]
                    tgt_channels = target_shape[1]

                    print(f"-> Partial loading '{key}': Source ({src_channels}) -> Target ({tgt_channels})")
                    new_weight = torch.zeros(target_shape, dtype=param.data.dtype, device=param.data.device)
                    
                    channels_to_copy = min(src_channels, tgt_channels)
                    source_data = value.to(param.data.dtype).to(param.data.device)
                    
                    new_weight[:, :channels_to_copy, ...] = source_data[:, :channels_to_copy, ...]
                    
                    param.data.copy_(new_weight)
                    updated_count += 1
                    print(f"   Successfully sliced and zero-initialized remainder for: {key}")

                else:
                    print(f"Error: Skipping '{key}' due to incompatible shape mismatch.")
                    print(f"    Target shape: {target_shape}, Source shape: {source_shape}")
                    mismatched_count += 1

            except AttributeError:
                print(f"Error: Could not find target module or parameter for key: '{key}'. Skipping.")
                mismatched_count += 1

        print(f"\n--- Checkpoint Loading Summary ---")
        print(f"Total parameters updated: {updated_count}")
        print(f"Total parameters ignored: {ignored_count}")
        print(f"Total parameters skipped: {mismatched_count}")
    
    def _replace_embedding_conv_in(self):
        _original_weight = self.pipe.dit.patch_embedding.weight.clone()
        _bias = self.pipe.dit.patch_embedding.bias.clone()

        _out_channels = self.pipe.dit.patch_embedding.out_channels
        _original_in_channels = self.pipe.dit.patch_embedding.in_channels
        
        _n_convin_in_channel = _original_in_channels + 16

        
        _new_channels_weight = torch.zeros(
            _out_channels, 
            16,
            self.pipe.dit.patch_embedding.kernel_size[0], # d
            self.pipe.dit.patch_embedding.kernel_size[1], # h
            self.pipe.dit.patch_embedding.kernel_size[2]  # w
        ).to(_original_weight.device).to(_original_weight.dtype)
        
        _new_weight = torch.cat([_original_weight, _new_channels_weight], dim=1)
        
        _new_conv_in = torch.nn.Conv3d(
            _n_convin_in_channel, 
            _out_channels, 
            kernel_size=self.pipe.dit.patch_embedding.kernel_size, 
            stride=self.pipe.dit.patch_embedding.stride, 
            padding=self.pipe.dit.patch_embedding.padding
        )

        _new_conv_in.weight = Parameter(_new_weight)
        _new_conv_in.bias = Parameter(_bias)

        self.pipe.dit.patch_embedding = _new_conv_in
        
        print("Patch embedding conv_in layer is replaced (3D) with input channels:", _n_convin_in_channel)
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        if 'joints_cam' in data:
            inputs_shared['joints_cam'] = data['joints_cam']
        if 'K' in data:
            inputs_shared['intrinsics'] = data['K']
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        inputs[0]['joints_cam'] = inputs[0]['joints_cam'].to(torch.float)
        inputs[0]['intrinsics'] = inputs[0]['intrinsics'].to(torch.float)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--use_lmdb_dataset", default=False, action="store_true", help="Whether to use LMDB dataset.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name.")
    parser.add_argument("--wandb_name", type=str, default=None, help="Weights & Biases run name.")
    parser.add_argument("--num_joints", type=int, default=42, help="Total number of joints for HandConditioningModule (42 for MANO).")
    parser.add_argument("--prior_hand_checkpoint", type=str, default=None, help="Path to a prior HandConditioningModule checkpoint (.safetensors or .pt) to load compatible weights from.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    if not args.use_lmdb_dataset:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
                "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            }
        )
    else:
        dataset = LMDBH5UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            h5_base_path=args.h5_base_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_joints=args.num_joints,
        prior_hand_checkpoint=args.prior_hand_checkpoint,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
