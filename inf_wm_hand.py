import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import LMDBH5UnifiedDataset
import itertools
from safetensors import safe_open
from torch.nn.parameter import Parameter


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def replace_modules_in_pipe(pipe: WanVideoPipeline):
    """
    Performs the architectural modification by replacing existing modules 
    with new, expanded modules, as requested by the user.
    """
    print("\n--- Performing Architecture Replacement (Channel Doubling) ---")

    # 1. Patch Embedding (Doubling In-Channels)
    original_conv = pipe.dit.patch_embedding
    _n_convin_out_channel = original_conv.out_channels
    _n_convin_in_channel = original_conv.in_channels + 16
    
    # Reinitialize patch_embedding with the new, doubled input channels
    new_conv_in = torch.nn.Conv3d(
        _n_convin_in_channel, _n_convin_out_channel, 
        kernel_size=original_conv.kernel_size, 
        stride=original_conv.stride, 
        padding=original_conv.padding
    )
    # Assign the new, empty module (weights will be loaded from state_dict next)
    pipe.dit.patch_embedding = new_conv_in.to(torch.bfloat16).to(pipe.device)
    print(f"Patched Conv3d: In-Channels changed from {original_conv.in_channels} to {new_conv_in.in_channels}")


ALLOWED_KEYS = {
    'patch_embedding.bias',
    'patch_embedding.weight',
}


def apply_custom_state_dict(pipe: WanVideoPipeline, state_dict):
    """
    Applies the state_dict by performing a direct, non-slicing copy, 
    assuming the module architecture has already been modified to match 
    the checkpoint sizes.
    """
    target_module = pipe.dit

    print(f"\n--- Starting Direct Checkpoint Loading (Exact Match) ---")
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
                # Case 1: Shapes must match exactly (direct copy)
                value = value.to(param.data.dtype).to(param.data.device) 
                param.data.copy_(value)
                updated_count += 1
                print(f"-> Successfully updated (Exact Match): {key}")

            else:
                # Any shape mismatch is treated as an error since slicing is not allowed
                print(f"Error: Skipping '{key}' due to major shape mismatch.")
                print(f"    Target shape: {target_shape}, Source shape: {source_shape}")
                mismatched_count += 1

        except AttributeError:
            print(f"Error: Could not find target module or parameter for key: '{key}'. Skipping.")
            mismatched_count += 1

    print(f"\n--- Checkpoint Loading Summary ---")
    print(f"Total parameters updated: {updated_count}")
    print(f"Total parameters ignored (filtered out): {ignored_count}")
    print(f"Total parameters skipped (mismatch/error): {mismatched_count}")


DIT_CKPT  = "./models_train_control/dit.safetensors"
HAND_CKPT = "./models_train_control/hand_controller.safetensors"

# Example data shipped under example_data/ (LMDB + H5 stores referenced by the split JSON).
# Swap these out for your own dataset to render hand-controlled videos on different clips.
DATASET_BASE_PATH     = "./example_data/lmdb"
DATASET_METADATA_PATH = "./example_data_split/test.json"
DATASET_H5_BASE_PATH  = "./example_data/h5"

state_dict = load_state_dict_from_safetensors(DIT_CKPT, torch_dtype=torch.bfloat16)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ModelConfig(path=HAND_CKPT),
    ],
)

replace_modules_in_pipe(pipe)
apply_custom_state_dict(pipe, state_dict)
pipe.load_lora(pipe.dit, DIT_CKPT, alpha=1)

dataset = LMDBH5UnifiedDataset(
    base_path=DATASET_BASE_PATH,
    metadata_path=DATASET_METADATA_PATH,
    h5_base_path=DATASET_H5_BASE_PATH,
    data_file_keys="video,image",
    height=480, width=832, num_frames=81, is_test=True,
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
data = next(itertools.islice(dataloader, 20, 21))
prompt = data['prompt']
image = data['image']
print(prompt)
joints_cam = data['joints_cam']
intrinsics = data['K']

video_rgb = pipe(
    prompt=prompt,
    input_image=image,
    joints_cam=joints_cam,
    intrinsics=intrinsics,
    negative_prompt="",
    cfg_scale=5.0,
    height=480,
    width=832,
    num_frames=81,
    seed=0, tiled=True,
)
import os
os.makedirs("outputs", exist_ok=True)
save_video(video_rgb, "outputs/demo.mp4", fps=15, quality=5)