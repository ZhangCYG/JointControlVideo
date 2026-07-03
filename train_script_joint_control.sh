#!/bin/bash

source torch27/bin/activate
export PYTHONPATH=torch27/lib/python3.12/site-packages

set -x

ACCEL_PROCS=$(( $SLURM_NNODES * $SLURM_GPUS_PER_NODE ))

MAIN_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MAIN_PORT=10237
WORLD_SIZE=$(( $SLURM_NNODES * $SLURM_GPUS_PER_NODE ))
LOCAL_RANK=$SLURM_NODEID

accelerate launch \
  --num_machines $SLURM_NNODES \
  --num_processes $WORLD_SIZE \
  --machine_rank $LOCAL_RANK \
  --main_process_ip $MAIN_ADDR \
  --main_process_port $MAIN_PORT \
  --mixed_precision "bf16" \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path  \
  --dataset_metadata_path  \
  --h5_base_path \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --num_frames 81 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --num_epochs 2 \
  --learning_rate 1e-4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --extra_inputs "input_image" \
  --data_file_keys "video" \
  --output_path "./models_train_control" \
  --use_lmdb_dataset \
  --save_steps 100 \
  --trainable_models "hand_controler" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 64