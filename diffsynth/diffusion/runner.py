import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
import wandb
import math
from torch.optim.lr_scheduler import LambdaLR


def lr_lambda_lora(current_step):
    return 1.0

# Scheduler Function 2: Warmup + Cosine for Hand Control
def lr_lambda_hand(current_step):
    # 1. Warmup Phase
    if current_step < 100:
        return float(current_step) / float(max(1, 100))
    
    # 2. Cosine Decay Phase
    progress = float(current_step - 100) / float(max(1, 1600 - 100))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        wandb_project = args.wandb_project
        wandb_name = args.wandb_name
    
    # 1. Separate the parameters
    lora_params = []
    hand_params = []
    hand_param_names = [] # Just for debugging verification

    # Adjust 'hand_controler' to match your exact module name in the state dict
    module_name_to_train_scratch = "hand_controler" 

    for name, param in model.trainable_name_modules():
        if module_name_to_train_scratch in name:
            hand_params.append(param)
            hand_param_names.append(name)
        else:
            lora_params.append(param)

    print(f"LoRA params found: {len(lora_params)}")
    print(f"Hand Control params found: {len(hand_params)}")

    # 2. Define the Optimizer with Parameter Groups
    # Group 0: LoRA (Low LR, Constant)
    # Group 1: Hand Control (High LR, Scratch Training)
    optimizer = torch.optim.AdamW([
        {
            'params': lora_params, 
            'lr': learning_rate, 
            'weight_decay': weight_decay
        },
        {
            'params': hand_params, 
            'lr': 2*learning_rate,         
            'weight_decay': weight_decay   # 0.01 is standard; can go to 0.05 if overfitting
        }
    ])
    # optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda_lora, lr_lambda_hand])
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    if wandb_project is not None and wandb_name is not None and accelerator.is_main_process:
        wandb_logger = wandb.init(project=wandb_project, name=wandb_name)
    else:
        wandb_logger = None
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # if dataset.load_from_cache:
                #     loss = model({}, inputs=data)
                # else:
                loss = model(data)
                accelerator.backward(loss)
                if wandb_logger is not None and accelerator.is_main_process:
                    wandb_logger.log({"loss": loss.item()})
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
