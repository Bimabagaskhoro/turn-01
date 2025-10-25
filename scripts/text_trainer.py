#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
import re
from datetime import datetime, timezone, timedelta

import yaml
from transformers import AutoTokenizer


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import train_cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import TaskType
import training_paths as train_paths
from instruct_config import get_training_json as get_instruct_training_json
from dpo_config import get_training_json as get_dpo_training_json
from grpo_config import get_training_json as get_grpo_training_json
import pathlib
from transformers import AutoConfig


def run_cmd_with_log(cmd: str, log_file_path: str, env_vars: dict = None):
    print(f"Running command: {cmd}", flush=True)
    with open(log_file_path, "w") as log_file:
        # Prepare environment variables
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)
        
        # Run the command, capturing stdout and stderr
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        # Wait for the process to complete
        return_code = process.wait()

        # Log the return code
        log_file.write(f"\nProcess completed with return code: {return_code}\n")


def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)([^\s]+))(\s+)", cmd)
    if match:
        left_index = match.start("p")
        right_index = match.end("p")
        return cmd[:left_index] + f" --{arg_name} {arg_value} " + cmd[right_index:]
    else:
        return None


def extract_value_from_cmd(cmd: str, arg_name: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)(?P<value>[^\s]+))(\s+)", cmd)
    if match:
        return match.group("value")
    else:
        return None


def get_model_architecture(model_name: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_name)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except Exception as e:
        if "model type `gpt_oss`" in str(e):
            return "GptOssForCausalLM"
        return "Unknown"
    
    
def is_openai_model(model_name: str) -> bool:
    architecture = get_model_architecture(model_name)
    if architecture.lower() == "gptossforcausallm":
        return True
    return False


def load_task_config(task_type: str) -> dict:
    """Load configuration from JSON file based on task type"""
    config_map = {
        "InstructTextTask": "instruct_config.json",
        "ChatTask": "chat_config.json",
        "DpoTask": "dpo_config.json", 
        "GrpoTask": "grpo_config.json"
    }
    
    config_file = config_map.get(task_type)
    if not config_file:
        return {}
    
    config_path = os.path.join(script_dir, "configs", config_file)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}: {config}", flush=True)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}", flush=True)
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing config file {config_path}: {e}", flush=True)
        return {}


OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
VLLM_OOM_ERROR = "ValueError: No available memory for the cache blocks"


def get_error_type(log_path: str):
    with open(log_path, "r") as f:
        text = f.read()
    if OOM_ERROR in text:
        return OOM_ERROR
    elif VLLM_OOM_ERROR in text:
        return VLLM_OOM_ERROR
    else:
        return None


def patch_wandb_symlinks(base_dir:str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", required=True, help="Dataset path or HF dataset name"
    )
    parser.add_argument(
        "--dataset-type", required=True, help="JSON string of dataset type config"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["InstructTextTask", "ChatTask", "DpoTask", "GrpoTask"],
        help="Type of task",
    )
    parser.add_argument(
        "--file-format",
        required=False,
        choices=["csv", "json", "hf", "s3"],
        help="File format",
        default="s3",
    )
    parser.add_argument(
        "--hours-to-complete",
        type=float,
        required=True,
        help="Number of hours to complete the task",
    )
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument(
        "--max-data-size",
        type=int,
        help="Max data size to use for training",
        default=-1,
    )
    parser.add_argument(
        "--max-steps", type=int, help="Max steps to use for training", default=-1
    )
    parser.add_argument(
        "--retries", type=int, help="Number of retries", default=5
    )
    parser.add_argument(
        "--min-steps", type=int, help="Min steps to use for training", default=100
    )
    
    parser.add_argument(
        "--reg-ratio", type=float, help="Reg ratio to use for training", default=1.001
    )
    parser.add_argument(
        "--lr-amplify-rate", type=float, help="LR amplify rate to use for training", default=1.051
    )
    parser.add_argument(
        "--find-lr", action="store_true", help="Enable LR Finder", default=True
    )
    parser.add_argument(
        "--min-lr", type=float, help="Minimum learning rate for LR finder", default=0.98
    )
    parser.add_argument(
        "--max-lr", type=float, help="Maximum learning rate for LR finder", default=1.23
    )
    
    args = parser.parse_args()
    original_model_name = args.model
    original_task_type = args.task_type
    
    # Load task-specific configuration from JSON
    task_config = load_task_config(args.task_type)
    
    # Override args with config values if not provided via command line
    if hasattr(args, 'reg_ratio') and args.reg_ratio == 1.001:  # default value
        args.reg_ratio = task_config.get("reg-ratio", args.reg_ratio)
    
    if hasattr(args, 'lr_amplify_rate') and args.lr_amplify_rate == 1.051:  # default value
        args.lr_amplify_rate = task_config.get("lr-amplify-rate", args.lr_amplify_rate)
    
    if args.min_lr is None:
        args.min_lr = task_config.get("min_lr", 1e-6)
    
    if args.max_lr is None:
        args.max_lr = task_config.get("max_lr", 1e-3)
    
    # Handle find_lr from config (convert string to boolean if needed)
    config_find_lr = task_config.get("find-lr", "True")
    if isinstance(config_find_lr, str):
        config_find_lr = config_find_lr.lower() == "true"
    args.find_lr = config_find_lr if not hasattr(args, 'find_lr') or args.find_lr is True else args.find_lr
    
    print(f"Using configuration - reg_ratio: {args.reg_ratio}, lr_amplify_rate: {args.lr_amplify_rate}", flush=True)
    print(f"LR Finder settings - find_lr: {args.find_lr}, min_lr: {args.min_lr}, max_lr: {args.max_lr}", flush=True)

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    submission_dir = train_paths.get_checkpoints_output_path(
        args.task_id, args.expected_repo_name
    )
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    output_dir = f"/workspace/scripts/soutputs/{args.task_id}"
    os.makedirs(output_dir, exist_ok=True)

    end_time = datetime.now(timezone.utc) + timedelta(
        hours=args.hours_to_complete - 3 / 60
    )  # assume that 3 minutes to go this far
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)

    ds_folder = "datasets"
    os.makedirs(ds_folder, exist_ok=True)
    request_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")
    model_path = str(train_paths.get_text_base_model_path(original_model_name))
    
    is_openai = False
    if is_openai_model(original_model_name):
        print("Upgrading python3 packages for openai model", flush=True)
        run_cmd_with_log(
            "pip uninstall -y transformers && pip install transformers==4.55.0", os.path.join(ds_folder, f"upgrade_transformers.log")
        )
        #upgrade deepspeed
        run_cmd_with_log("pip uninstall -y deepspeed && pip install deepspeed==0.17.4", os.path.join(ds_folder, f"upgrade_deepspeed.log"))
        # install kernel 
        run_cmd_with_log("pip install kernels==0.9.0", os.path.join(ds_folder, f"install_kernel.log"))
        is_openai = True
    
    train_info = {
        "model_name": original_model_name,
        "model_path": model_path,
        "task_id": args.task_id,
        "dataset": dataset_path,
        "hours_to_complete": args.hours_to_complete,
        "expected_repo_name": args.expected_repo_name,
        "end_time": end_time,
        "dataset_type": dataset_type_dict,
        "submission_dir": submission_dir,
        "output_dir": output_dir,
        "min_steps":  150,
        "adjust_batch_size": True,
        "request_path": request_path,
        "max_data_size": args.max_data_size,
        "max_steps": args.max_steps,
        "wandb_log_dir": train_cst.WANDB_LOGS_DIR,
        "min_steps": args.min_steps,
        "is_openai": is_openai,
        "reg_ratio": args.reg_ratio,
        "find_lk_lr": True,
        "find_lr": args.find_lr,
        "lr_amplify_rate": args.lr_amplify_rate,
        "min_lr": args.min_lr,
        "max_lr": args.max_lr,
    }

    if args.task_type == TaskType.INSTRUCTTEXTTASK.value or args.task_type == TaskType.CHATTASK.value:
        print(f"Processing {args.task_type} using instruct configuration", flush=True)
        train_info = get_instruct_training_json(train_info)
        tokenize_cmd = (
            f"/workspace/axo_py/bin/python3 tokenize_instruct.py {request_path}"
        )
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.DPOTASK.value:
        train_info = get_dpo_training_json(train_info)
        tokenize_cmd = f"python3 tokenize_dpo.py {request_path}"
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.GRPOTASK.value:
        train_info = get_grpo_training_json(train_info)
        tokenize_cmd = f"python3 tokenize_grpo.py {request_path}"
        train_cmd = train_info["run_cmd"]
    else:
        raise ValueError(f"Task type {args.task_type} not supported")

    with open(request_path, "w") as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)

    run_cmd_with_log(
        tokenize_cmd, os.path.join(ds_folder, f"tokenize_{args.task_id}.log")
    )

    train_success = False
    log_path = os.path.join(ds_folder, f"train_{args.task_id}.log")
    for i in range(args.retries):
        print(
            f"************* Training attempt {i+1}/{args.retries} for task {args.task_id}*************",
            flush=True,
        )
        if i > 0:  # there was something wrong so we will reduce the batch_size
            # first check if the training is OOM
            if os.path.exists(log_path):
                error_type = get_error_type(log_path)
                if error_type == OOM_ERROR:
                    current_batch_size = extract_value_from_cmd(
                        train_cmd, "per_device_train_batch_size"
                    )
                    current_batch_size = int(current_batch_size)
                    if current_batch_size > 1:
                        new_batch_size = current_batch_size // 2
                        print(
                            f"Reducing batch size from {current_batch_size} to {new_batch_size}",
                            flush=True,
                        )
                        train_cmd = replace_args_in_cmd(
                            train_cmd,
                            "per_device_train_batch_size",
                            str(new_batch_size),
                        )
                        print(f"New train command: {train_cmd}", flush=True)
                    else:
                        print(f"batch size is 1, cannot reduce further", flush=True)
                        if args.task_type == TaskType.GRPOTASK.value:
                            # disable vllm
                            train_cmd = replace_args_in_cmd(
                                train_cmd, "use_vllm", "False"
                            )
                            print(f"disable VLLM {train_cmd}", flush=True)
                elif error_type == VLLM_OOM_ERROR:
                    if args.task_type == TaskType.GRPOTASK.value:
                        print(f"VLLM OOM error, disable VLLM", flush=True)
                        train_cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")

        # empty the log file if it exists
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("STARTING TRAINING")

        task_id = args.task_id
        expected_repo_name = args.expected_repo_name
        
        training_env_vars = {
            "WANDB_MODE": "offline",
            "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
            "WANDB_NAME": f"{task_id}_{expected_repo_name}",
        }
        
        run_cmd_with_log(train_cmd, log_path, env_vars=training_env_vars)
        # check if the training is successfully done; it is done, the output_dir should not be empty there is at least 2 files in the submission_dir
        if not os.path.exists(submission_dir) or len(os.listdir(submission_dir)) < 2:
            print(f"Training failed for task {args.task_id}", flush=True)
        else:
            print(f"Training successfully done for task {args.task_id}", flush=True)
            train_success = True
            break

    if not train_success:
        print(f"Training failed for task {args.task_id}", flush=True)
        # add noise to the model
        add_noise_cmd = f"python3 add_random_noise.py {model_path} {submission_dir}"
        run_cmd_with_log(
            add_noise_cmd, os.path.join(ds_folder, f"add_noise_{args.task_id}.log")
        )
    
    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    main()
