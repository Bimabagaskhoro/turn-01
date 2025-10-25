from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
from typing import Callable, Optional, Dict
import shutil
import json
from transformers.trainer_utils import is_main_process
import wandb
import torch


MIS_MATCH_VOCAB_SIZE_MODELS = [
    'NousResearch/Nous-Capybara-7B-V1',
    'berkeley-nest/Starling-LM-7B-alpha',
    'NousResearch/Hermes-2-Theta-Llama-3-8B',
    'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4'
]

ERROR_GENERATION_CONFIG_MODELS = [
    "lmsys/vicuna-7b-v1.5", 
    "lmsys/vicuna-13b-v1.5",
    "NousResearch/Nous-Hermes-llama-2-7b", 
    "defog/llama-3-sqlcoder-8b"
]

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

print(f"LOCAL_RANK: {LOCAL_RANK} in customized_trainer.py", flush=True)
def patch_model_metadata(output_dir: str, base_model_id: str):
    try:
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                config = json.load(f)

            config["base_model_name_or_path"] = base_model_id

            with open(adapter_config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Updated adapter_config.json with base_model: {base_model_id}", flush=True)
        else:
            print(" adapter_config.json not found", flush=True)

        readme_path = os.path.join(output_dir, "README.md")

        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.strip().startswith("base_model:"):
                    new_lines.append(f"base_model: {base_model_id}\n")
                else:
                    new_lines.append(line)

            with open(readme_path, "w") as f:
                f.writelines(new_lines)

            print(f"Updated README.md with base_model: {base_model_id}", flush=True)
        else:
            print("README.md not found", flush=True)

    except Exception as e:
        print(f"Error updating metadata: {e}", flush=True)
        pass 
class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
        max_steps: int = -1,
        checkpoint_manager = None  # Add EMA integration
    ):
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        self.max_steps = max_steps
        self.has_checkpoint = False
        self.save_only = False
        
        # EMA Integration
        self.checkpoint_manager = checkpoint_manager
        self.best_is_ema = False
        self.ema_evaluation_in_progress = False  # Prevent recursion
    
    def _evaluate_ema_model_safe(self, model, trainer):
        """
        Safely evaluate EMA model without causing recursion.
        Returns EMA model evaluation loss.
        """
        if not self.checkpoint_manager or not trainer:
            return float('inf')
            
        try:
            # Store original weights
            original_state = self.checkpoint_manager._get_model_state(model)
            
            # Apply EMA weights
            self.checkpoint_manager._apply_weights(model, self.checkpoint_manager.ema_state)
            
            # Evaluate with EMA weights (using trainer's evaluate but with recursion protection)
            model.eval()
            with torch.no_grad():
                eval_results = trainer.evaluate()
                ema_loss = eval_results.get("eval_loss", float('inf'))
            
            # Restore original weights
            self.checkpoint_manager._apply_weights(model, original_state)
            
            return ema_loss
            
        except Exception as e:
            print(f"Error in EMA evaluation: {e}", flush=True)
            # Ensure we restore original weights even on error
            try:
                if 'original_state' in locals():
                    self.checkpoint_manager._apply_weights(model, original_state)
            except:
                pass
            return float('inf')
            
    def compute_loss(self, state: TrainerState, metrics):
        return metrics.get("eval_loss", None)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Custom logic to decide whether to save or evaluate
        # print(f"************* on_step_end: {state.global_step}, check eval", flush=True)
        # TODO: implement the logic to save the model without evaluating if there is no check points --> avoid evaluating takes too much time
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        if when_to_eval["eval"]:
            # do not allow the pod to be stopped by any reason 
                # first check if there is at least one checkpoint or not 
            print(f"Evaluating the model at step: {state.global_step} the reason: {when_to_eval['reason']}", flush=True)
            control.should_evaluate = True
            control.should_save = True
            if when_to_eval["reason"] == "end_time":
                if not self.has_checkpoint: # if there is no checkpoint, we just save the model, do not evaluate
                    print(f"No checkpoint found, just save the model at step: {state.global_step}", flush=True)
                    control.should_evaluate = False
                    self.save_only = True
        return control


    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        self.save_only = False
        
        # Prevent recursion during EMA evaluation
        if self.ema_evaluation_in_progress:
            return
            
        # Get standard model eval_loss
        standard_loss = self.compute_loss(state, metrics)
        if state.global_step < 2:
            return 
            
        print(f"GO INTO CUSTOMIZED EVALUATE AT STEP: {state.global_step}", flush=True)
        print(f"Standard model eval_loss: {standard_loss}", flush=True)
        
        # Get EMA loss if checkpoint manager is available
        ema_loss = float('inf')
        if self.checkpoint_manager and hasattr(self.checkpoint_manager, 'ema_state'):
            # Initialize EMA state if not exists
            if not self.checkpoint_manager.ema_state and self.checkpoint_manager.use_ema:
                try:
                    model = kwargs.get('model')
                    if model:
                        current_state = self.checkpoint_manager._get_model_state(model)
                        self.checkpoint_manager._update_ema_state(current_state)
                        print(f"EMA state initialized at step {state.global_step}", flush=True)
                except Exception as e:
                    print(f"EMA state initialization failed: {e}", flush=True)
            
            # Evaluate EMA model if state is ready
            if self.checkpoint_manager.ema_state:
                try:
                    self.ema_evaluation_in_progress = True
                    ema_loss = self._evaluate_ema_model_safe(kwargs.get('model'), kwargs.get('trainer'))
                    print(f"EMA model eval_loss: {ema_loss}", flush=True)
                except Exception as e:
                    print(f"EMA evaluation failed: {e}", flush=True)
                    ema_loss = float('inf')
                finally:
                    self.ema_evaluation_in_progress = False
            else:
                print(f"EMA state not ready at step {state.global_step} - using standard model only", flush=True)
        
        # Determine which model is better
        current_is_ema = ema_loss < standard_loss
        current_best_loss = ema_loss if current_is_ema else standard_loss
        
        print(f"Best model this cycle: {'EMA' if current_is_ema else 'Standard'} with loss {current_best_loss}", flush=True)
        
        # Update best checkpoint if this is better
        if self.best_checkpoint_info is None or current_best_loss < self.best_checkpoint_info["loss"]:
            print(f"Updating the best checkpoint info at step: {state.global_step} with eval_loss: {current_best_loss}", flush=True)
            self.best_checkpoint_info = {
                "loss": current_best_loss,
                "step": state.global_step
            }
            self.best_is_ema = current_is_ema
            self.update_best_checkpoint = True
        else:
            if self.best_checkpoint_info is not None:
                print(f"At step: {state.global_step} The eval_loss: {current_best_loss} is not smaller than the current best eval_loss: {self.best_checkpoint_info['loss']}, update_best_checkpoint={self.update_best_checkpoint}", flush=True)
            

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.global_step == self.max_steps and self.max_steps != -1:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            control.should_training_stop = True
        
        self.has_checkpoint = True
        
        if not is_main_process(LOCAL_RANK): # if not main process, skip this
            return 
            
        if self.save_only: # if only save, do not evaluate 
            print(f"Only save the model at step: {state.global_step}, no evaluation", flush=True)
            current_step = state.global_step
            # Remove existing directory if it exists
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
                
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{current_step}"),
                self.submission_dir
            )
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            patch_model_metadata(self.submission_dir, self.original_model_name)

            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{current_step},no_eval")
            
            # release the flag
            self.save_only = False
            return 
            
        # Custom logic after model is saved
        # You can trigger external services, logs, or backups here
        if (
            self.update_best_checkpoint
            and is_main_process(LOCAL_RANK)
        ):
            print(f"Copy the best checkpoint to the submission directory at step: {state.global_step}", flush=True)
            print(f"Best model type: {'EMA' if self.best_is_ema else 'Standard'}", flush=True)
            
            # Remove existing directory if it exists
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
            best_eval_loss = self.best_checkpoint_info["loss"]
            
            # Copy standard checkpoint first
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}"),
                self.submission_dir
            )
            
            # If best model is EMA, apply EMA weights to the copied model
            if self.best_is_ema and self.checkpoint_manager:
                try:
                    print("Applying EMA weights to submission model...", flush=True)
                    model = kwargs.get('model')
                    if model and self.checkpoint_manager.ema_state:
                        # Apply EMA weights to model
                        self.checkpoint_manager._apply_weights(model, self.checkpoint_manager.ema_state)
                        
                        # Save the EMA model to submission directory
                        model.save_pretrained(self.submission_dir)
                        print("EMA model saved to submission directory", flush=True)
                except Exception as e:
                    print(f"Failed to apply EMA weights to submission: {e}", flush=True)
                    print("Using standard checkpoint instead", flush=True)
            
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            patch_model_metadata(self.submission_dir, self.original_model_name)
            model_type = "ema" if self.best_is_ema else "standard"
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{self.best_checkpoint_info['step']},{best_eval_loss},{model_type}")


class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics):
        eval_loss = None
        if state.log_history:
            last_log_entry = state.log_history[-1]
            eval_loss = last_log_entry.get("eval_reward", None)
            print(f"choose eval_loss ({eval_loss}) as eval_reward from: last_log_entry: {last_log_entry}; \n metrics: {metrics}", flush=True)
        else:
            print(f"state.log_history is empty", flush=True)
            
        if eval_loss is not None:
            eval_loss = - eval_loss
            
        return eval_loss
    
    def _evaluate_ema_model_safe(self, model, trainer):
        """
        Override EMA evaluation for GRPO - use reward-based evaluation.
        Returns negative EMA model evaluation reward (to match loss convention).
        """
        if not self.checkpoint_manager or not trainer:
            return float('inf')
            
        try:
            # Store original weights
            original_state = self.checkpoint_manager._get_model_state(model)
            
            # Apply EMA weights
            self.checkpoint_manager._apply_weights(model, self.checkpoint_manager.ema_state)
            
            # Evaluate with EMA weights (using trainer's evaluate but with recursion protection)
            model.eval()
            with torch.no_grad():
                eval_results = trainer.evaluate()
                # For GRPO, we need to get the reward and convert to loss
                ema_reward = eval_results.get("eval_reward", 0.0)
                ema_loss = -ema_reward  # Convert reward to loss (higher reward = lower loss)
            
            # Restore original weights
            self.checkpoint_manager._apply_weights(model, original_state)
            
            return ema_loss
            
        except Exception as e:
            print(f"Error in GRPO EMA evaluation: {e}", flush=True)
            # Ensure we restore original weights even on error
            try:
                if 'original_state' in locals():
                    self.checkpoint_manager._apply_weights(model, original_state)
            except:
                pass
            return float('inf')
    
    def penalize_eval_loss(self, eval_loss: float):
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool: 
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)  # Make end_time timezone-aware in UTC
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result =  time_diff.total_seconds() < minutes * 60
    if result:
        print(f"*** current time: {now} end_time: {end_time} time_diff: {time_diff}", flush=True)
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, periodic_save_steps: int = -1, steps_per_epoch: int = -1, max_steps: int = -1):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps

    def __call__(self, global_step: int) -> dict:
        
        if self.steps_per_epoch != -1 and global_step % self.steps_per_epoch == 0 and global_step > 1:
            return {"eval": True, "reason": "epoch"}
        
        if self.periodic_save_steps != -1 and global_step % self.periodic_save_steps == 0 and global_step > 1:
            return {"eval": True, "reason": "periodic"}
        
        if self.save_before_remaining_time > 0 and not self.run_eval:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                print(f"***ALERT: The time is about to run out need to eval & save the model", flush=True)
                # the eval time might be higher than the end_time, so we need to let the pod not stop by setting a flag for this
                self.run_eval = True
                return {"eval": True, "reason": "end_time"}
        
        if self.max_steps != -1 and global_step == self.max_steps:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            return {"eval": True, "reason": "max_step"}

        return {"eval": False, "reason": "none"}


def set_generation_config(model_name, model):
    try:
        if model_name in ERROR_GENERATION_CONFIG_MODELS:
            model.generation_config = GenerationConfig(temperature=None, top_p=None)
    except:
        print(f"Error setting generation config for model {model_name}")
        pass


def resize_if_needed(model_name, model, token_nums):
    try:
        if model_name in MIS_MATCH_VOCAB_SIZE_MODELS:
            model.resize_token_embeddings(token_nums)
    except:
        print(f"Error resizing token embeddings for model {model_name}")
        pass


def init_wandb(train_request: Dict):
    # set wandb_mode=offline; do not upload the data to wandb export WANDB_MODE=offline
    return True
    task_id = train_request["task_id"]
    expected_repo_name = train_request["expected_repo_name"]
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = train_request["wandb_log_dir"]
    os.environ["WANDB_RUN_ID"] = f"{task_id}_{expected_repo_name}"
    os.environ["WANDB_NAME"] = f"{task_id}_{expected_repo_name}"
    if is_main_process(LOCAL_RANK):
        os.makedirs(train_request["wandb_log_dir"], exist_ok=True)
    return True