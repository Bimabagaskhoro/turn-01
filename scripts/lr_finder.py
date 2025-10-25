import time
import logging
import numpy as np
from src.run_container import start_training_container
import torch
import gc
from torch.optim import AdamW
from lion_pytorch import Lion
from sophia import SophiaG
from dataclasses import dataclass
from transformers import Trainer

logger = logging.getLogger(__name__)


@dataclass
class BatchSizeState:
    current_lr: float | None = None
    found_lr: bool = False


batch_size_state = BatchSizeState()


def quick_landscape_analysis(
    model, batch, current_lr, trainer, num_points=5, radius=0.1, lora=True
):
    """
    Perform a quick analysis of the loss landscape around the current parameters.
    When lora=True, only the LoRA and lm_head parameters are saved and perturbed.
    When lora=False, all parameters are saved and perturbed.
    """
    # Save original parameters (only LoRA and lm_head if lora=True, otherwise all parameters)
    if lora:
        original_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if "lora_" in name or "lm_head" in name
        }
    else:
        original_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

    losses = []
    perturbation_factors = np.linspace(1 - radius, 1 + radius, num_points)

    loss = trainer.compute_loss(model, batch)
    loss.backward()

    # Compute gradient norm only for relevant parameters with gradients
    grad_params = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if lora:
                if "lora_" in name or "lm_head" in name:
                    grad_params.append(torch.norm(param.grad))
            else:
                grad_params.append(torch.norm(param.grad))

    grad_norm = torch.norm(torch.stack(grad_params)) if grad_params else torch.tensor(0.0)

    logger.debug(f"Initial loss: {loss.item():.4f}, grad norm: {grad_norm.item():.4f}")

    for factor in perturbation_factors:
        lr = current_lr * factor
        # Perturb only the relevant parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if lora:
                        if "lora_" in name or "lm_head" in name:
                            param.data = original_params[name] - lr * param.grad
                    else:
                        param.data = original_params[name] - lr * param.grad

        loss_val = float(trainer.compute_loss(model, batch))
        losses.append(loss_val)
        logger.debug(
            f"Perturbation factor: {factor:.2f}, lr: {lr:.2e}, loss: {loss_val:.4f}"
        )

        # Restore original parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])

    losses = np.array(losses)
    smoothness = (
        np.mean(np.abs(np.diff(losses, 2))) if len(losses) > 2 else float("inf")
    )
    convexity = np.mean(np.diff(losses, 2)) if len(losses) > 2 else 0
    min_idx = np.argmin(losses)

    landscape_metrics = {
        "smoothness": smoothness,
        "convexity": convexity,
        "centered": min_idx
        in [num_points // 2 - 1, num_points // 2, num_points // 2 + 1],
        "grad_norm": float(grad_norm),
        "min_loss": float(np.min(losses)),
    }

    # Clean up to prevent memory leaks
    del original_params
    torch.cuda.empty_cache()

    return landscape_metrics


def evaluate_single_lr(
    lr,
    dataloader_iter,
    steps,
    trainer,
    results,
    initial_state,
    start_time,
    model,
    preserve_state=True,
    lora=True,
):
    """Helper function to evaluate a single learning rate"""
    logger.info(f"Evaluating LR: {lr:.2e} for {steps} steps")
    best_state_dict = None
    best_loss = float("inf")

    # Reset parameters to their initial state
    with torch.no_grad():
        for name, param in model.named_parameters():
            if lora:
                if ("lora_" in name or "lm_head" in name) and name in initial_state:
                    param.copy_(initial_state[name].to(param.device))
            else:
                if name in initial_state:
                    param.copy_(initial_state[name].to(param.device))

    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    losses = []

    try:
        first_batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(trainer.get_validation_dataloader())
        first_batch = next(dataloader_iter)
    first_batch = trainer._prepare_inputs(first_batch)

    for step in range(steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(trainer.get_train_dataloader())
            batch = next(dataloader_iter)

        batch = trainer._prepare_inputs(batch)
        optimizer.zero_grad()
        loss = trainer.compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

        logger.debug(f"Step {step + 1}/{steps}, LR: {lr:.2e}, Loss: {loss.item():.4f}")

        if preserve_state and float(loss) < best_loss:
            # Only save the best state, overwrite previous best to save memory
            if best_state_dict is not None:
                del best_state_dict
                torch.cuda.empty_cache()

            best_loss = float(loss)
            # Store parameters on CPU to save GPU memory
            best_state_dict = {
                name: param.detach().cpu().clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }

    mean_loss = np.mean(losses)
    results[float(lr)] = {"loss": mean_loss, "losses": losses}

    # Only store the state dict in results if it's the best seen so far
    if "global_best_loss" not in results or mean_loss < results["global_best_loss"]:
        results["global_best_loss"] = mean_loss
        results["global_best_state_dict"] = best_state_dict
        results["global_best_lr"] = float(lr)
    elif best_state_dict is not None:
        # We don't need this anymore if it's not the global best
        del best_state_dict
        torch.cuda.empty_cache()

    logger.info(
        f"Finished evaluating LR: {lr:.2e}, Mean Loss: {mean_loss:.4f}, Time: {time.time() - start_time:.2f}s"
    )

    # Clean up to prevent memory leaks
    torch.cuda.empty_cache()
    gc.collect()

    return (
        mean_loss,
        dataloader_iter,
        best_state_dict
        if "global_best_lr" in results and results["global_best_lr"] == float(lr)
        else None,
    )

def find_lr_and_continue(
    trainer,
    start_lr: float = 5e-6,
    end_lr: float = 1e-4,
    time_budget_minutes: float = 5.0,
    warmup_fraction: float = 0.1,
    num_candidates: int = 2,
    preserve_state: bool = True,
    lora: bool = True
):
    """
    Enhanced learning rate finder that explores multiple promising candidates.
    Memory-optimized version that only keeps the best state dict.
    Performs adaptive boundary exploration during the probe phase.
    When lora=True, only LoRA and lm_head parameters are considered.
    When lora=False, all parameters are considered.
    """
    global batch_size_state
    model = trainer.model
    time_budget_seconds = time_budget_minutes * 60

    if batch_size_state.current_lr is not None:
        logger.info(
            f"Restarting after OOM with previous LR: {batch_size_state.current_lr:.2e}"
        )
        return batch_size_state.current_lr, None, None, 0

    # Save initial state based on lora flag
    if lora:
        logger.info("Saving only LoRA and lm_head parameters")
        initial_state = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if ("lora_" in name or "lm_head" in name) and param.requires_grad
        }
    else:
        logger.info("Saving all parameters")
        initial_state = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    # Initial evaluation to estimate time per step
    dataloader_iter = iter(trainer.get_train_dataloader())
    start_time = time.time()
    test_steps = 2
    results = {}

    loss, dataloader_iter, _ = evaluate_single_lr(
        start_lr,
        dataloader_iter,
        test_steps,
        trainer,
        results,
        initial_state,
        start_time,
        model,
        preserve_state,
        lora,
    )

    eval_time = time.time() - start_time
    time_per_step = eval_time / test_steps

    available_time = time_budget_seconds

    probe_time_budget = available_time * 0.6
    probe_steps = 7
    max_probe_points = 10
    min_probe_points = 4

    affordable_probe_points = int(probe_time_budget / (time_per_step * probe_steps))
    probe_points = min(max(min_probe_points, affordable_probe_points), max_probe_points)
    if probe_points > 7:
        end_lr *= 1.5
        start_lr *= 0.5

    logger.info(f"Using {probe_points} probe points")

    # Initial probe phase
    logger.info("Starting initial probe phase")
    probe_lrs = np.geomspace(start_lr, end_lr, probe_points)
    np.random.shuffle(probe_lrs)  # Shuffle to avoid order bias
    probe_results = {}

    # Function to evaluate and track best learning rate during probing
    def evaluate_probe_lr(lr):
        loss, dl_iter, _ = evaluate_single_lr(
            float(lr),
            dataloader_iter,
            probe_steps,
            trainer,
            probe_results,
            initial_state,
            start_time,
            model,
            False,  # Don't preserve state during probe
            lora,  # Pass the lora flag
        )
        logger.debug(f"Probe LR: {lr:.2e}, loss: {loss:.4f}")

        # Clean up between evaluations
        torch.cuda.empty_cache()
        gc.collect()

        return loss, dl_iter

    # Evaluate all initial probe points
    for lr in probe_lrs:
        _, dataloader_iter = evaluate_probe_lr(lr)

        if time.time() - start_time > time_budget_seconds * 0.4:  # Leave time for exploration
            logger.warning("Time budget partially exceeded during initial probe phase")
            break

    # Adaptive exploration of boundaries
    # Sort the evaluated learning rates by loss
    numeric_keys = [key for key in probe_results.keys() if isinstance(key, (int, float))]
    sorted_probe_results = sorted(
        [(float(lr), probe_results[lr]["loss"]) for lr in numeric_keys],
        key=lambda x: x[1],
    )

    if not sorted_probe_results:
        logger.warning("No valid probe results found. Using default learning rate.")
        final_lr = 6e-5
        batch_size_state.current_lr = final_lr
        batch_size_state.found_lr = True
        return final_lr, None, None, probe_points

    # Get current min/max LRs and best LR
    min_lr = min(lr for lr, _ in sorted_probe_results)
    max_lr = max(lr for lr, _ in sorted_probe_results)
    best_lr, best_loss = sorted_probe_results[0]

    logger.info(f"Initial probe results - Best LR: {best_lr:.2e}, Loss: {best_loss:.4f}")

    logger.info(f"Updated LR range after boundary exploration: {min_lr:.2e} to {max_lr:.2e}")
    logger.info(f"Best probe loss: {best_loss:.4f} at LR {best_lr:.2e}")

    # Select candidates based on probe results with tolerance threshold
    try:
        tolerance_threshold = best_loss * 1.1
        logger.info(f"Tolerance threshold (10%): {tolerance_threshold:.4f}")

        # Find the highest learning rate within tolerance threshold
        highest_viable_lr = None
        highest_viable_loss = None

        for lr, loss in sorted_probe_results:
            if loss <= tolerance_threshold and (
                highest_viable_lr is None or lr > highest_viable_lr
            ):
                highest_viable_lr = lr
                highest_viable_loss = loss
                logger.debug(f"New highest viable LR: {lr:.2e} with loss {loss:.4f}")

        # Create the candidates list, starting with the best performing LR
        candidates = [(best_lr, best_loss)]

        # Check if best_lr is at the min or max of the probe range
        is_at_min = best_lr == min_lr
        is_at_max = best_lr == max_lr

        if is_at_min:
            # If best LR is the smallest, add half of it as a candidate
            half_lr = best_lr / 2
            logger.info(f"Best LR is at minimum. Adding half LR {half_lr:.2e} as a candidate")
            candidates.append((half_lr, float("inf")))  # Loss will be evaluated later
        elif is_at_max:
            # If best LR is the largest, add double of it as a candidate
            double_lr = best_lr * 2
            logger.info(f"Best LR is at maximum. Adding double LR {double_lr:.2e} as a candidate")
            candidates.append((double_lr, float("inf")))  # Loss will be evaluated later

        # Now continue with the normal logic for additional candidates
        # Add the highest viable LR if it's different from the best
        if highest_viable_lr is not None and highest_viable_lr != best_lr:
            candidates.append((highest_viable_lr, highest_viable_loss))

            # Calculate the exact middle point between best and highest viable
            middle_lr = (best_lr + highest_viable_lr) / 2

            # Use the exact middle point as our third candidate
            middle_candidate = (
                middle_lr,
                float("inf"),
            )  # Actual loss will be determined during detailed evaluation

            # Add the middle candidate to our list
            candidates.append(middle_candidate)
            logger.info(
                f"Selected candidates: best {best_lr:.2e}, middle {middle_lr:.2e}, and highest viable {highest_viable_lr:.2e}"
            )
        else:
            # If there's no distinct highest viable or only one candidate
            if len(sorted_probe_results) > 1 and not (is_at_min or is_at_max):
                # Only add second best if we haven't already added a candidate for min/max case
                candidates.append(sorted_probe_results[1])
                logger.info(
                    f"No distinct highest viable LR found. Using best {best_lr:.2e} and second best {sorted_probe_results[1][0]:.2e}"
                )
            elif not (is_at_min or is_at_max):
                logger.info(f"Only one candidate available: {best_lr:.2e}")

        logger.info(
            f"Selected candidates for detailed exploration: {[lr for lr, _ in candidates]}"
        )

    except Exception as e:
        logger.warning(f"Error during candidate selection: {str(e)}")
        logger.warning("Learning rate finder failed: " + str(e))
        logger.info("Using default learning rate: 8.00e-05")

        # Make sure to clean up resources even when throwing
        try:
            del initial_state
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

        final_lr = 8e-5
        batch_size_state.current_lr = final_lr
        batch_size_state.found_lr = True
        return final_lr, None, None, probe_points

    # Calculate time left for detailed exploration
    remaining_time = time_budget_seconds - (time.time() - start_time)
    explore_steps = int((remaining_time / len(candidates)) / time_per_step)
    explore_steps = min(max(15, explore_steps), 150)  # Keep reasonable bounds

    logger.info(f"Using {explore_steps} steps for detailed evaluation")

    # Explore phase: Detailed evaluation of promising candidates
    explore_results = {}
    best_lr = None
    best_loss = float("inf")
    global_best_state_dict = None

    for lr, _ in candidates:
        loss, dataloader_iter, state_dict = evaluate_single_lr(
            float(lr),
            dataloader_iter,
            explore_steps,
            trainer,
            explore_results,
            initial_state,
            start_time,
            model,
            preserve_state,
            lora,  # Pass the lora flag
        )

        if loss < best_loss:
            best_loss = loss
            best_lr = float(lr)
            # Keep only the best state dict to save memory
            if global_best_state_dict is not None:
                del global_best_state_dict
                torch.cuda.empty_cache()
            global_best_state_dict = state_dict

        if time.time() - start_time > time_budget_seconds:
            logger.warning("Time budget exceeded during explore phase")
            break

        # Clean up between evaluations
        torch.cuda.empty_cache()
        gc.collect()

    try:
        # Select final learning rate - MODIFIED HERE to use 2% tolerance
        # Prefer higher learning rates when performance is similar
        tolerance = 0.02

        # Filter out special keys from explore_results too
        numeric_keys = [
            key for key in explore_results.keys() if isinstance(key, (int, float))
        ]
        final_candidates = sorted(
            [
                (float(lr), explore_results[lr]["loss"])
                for lr in numeric_keys
                if "early_stop" not in explore_results[lr]
            ],
            key=lambda x: x[1],
        )

        if not final_candidates:
            logger.warning(
                "All detailed explorations failed. Using default fallback learning rate."
            )
            final_lr = 6e-5
        else:
            best_loss = final_candidates[0][1]  # Get the best loss value
            final_lr = float(final_candidates[0][0])  # Start with best performing

            logger.info(
                f"Best loss: {best_loss:.4f}, initial LR selection: {final_lr:.2e}"
            )
            logger.info(f"Using tolerance of exactly 2% ({best_loss * 1.02:.4f})")

            # Consider higher learning rates within tolerance of the best loss
            for lr, loss in final_candidates[1:]:
                if loss <= best_loss * 1.02 and lr > final_lr:
                    logger.info(
                        f"Found better LR: {lr:.2e} with loss: {loss:.4f} (within 2% of best: {best_loss:.4f})"
                    )
                    final_lr = float(lr)  # Prefer higher LR if within tolerance

    except Exception as e:
        logger.warning(f"Error during final learning rate selection: {str(e)}")
        logger.warning("Learning rate finder failed: " + str(e))
        logger.info("Using default learning rate: 8.00e-05")

        # Make sure to clean up resources even when throwing
        try:
            del initial_state
            if global_best_state_dict is not None:
                del global_best_state_dict
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

        final_lr = 8e-5
        batch_size_state.current_lr = final_lr
        batch_size_state.found_lr = True
        return final_lr, None, None, probe_points

    batch_size_state.current_lr = float(final_lr)
    batch_size_state.found_lr = True

    logger.info(f"Selected final LR: {final_lr:.2e} with loss: {best_loss:.4f}")

    # Clean up results dictionary to free memory
    for lr in explore_results:
        if isinstance(lr, (int, float)) and "losses" in explore_results[lr]:
            del explore_results[lr]["losses"]

    # Final cleanup
    del probe_results
    del explore_results
    del initial_state
    torch.cuda.empty_cache()
    gc.collect()

    return float(final_lr), global_best_state_dict, None, probe_points

