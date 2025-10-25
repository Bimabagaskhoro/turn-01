import time
import logging
import numpy as np
import torch
import gc
from torch.optim import AdamW
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BatchSizeState:
    current_lr: float | None = None
    found_lr: bool = False


batch_size_state = BatchSizeState()


def analyze_loss_landscape(model, batch, lr, trainer, num_points=5, radius=0.1, lora=True):
    """
    Perform a quick analysis of the loss landscape around the given learning rate.

    Args:
        model: The PyTorch model
        batch: A batch of data
        lr: Current learning rate to analyze around
        trainer: Hugging Face Trainer instance
        num_points: Number of points to evaluate
        radius: Relative radius around lr to analyze
        lora: If True, only consider LoRA and lm_head parameters

    Returns:
        Dictionary with landscape metrics
    """
    # Save original parameters
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

    # Get initial loss and gradient
    loss = trainer.compute_loss(model, batch)
    loss.backward()

    # Calculate gradient norm for relevant parameters
    grad_params = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if lora:
                if "lora_" in name or "lm_head" in name:
                    grad_params.append(torch.norm(param.grad))
            else:
                grad_params.append(torch.norm(param.grad))

    grad_norm = torch.norm(torch.stack(grad_params)) if grad_params else torch.tensor(0.0)

    # Evaluate loss at different learning rates around the current one
    lr_factors = np.linspace(1 - radius, 1 + radius, num_points)
    losses = []

    for factor in lr_factors:
        current_lr = lr * factor

        # Apply the current learning rate as a gradient step
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if lora:
                        if "lora_" in name or "lm_head" in name:
                            param.data = original_params[name] - current_lr * param.grad
                    else:
                        param.data = original_params[name] - current_lr * param.grad

        # Calculate loss at this point
        loss_val = float(trainer.compute_loss(model, batch))
        losses.append(loss_val)

        # Restore original parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])

    # Calculate landscape metrics
    losses = np.array(losses)
    min_idx = np.argmin(losses)

    # Calculate smoothness (average of second derivatives)
    smoothness = np.mean(np.abs(np.diff(losses, 2))) if len(losses) > 2 else float("inf")

    # Calculate convexity (average of second derivatives)
    convexity = np.mean(np.diff(losses, 2)) if len(losses) > 2 else 0

    # Check if minimum is near the center
    is_centered = min_idx in [num_points // 2 - 1, num_points // 2, num_points // 2 + 1]

    metrics = {
        "smoothness": smoothness,
        "convexity": convexity,
        "centered": is_centered,
        "grad_norm": float(grad_norm),
        "min_loss": float(np.min(losses)),
        "min_lr_factor": lr_factors[min_idx],
    }

    # Clean up to prevent memory leaks
    del original_params
    torch.cuda.empty_cache()

    return metrics


def find_lr_and_continue(
    trainer,
    start_lr: float = 1e-6,  # Increased minimum to avoid too small values
    end_lr: float = 1e-3,    # Adjusted maximum for typical LoRA fine-tuning
    time_budget_minutes: float = 5.0,
    warmup_fraction: float = 0.1,
    num_candidates: int = 2,
    preserve_state: bool = True,
    lora: bool = True
):
    """
    Enhanced learning rate finder based on Leslie Smith's approach with less conservative selection.

    Args:
        trainer: Hugging Face Trainer instance
        start_lr: Minimum learning rate to test
        end_lr: Maximum learning rate to test (may not reach this if loss diverges)
        time_budget_minutes: Maximum time for the LR finder in minutes
        warmup_fraction: Not used in this implementation but kept for interface compatibility
        num_candidates: Not used in this implementation but kept for interface compatibility
        preserve_state: Whether to save and restore model state
        lora: If True, only update LoRA and lm_head parameters

    Returns:
        Tuple of (selected learning rate, best state dict, None, number of evaluations)
    """
    global batch_size_state
    model = trainer.model
    time_budget_seconds = time_budget_minutes * 60
    start_time = time.time()

    # Check if we're restarting after OOM
    if batch_size_state.current_lr is not None:
        logger.info(f"Restarting after OOM with previous LR: {batch_size_state.current_lr:.2e}")
        return batch_size_state.current_lr, None, None, 0

    # Save initial state
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

    # Initialize results dictionary
    results = {}
    best_state_dict = None

    # Determine step size and number of evaluations based on time budget
    # First, measure time for a single step
    test_steps = 3
    dataloader_iter = iter(trainer.get_train_dataloader())
    batch = next(dataloader_iter)
    batch = trainer._prepare_inputs(batch)

    model.train()
    optimizer = AdamW(model.parameters(), lr=start_lr)

    # Measure time for a few test steps
    test_start_time = time.time()
    for _ in range(test_steps):
        optimizer.zero_grad()
        loss = trainer.compute_loss(model, batch)
        loss.backward()
        optimizer.step()

    time_per_step = (time.time() - test_start_time) / test_steps
    logger.info(f"Time per step: {time_per_step:.4f} seconds")

    # Calculate how many evaluations we can afford
    # We'll use 20% of time for the final validation
    main_time_budget = time_budget_seconds * 0.8
    max_evaluations = int(main_time_budget / (time_per_step * 5))  # 5 steps per evaluation
    max_evaluations = min(max(10, max_evaluations), 30)

    logger.info(f"Planning to perform up to {max_evaluations} LR evaluations")

    # Reset model parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if lora and ("lora_" in name or "lm_head" in name) and name in initial_state:
                param.copy_(initial_state[name].to(param.device))
            elif not lora and name in initial_state:
                param.copy_(initial_state[name].to(param.device))

    # Calculate the multiplier to get from start_lr to end_lr in max_evaluations steps
    lr_multiplier = np.power(end_lr / start_lr, 1 / max_evaluations)

    # List to track learning rates and losses
    lrs = []
    losses = []
    smoothed_losses = []
    raw_losses = []  # Keep raw losses for fine-grained analysis

    # Variables to track best loss
    best_loss = float('inf')

    # Variables to detect divergence
    min_loss_so_far = float('inf')
    divergence_counter = 0

    # Steps per evaluation - adaptive based on the phase
    steps_per_evaluation = 5

    # Start with initial learning rate
    current_lr = start_lr
    evaluation_count = 0

    # Learning rate range test loop
    while evaluation_count < max_evaluations and time.time() - start_time < main_time_budget:
        evaluation_count += 1
        logger.info(f"Evaluation {evaluation_count}/{max_evaluations}, LR: {current_lr:.2e}")

        # Create optimizer with current learning rate
        optimizer = AdamW(model.parameters(), lr=current_lr)

        # Train for a few steps
        batch_losses = []
        for step in range(steps_per_evaluation):
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

            batch_losses.append(float(loss))

        # Calculate average loss for this learning rate
        avg_loss = np.mean(batch_losses)
        raw_losses.extend(batch_losses)  # Keep all raw loss values

        # Save results
        lrs.append(current_lr)
        losses.append(avg_loss)

        # Calculate smoothed loss - use lighter smoothing (0.7 instead of 0.8)
        if len(losses) == 1:
            smoothed_losses.append(avg_loss)
        else:
            smoothed_losses.append(0.7 * smoothed_losses[-1] + 0.3 * avg_loss)

        logger.info(f"LR: {current_lr:.2e}, Loss: {avg_loss:.4f}, Smoothed Loss: {smoothed_losses[-1]:.4f}")

        # Save model state if this is the best loss so far and we're preserving state
        if preserve_state and avg_loss < best_loss:
            # Free up previous best state dict if it exists
            if best_state_dict is not None:
                del best_state_dict
                torch.cuda.empty_cache()

            best_loss = avg_loss
            best_state_dict = {
                name: param.detach().cpu().clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }

        # Check for loss divergence - LESS CONSERVATIVE

# Check for loss divergence - LESS CONSERVATIVE
        if avg_loss < min_loss_so_far:
            min_loss_so_far = avg_loss
            divergence_counter = 0
        else:
            # Only count as increasing if the increase is significant
            # Change from 1.05 (5%) to 1.10 (10%)
            if avg_loss > min_loss_so_far * 1.10:  # <-- CHANGE HERE: 1.05 to 1.10
                divergence_counter += 1
            else:
                # Small fluctuations don't count as divergence
                divergence_counter = 0

# More permissive criteria for stopping the search
# Handle negative losses (common in GRPO/reward-based training)
        # For negative losses, we want to avoid early stopping due to "divergence"
        # because negative losses can be normal and better (higher rewards)
        if min_loss_so_far >= 0:
            # Traditional logic for positive losses
            divergence_threshold = 6.0 * min_loss_so_far
        else:
            # For negative losses, use absolute value logic
            # Stop only if loss becomes significantly more negative (worse reward)
            divergence_threshold = min_loss_so_far / 6.0  # More negative = worse
            
        if avg_loss > divergence_threshold or np.isnan(avg_loss) or np.isinf(avg_loss):
            logger.info(f"Stopping early due to significant loss divergence at LR: {current_lr:.2e}")
            logger.info(f"avg_loss={avg_loss:.6f}, threshold={divergence_threshold:.6f}, min_loss_so_far={min_loss_so_far:.6f}")
            break

        if divergence_counter >= 8:
            logger.info(f"Stopping early due to consistent significant loss increase at LR: {current_lr:.2e}")
            break

        # Increase learning rate
        current_lr *= lr_multiplier

    # Analyze results to find optimal learning rate automatically
    if len(lrs) < 3:
        logger.warning("Too few learning rates evaluated, using default value")
        final_lr = 8e-5  # Slightly higher default than before (was 6e-5)
    else:
        # Find the point of steepest descent on the loss curve
        smoothed_losses = np.array(smoothed_losses)
        lrs = np.array(lrs)

        # Calculate gradients of the loss curve
        gradients = np.gradient(smoothed_losses) / np.gradient(np.log10(lrs))

        # Find the point of steepest descent (most negative gradient)
        # Use windowed approach to be more robust to noise
        window_size = min(3, len(gradients) // 4)  # Keep window small enough
        windowed_gradients = [np.mean(gradients[max(0, i-window_size):min(len(gradients), i+window_size+1)])
                             for i in range(len(gradients))]
        steepest_idx = np.argmin(windowed_gradients)

        # Find the minimum loss point
        min_loss_idx = np.argmin(smoothed_losses)

        # Analyze the overall shape of the loss curve
        # If loss explodes at the end, we've explored well into high LR territory
        if len(smoothed_losses) > 5:
            loss_end_to_min_ratio = smoothed_losses[-1] / np.min(smoothed_losses)
            loss_exploration_quality = min(1.0, loss_end_to_min_ratio / 10.0)  # Scale from 0 to 1
        else:
            # Initialize variables for short runs
            loss_end_to_min_ratio = 1.0  # Default neutral ratio
            loss_exploration_quality = 0.5  # Neutral value for short runs

        # The learning rate is typically chosen to be at or slightly before the steepest descent point
        if steepest_idx > 1:  # Make sure we have enough points
            # Use a dynamic divider based on loss curve shape instead of fixed 2.0
            # If we have good exploration (loss increases at end), use a smaller divider
            if loss_exploration_quality > 0.8:
                # We explored well into high learning rates, be more aggressive
                divider = 1.3
                logger.info(f"Loss curve shows good exploration (ratio={loss_end_to_min_ratio:.1f}), using aggressive divider={divider:.1f}")
            else:
                # Be more conservative when the loss curve doesn't show clear divergence
                divider = 1.8
                logger.info(f"Loss curve shows limited exploration (ratio={loss_end_to_min_ratio:.1f}), using moderate divider={divider:.1f}")

            # Use the steepest descent point directly, or one step before
            final_idx = max(0, steepest_idx)
            candidate_lr = lrs[final_idx] / divider

            # Less conservative bound - reduced from 1/3 to 1/2
            min_bound = lrs[min_loss_idx] / 2.0

            # Take the maximum to avoid going too low
            final_lr = max(candidate_lr, min_bound)

            logger.info(f"Steepest descent at LR: {lrs[steepest_idx]:.2e}")
            logger.info(f"Minimum loss at LR: {lrs[min_loss_idx]:.2e}")
            logger.info(f"Selected LR: {final_lr:.2e} (after applying divider of {divider:.1f})")
        else:
            # If we don't have enough points, choose a reasonable default
            # based on the minimum loss point with a smaller divider
            min_loss_idx = np.argmin(smoothed_losses)
            final_lr = lrs[min_loss_idx] / 2.0  # Less conservative than before (was 3.0)
            logger.info(f"Limited data, selecting LR based on minimum loss: {final_lr:.2e}")

    # Validate the learning rate with a quick landscape analysis
    remaining_time = time_budget_seconds - (time.time() - start_time)
    if remaining_time > 60:  # Only if we have at least a minute left
        logger.info(f"Validating learning rate with landscape analysis: {final_lr:.2e}")
        try:
            batch = next(dataloader_iter)
            batch = trainer._prepare_inputs(batch)

            # Reset model parameters to initial state
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if lora and ("lora_" in name or "lm_head" in name) and name in initial_state:
                        param.copy_(initial_state[name].to(param.device))
                    elif not lora and name in initial_state:
                        param.copy_(initial_state[name].to(param.device))

            landscape = analyze_loss_landscape(model, batch, final_lr, trainer, lora=lora)

            # Adjust learning rate if the minimum is not centered - more permissive adjustment
            if not landscape["centered"]:
                # Cap the adjustment to prevent extreme changes but wider bounds
                adjustment = max(0.6, min(1.5, landscape["min_lr_factor"]))
                logger.info(f"Adjusting LR by factor {adjustment:.2f} based on landscape analysis")
                final_lr *= adjustment

            # Less conservative lower bound for LoRA fine-tuning
            model_size_based_min = 5e-6
            # Try to get model size information if available
            try:
                if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                    hidden_size = model.config.hidden_size
                    if hidden_size > 2048:
                        model_size_based_min = 1e-5  # Larger models can handle higher LRs
            except:
                pass

            # Final sanity check - wider range of reasonable minimums
            reasonable_minimum = model_size_based_min
            if final_lr < reasonable_minimum:
                logger.info(f"Adjusting selected LR ({final_lr:.2e}) to minimum threshold ({reasonable_minimum:.2e})")
                final_lr = reasonable_minimum

            logger.info(f"Landscape analysis: smoothness={landscape['smoothness']:.4f}, "
                       f"convexity={landscape['convexity']:.4f}, centered={landscape['centered']}")
        except Exception as e:
            logger.warning(f"Error during landscape analysis: {e}")

    # Add a small bias toward higher learning rates to counter conservatism
    # This makes it slightly less conservative by default
    final_lr *= 1.15
    logger.info(f"Applied general 15% boost to counteract conservative tendencies: {final_lr:.2e}")

    # Clean up memory
    del initial_state
    torch.cuda.empty_cache()
    gc.collect()

    # Set the batch size state
    batch_size_state.current_lr = final_lr
    batch_size_state.found_lr = True

    logger.info(f"Final selected learning rate: {final_lr:.2e}")
    return final_lr, best_state_dict, None, evaluation_count

