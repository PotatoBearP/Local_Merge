import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from methods.utility import load_model_weights
import numpy as np
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def topk_values_mask(input_tensor, frac=0.7):
    """
    Creates a mask for the top K% values in the tensor by magnitude.
    
    Args:
        input_tensor: Input tensor
        K: Fraction of values to keep (0.7 means keep top 70%)
    
    Returns:
        tuple: (masked tensor, boolean mask)
    """
    if frac > 1:
        frac /= 100
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if needed
    n, d = input_tensor.shape
    k = d - int(d * frac)  # Calculate threshold index
    kth_values, _ = input_tensor.abs().kthvalue(k, dim=1, keepdim=True)  # Find threshold values
    mask = input_tensor.abs() >= kth_values  # Create mask for values above threshold
    return input_tensor * mask, mask

def resolve_zero_signs(signs: torch.Tensor, method="majority") -> torch.Tensor:
    """
    Resolves zero signs in a tensor based on the majority or minority rule.
    
    Args:
        signs: Tensor containing -1, 0, and 1 values
        method: Either "majority" or "minority" to determine how to handle zeros
    """
    majority = torch.sign(signs.sum())  # Get overall sign direction
    if method == "majority":
        signs[signs == 0] = majority  # Set zeros to majority sign
    elif method == "minority":
        signs[signs == 0] = -majority  # Set zeros to opposite of majority sign
    return signs

def disjoint_merge(Tensor: torch.Tensor, method="mean", sign: torch.Tensor = None) -> torch.Tensor:
    """
    Merges tensor values using different aggregation methods while respecting signs.
    
    Args:
        Tensor: Input tensor to merge
        method: Aggregation method ("mean", "sum", or "max")
        sign: Optional tensor of signs to guide the merge
    """
    # Select values based on sign consistency
    if sign is not None:
        rows_to_keep = torch.where(sign.unsqueeze(0) > 0, Tensor > 0, Tensor < 0)
        selected = Tensor * rows_to_keep
    else:
        selected = Tensor * (Tensor != 0)  # Keep non-zero values

    # Apply the chosen aggregation method
    if method == "mean":
        count = (selected != 0).sum(dim=0).float()  # Count non-zero elements
        return selected.sum(dim=0) / torch.clamp(count, min=1.0)  # Average, avoiding div by zero
    elif method == "sum":
        return selected.sum(dim=0)  # Simple sum
    elif method == "max":
        return selected.abs().max(dim=0)[0] * sign  # Max magnitude while preserving sign
    else:
        raise ValueError(f"Unknown method: {method}")

def ties_merge_models(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    reset_thresh: float = 0.7,
    merge_func: str = "mean"
):
    """
    Merges two models using TIES (Tensor-Independent Efficient Scaling) method.
    
    The process involves:
    1. Loading both models' weights
    2. Flattening compatible tensors
    3. Applying magnitude-based masking
    4. Resolving sign conflicts
    5. Merging using specified method
    6. Reconstructing and saving the final model
    """
    logger.info(f"Loading weights from model A: {model_a_path}")
    state_dict_a = load_model_weights(model_a_path)

    logger.info(f"Loading weights from model B: {model_b_path}")
    state_dict_b = load_model_weights(model_b_path)

    shared_keys = set(state_dict_a.keys()) & set(state_dict_b.keys())
    flat_a, flat_b = [], []

    for key in sorted(shared_keys):
        t1, t2 = state_dict_a[key], state_dict_b[key]
        if t1.shape == t2.shape and t1.dtype in (torch.float16, torch.float32, torch.bfloat16):
            flat_a.append(t1.view(-1).float())
            flat_b.append(t2.view(-1).float())
        else:
            logger.warning(f"Skipping {key} due to shape/type mismatch")

    stacked = torch.stack([torch.cat(flat_a), torch.cat(flat_b)])  # shape: (2, num_params)

    logger.info("Applying top-k masking")
    masked_tensor, mask = topk_values_mask(stacked, K=reset_thresh)

    logger.info("Resolving consistent signs")
    signs = resolve_zero_signs(masked_tensor)

    logger.info(f"Merging with method: {merge_func}")
    merged_flat = disjoint_merge(masked_tensor, method=merge_func, sign=signs)

    logger.info("Reconstructing merged state dict")
    merged_state_dict = {}
    idx = 0
    for key in sorted(shared_keys):
        t1 = state_dict_a[key]
        numel = t1.numel()
        merged_tensor = merged_flat[idx:idx+numel].view(t1.shape).to(dtype=t1.dtype)
        merged_state_dict[key] = merged_tensor
        idx += numel

    logger.info(f"Saving merged model to {output_path}")
    model = AutoModelForCausalLM.from_pretrained(model_a_path)
    tokenizer = AutoTokenizer.from_pretrained(model_a_path)
    model.load_state_dict(merged_state_dict)
    model = model.to(torch.bfloat16)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)