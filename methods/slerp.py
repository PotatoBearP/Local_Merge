import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from methods.utility import load_model_weights
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalizes a tensor to unit length to avoid numerical instability.

    Args:
        tensor (torch.Tensor): The input tensor.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    norm = torch.norm(tensor)
    return tensor / (norm + eps) if norm > eps else tensor

def slerp_tensor(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
    """
    Performs spherical linear interpolation (SLERP) between two tensors.

    Args:
        t (float): Interpolation factor between 0.0 and 1.0.
        v0 (torch.Tensor): First tensor (base).
        v1 (torch.Tensor): Second tensor.
        dot_threshold (float): Threshold to fall back to LERP if vectors are nearly parallel.

    Returns:
        torch.Tensor: Interpolated tensor.
    """
    v0_np = v0.detach().cpu().float().numpy()
    v1_np = v1.detach().cpu().float().numpy()

    v0n = v0_np / (np.linalg.norm(v0_np) + 1e-8)
    v1n = v1_np / (np.linalg.norm(v1_np) + 1e-8)
    dot = np.sum(v0n * v1n)

    # Fall back to linear interpolation if vectors are nearly colinear
    if np.abs(dot) > dot_threshold:
        return ((1 - t) * v0 + t * v1).to(v0.dtype)

    # Compute spherical interpolation
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = np.sin(theta_t) / sin_theta_0

    result_np = s0 * v0_np + s1 * v1_np
    return torch.from_numpy(result_np).to(v0.device).to(v0.dtype)

def slerp_models(
    model_a_path: str, 
    model_b_path: str, 
    output_path: str, 
    t: float = 0.5,
    mask_path: str = None
):
    """
    Merges two models using SLERP and saves the result.

    Args:
        model_a_path (str): Path to the base model.
        model_b_path (str): Path to the target model.
        output_path (str): Path to save the merged model.
        t (float): Interpolation parameter (0 = all A, 1 = all B).
        mask_path (str, optional): Path to the saved tensor masks.
    """
    logger.info(f"Loading weights from model A: {model_a_path}")
    state_dict_a = load_model_weights(model_a_path)

    logger.info(f"Loading weights from model B: {model_b_path}")
    state_dict_b = load_model_weights(model_b_path)

    # Load masks if provided
    masks = None
    if mask_path:
        logger.info(f"Loading masks from: {mask_path}")
        masks = torch.load(mask_path)

    merged_state_dict = {}

    logger.info("Performing SLERP on model weights...")
    for key in state_dict_a.keys():
        if key not in state_dict_b:
            logger.warning(f"Skipping {key}: not found in model B")
            merged_state_dict[key] = tensor_a
            continue

        tensor_a = state_dict_a[key]
        tensor_b = state_dict_b[key]

        if tensor_a.shape != tensor_b.shape:
            logger.warning(f"Shape mismatch for {key}: {tensor_a.shape} vs {tensor_b.shape}")
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(tensor_a.shape, tensor_b.shape))
            sliced_a = tensor_a
            sliced_b = tensor_b
            for dim, size in enumerate(min_shape):
                sliced_a = sliced_a.narrow(dim, 0, size)
                sliced_b = sliced_b.narrow(dim, 0, size)

        if masks is not None and key in masks:
            mask = masks[key].to(tensor_a.device)
            
            if mask.shape != tensor_a.shape:
                logger.warning(f"Skipping mask for {key}: shape mismatch {mask.shape} vs {tensor_a.shape}")
                merged_tensor = slerp_tensor(t, tensor_a, tensor_b)
            else:
                # Apply SLERP only where mask is True
                slerp_result = slerp_tensor(t, tensor_a, tensor_b)
                merged_tensor = torch.where(mask, slerp_result, tensor_a)
                coverage = mask.float().mean().item() * 100
                print(f"{key}: Applied SLERP to {coverage:.2f}% of parameters")
        else:
            merged_tensor = slerp_tensor(t, tensor_a, tensor_b)

        merged_state_dict[key] = merged_tensor

    print.info(f"Saving merged model to {output_path}")
    model = AutoModelForCausalLM.from_pretrained(model_a_path)
    tokenizer = AutoTokenizer.from_pretrained(model_a_path)
    if hasattr(model.config, '_name_or_path'):
        model.config._name_or_path = ""
    model.load_state_dict(merged_state_dict)
    model = model.to(torch.float16)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
