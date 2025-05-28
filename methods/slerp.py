import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_weights(model_path: str) -> dict:
    """
    Loads a model from a given path and returns its state dict.
    Assumes model is compatible with AutoModelForCausalLM.

    Args:
        model_path (str): Hugging Face-style path or local directory.

    Returns:
        dict: State dictionary of the model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model.state_dict()

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

def slerp_models(model_a_path: str, model_b_path: str, output_path: str, t: float = 0.5):
    """
    Merges two models using SLERP and saves the result.

    Args:
        model_a_path (str): Path to the base model.
        model_b_path (str): Path to the target model.
        output_path (str): Path to save the merged model.
        t (float): Interpolation parameter (0 = all A, 1 = all B).
    """
    print(f"Loading weights from model A: {model_a_path}")
    state_dict_a = load_model_weights(model_a_path)

    print(f"Loading weights from model B: {model_b_path}")
    state_dict_b = load_model_weights(model_b_path)

    merged_state_dict = {}

    print("Performing SLERP on model weights...")
    for key in state_dict_a.keys():
        if key not in state_dict_b:
            print(f"⚠️ Skipping {key}: not found in model B.")
            continue

        tensor_a = state_dict_a[key]
        tensor_b = state_dict_b[key]

        if tensor_a.shape != tensor_b.shape:
            print(f"⚠️ Skipping {key}: shape mismatch {tensor_a.shape} vs {tensor_b.shape}")
            continue

        merged_tensor = slerp_tensor(t, tensor_a, tensor_b)
        merged_state_dict[key] = merged_tensor

    print(f"Saving merged model to {output_path}")
    model = AutoModelForCausalLM.from_pretrained(model_a_path)
    model.load_state_dict(merged_state_dict)
    model.save_pretrained(output_path)
