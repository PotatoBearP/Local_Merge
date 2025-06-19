import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from methods.utility import load_model_weights
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def crop_model_deltas(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    threshold: float = 1e-6,
    norm: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Crops and optionally saves the delta parameters between two models.

    Args:
        model_a_path (str): Path to the base model
        model_b_path (str): Path to the target model
        output_path (str): Path to save the resulting cropped vector or norm dict
        threshold (float): Minimum difference threshold to keep
        norm (bool): If True, save only module-wise norm of cropped deltas

    Returns:
        Dict[str, torch.Tensor] or Dict[str, float]: Cropped deltas or module norms
    """
    print(f"Loading weights from model A: {model_a_path}")
    state_dict_a = load_model_weights(model_a_path)

    print(f"Loading weights from model B: {model_b_path}")
    state_dict_b = load_model_weights(model_b_path)

    result = {}
    print("Calculating and cropping deltas...")
    for key in state_dict_a.keys():
        if key not in state_dict_b:
            print(f"Skipping {key}: not found in model B")
            continue

        tensor_a = state_dict_a[key]
        tensor_b = state_dict_b[key]

        if tensor_a.shape != tensor_b.shape:
            print(f"Skipping {key}: shape mismatch {tensor_a.shape} vs {tensor_b.shape}")
            continue

        delta = tensor_b - tensor_a
        mask = torch.abs(delta) >= threshold
        cropped_delta = delta * mask

        if torch.any(mask):
            if norm:
                norm_value = torch.norm(cropped_delta.float()).item()
                if norm_value > 0:
                    result[key] = norm_value
            else:
                if torch.any(cropped_delta):
                    result[key] = cropped_delta

    if norm:
        print(f"Saving norm dict to {output_path}")
    else:
        print(f"Saving delta weights to {output_path}")
    torch.save(result, output_path)

    return result

def splice_model_deltas(
    base_model_path: str,
    delta_path: str,
    output_path: str
) -> None:
    """
    Applies delta weights to a base model and saves the result.

    Args:
        base_model_path (str): Path to the base model to modify
        delta_path (str): Path to the saved delta weights
        output_path (str): Path to save the resulting spliced model
    """
    print(f"Loading base model from: {base_model_path}")
    state_dict_base = load_model_weights(base_model_path)

    print(f"Loading delta weights from: {delta_path}")
    deltas = torch.load(delta_path)

    # Apply deltas to base model
    print("Applying delta weights...")
    modified_state_dict = state_dict_base.copy()
    for key, delta in deltas.items():
        if key not in state_dict_base:
            print(f"Skipping {key}: not found in base model")
            continue
            
        if state_dict_base[key].shape != delta.shape:
            print(f"Skipping {key}: shape mismatch {state_dict_base[key].shape} vs {delta.shape}")
            continue
            
        modified_state_dict[key] = state_dict_base[key] + delta

    # Save modified model
    print(f"Saving spliced model to: {output_path}")
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    if hasattr(model.config, '_name_or_path'):
        model.config._name_or_path = ""
    
    model.load_state_dict(modified_state_dict)
    model = model.to(torch.bfloat16)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)