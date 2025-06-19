import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from methods.utility import load_model_weights
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def replace_masked_area(model_a_path: str,
                        model_b_path: str,
                        mask_path: str,
                        output_path: str):
    """
    Replace the masked area parameters of model_a with model_b based on mask values
    
    Args:
        model_a_path (str): Path to the base model.
        model_b_path (str): Path to the target model.
        mask_path: Path to mask file containing values between 0 and 1
        output_path (str): Path to save the merged model.
        
    Returns:
        model_a: Updated model with replaced parameters
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

    replaced_state_dict = {}

    logger.info("Performing model weights replacements...")
    for key in state_dict_a.keys():
        if key not in state_dict_b:
            logger.warning(f"Skipping {key}: not found in model B")
            replaced_state_dict[key] = tensor_a
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
                merged_tensor = tensor_a  # Keep original tensor if mask shape doesn't match
            else:
                # Apply replace only where mask is True
                merged_tensor = torch.where(mask, tensor_b, tensor_a)
                coverage = mask.float().mean().item() * 100
                logger.info(f"{key}: Applied replace to {coverage:.2f}% of parameters")
        else:
            merged_tensor = tensor_a # this should not happen as mask is compulsory, but just in case
        replaced_state_dict[key] = merged_tensor

    logger.info(f"Saving merged model to {output_path}")
    model = AutoModelForCausalLM.from_pretrained(model_a_path)
    tokenizer = AutoTokenizer.from_pretrained(model_a_path)
    if hasattr(model.config, '_name_or_path'):
        model.config._name_or_path = ""
    model.load_state_dict(replaced_state_dict)
    model = model.to(torch.float16)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
