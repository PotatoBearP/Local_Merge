import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utility import load_model_weights
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def task_arithmetic_tensor(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    Performs task arithmetic merge between two tensors.
    Merged = v0 + t * (v1 - v0)

    Args:
        t (float): Merge factor.
        v0 (torch.Tensor): First tensor (base).
        v1 (torch.Tensor): Second tensor (target).

    Returns:
        torch.Tensor: Merged tensor.
    """
    return (v0 + t * (v1 - v0)).to(v0.dtype)


def task_arithmetic_models(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    t: float = 0.5,
    mask_path: str = None
):
    """
    Merges two models using Task Arithmetic and saves the result.

    Args:
        model_a_path (str): Path to the base model.
        model_b_path (str): Path to the target model.
        output_path (str): Path to save the merged model.
        t (float): Interpolation parameter.
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

    logger.info("Performing Task Arithmetic on model weights...")
    for key in state_dict_a.keys():
        if key not in state_dict_b:
            logger.warning(f"Skipping {key}: not found in model B")
            merged_state_dict[key] = state_dict_a[key]
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
            tensor_a, tensor_b = sliced_a, sliced_b

        if masks is not None and key in masks:
            mask = masks[key].to(tensor_a.device)

            if mask.shape != tensor_a.shape:
                logger.warning(f"Skipping mask for {key}: shape mismatch {mask.shape} vs {tensor_a.shape}")
                merged_tensor = task_arithmetic_tensor(t, tensor_a, tensor_b)
            else:
                task_arith_result = task_arithmetic_tensor(t, tensor_a, tensor_b)
                merged_tensor = torch.where(mask, task_arith_result, tensor_a)
                coverage = mask.float().mean().item() * 100
                print(f"{key}: Applied Task Arithmetic to {coverage:.2f}% of parameters")
        else:
            merged_tensor = task_arithmetic_tensor(t, tensor_a, tensor_b)

        merged_state_dict[key] = merged_tensor

    print(f"Saving merged model to {output_path}")
    model = AutoModelForCausalLM.from_pretrained(model_a_path)
    tokenizer = AutoTokenizer.from_pretrained(model_a_path)
    if hasattr(model.config, '_name_or_path'):
        model.config._name_or_path = ""
    model.load_state_dict(merged_state_dict)
    model = model.to(torch.float16)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)