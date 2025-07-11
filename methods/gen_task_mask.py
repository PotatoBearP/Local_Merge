import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from methods.utility import load_model_weights, minimum_tensor_slices
from typing import Dict, Optional

def generate_significance_mask(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    top_percentile: float = 0.1,
    min_threshold: float = 1e-5,
    save_mask: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Generates mask vectors highlighting the most significant parameter differences between two models.

    Args:
        model_a_path (str): Path to the first model
        model_b_path (str): Path to the second model
        output_path (str): Path to save the resulting task mask
        top_percentile (float): Percentile of top differences to keep (0.1 = top 10%)
        min_threshold (float): Minimum absolute difference threshold
        save_mask (Optional[str]): Path to save task mask if specified

    Returns:
        Dict[str, torch.Tensor]: Dictionary of boolean masks for significant parameters
    """
    print(f"Loading weights from model A: {model_a_path}")
    state_dict_a = load_model_weights(model_a_path)

    print(f"Loading weights from model B: {model_b_path}")
    state_dict_b = load_model_weights(model_b_path)

    task_masks = {}
    total_params = 0
    all_deltas = []
    adjusted_tensors = {}  # Store adjusted tensors

    print("Calculating parameter differences...")
    for key in state_dict_a.keys():
        if key not in state_dict_b:
            print(f"Skipping {key}: not found in model B")
            continue

        tensor_a = state_dict_a[key]
        tensor_b = state_dict_b[key]

        if tensor_a.shape != tensor_b.shape:
            tensor_a, tensor_b= minimum_tensor_slices(tensor_a, tensor_b)
            print(f"Shape mismatch for {key}: {tensor_a.shape} vs {tensor_b.shape}")
            # Store adjusted tensors for later use
            adjusted_tensors[key] = (tensor_a, tensor_b)

        delta = torch.abs(tensor_b - tensor_a)
        all_deltas.append(delta.flatten())
        total_params += delta.numel()

    # Calculate threshold based on percentile
    if all_deltas:
        combined_deltas = torch.cat(all_deltas)
        sorted_deltas, _ = torch.sort(combined_deltas)
        k = int(len(sorted_deltas) * (1 - top_percentile))
        percentile_threshold = sorted_deltas[k]
        final_threshold = max(percentile_threshold.item(), min_threshold)
        
        print(f"Calculated significance threshold: {final_threshold:.2e}")

        for key in state_dict_a.keys():
            if key not in state_dict_b:
                continue

            # Use adjusted tensors if available
            if key in adjusted_tensors:
                tensor_a, tensor_b = adjusted_tensors[key]
            else:
                tensor_a = state_dict_a[key]
                tensor_b = state_dict_b[key]

            delta = torch.abs(tensor_b - tensor_a)
            mask = delta >= final_threshold
            
            if torch.any(mask):
                task_masks[key] = mask
                coverage = mask.sum().item() / mask.numel() * 100
                print(f"{key}: {coverage:.2f}% parameters marked as significant")

    print(f"Generated masks for {len(task_masks)} layers")

    print(f"Saving task mask to {output_path}")
    torch.save(task_masks, output_path)

    return task_masks