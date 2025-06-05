from huggingface_hub import HfApi
import torch
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

def minimum_tensor_slices(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create and return minimum tensors for two input tensors.
    
    Args:
        tensor_a (torch.Tensor): First tensor
        tensor_b (torch.Tensor): Second tensor
        
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict]: (sliced tensor_a, sliced tensor_b, shape info)
    """
    min_shape = tuple(min(a, b) for a, b in zip(tensor_a.shape, tensor_b.shape))
    slice_indices = tuple(slice(0, dim) for dim in min_shape)
    
    # Slice tensors to minimum shape
    min_tensor_a = tensor_a[slice_indices]
    min_tensor_b = tensor_b[slice_indices]
    
    return min_tensor_a, min_tensor_b

def upload_to_hub(
    model_path: str,
    repo_name: str,
    commit_message: str = None,
    private: bool = False
) -> str:
    """
    Uploads a merged model to Hugging Face Hub.

    Args:
        model_path (str): Local path to the merged model
        repo_name (str): Name for the model on HF Hub (format: 'username/model-name')
        commit_message (str, optional): Custom commit message
        private (bool): Whether to create a private repository

    Returns:
        str: URL of the uploaded model on HF Hub
    """
    api = HfApi()

    # Create repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True
        )
    except Exception as e:
        raise Exception(f"Failed to create repository: {str(e)}")

    print(f"Uploading model folder to {repo_name}...")
    
    # Upload the entire folder
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        commit_message=commit_message or "Upload model files",
    )

    print(f"✅ Model successfully uploaded to: https://huggingface.co/{repo_name}")
    return f"https://huggingface.co/{repo_name}"