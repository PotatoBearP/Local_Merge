from huggingface_hub import HfApi
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