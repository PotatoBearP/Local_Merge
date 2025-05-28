from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    # Create repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True
        )
    except Exception as e:
        raise Exception(f"Failed to create repository: {str(e)}")

    print(f"Uploading model to {repo_name}...")
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Upload model and tokenizer
    model.push_to_hub(
        repo_id=repo_name,
        commit_message=commit_message or "Upload merged model",
        token=token
    )
    
    tokenizer.push_to_hub(
        repo_id=repo_name,
        commit_message=commit_message or "Upload tokenizer",
        token=token
    )

    print(f"✅ Model successfully uploaded to: https://huggingface.co/{repo_name}")
    return f"https://huggingface.co/{repo_name}"