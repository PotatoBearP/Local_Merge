import click
from methods.slerp import slerp_models
from methods.utility import validate_models
from utils.upload import upload_to_hub

@click.command("slerp_merge")
@click.argument("model_a_path", type=str)
@click.argument("model_b_path", type=str)
@click.argument("output_path", type=str)
@click.option(
    "--interpolation", "-t",
    type=float,
    default=0.5,
    help="Interpolation factor (0 = model A, 1 = model B)"
)
@click.option(
    "--upload", "-u",
    is_flag=True,
    help="Upload model to Hugging Face Hub after merging"
)
@click.option(
    "--repo-name",
    type=str,
    help="Repository name for HF Hub (format: 'username/model-name')"
)
@click.option(
    "--private",
    is_flag=True,
    help="Make the uploaded model repository private"
)
@click.option(
    "--commit-message",
    type=str,
    help="Custom commit message for the upload"
)
def main(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    interpolation: float,
    upload: bool,
    repo_name: str,
    private: bool,
    commit_message: str,
):
    """
    Merges two models using SLERP and optionally uploads to Hugging Face Hub.
    
    Args:
        model_a_path: Path to the first model
        model_b_path: Path to the second model
        output_path: Path where to save the merged model
        interpolation: Interpolation factor between 0.0 and 1.0
        upload: Whether to upload to HF Hub
        repo_name: Repository name on HF Hub
        private: Whether to make the repo private
        commit_message: Custom commit message
    """
    try:
        
        # Validate interpolation factor
        if not 0 <= interpolation <= 1:
            raise click.BadParameter("Interpolation factor must be between 0 and 1")

        # Perform SLERP merge
        slerp_models(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            output_path=output_path,
            t=interpolation
        )
        
        click.echo(f"Successfully merged models with interpolation factor {interpolation}")
        click.echo(f"Merged model saved to: {output_path}")

        # Handle upload if requested
        if upload:
            if not repo_name:
                raise click.BadParameter("--repo-name is required when using --upload")
            
            hub_url = upload_to_hub(
                model_path=output_path,
                repo_name=repo_name,
                private=private,
                commit_message=commit_message
            )
            click.echo(f"Model uploaded successfully to: {hub_url}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    main()