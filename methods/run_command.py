import click
from methods.slerp import slerp_models
from methods.crop_splice import crop_model_deltas, splice_model_deltas
from methods.utility import upload_to_hub
from methods.crop_splice import generate_significance_mask

@click.command("generate_mask")
@click.argument("model_a_path", type=str)
@click.argument("model_b_path", type=str)
@click.argument("output_path", type=str)
@click.option(
    "--top-percentile", "-p",
    type=float,
    default=0.1,
    help="Percentile of top differences to keep (0.1 = top 10%)"
)
@click.option(
    "--min-threshold", "-t",
    type=float,
    default=1e-5,
    help="Minimum absolute difference threshold"
)
def generate_mask(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    top_percentile: float,
    min_threshold: float,
):
    """
    Generates significance masks based on parameter differences between models.
    
    Args:
        model_a_path: Path to the first model
        model_b_path: Path to the second model
        output_path: Path to save the generated masks
        top_percentile: Percentile of top differences to keep
        min_threshold: Minimum absolute difference threshold
    """
    try:
        if not 0 < top_percentile <= 1:
            raise click.BadParameter("Top percentile must be between 0 and 1")

        generate_significance_mask(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            output_path=output_path,
            top_percentile=top_percentile,
            min_threshold=min_threshold
        )
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

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
    "--mask-path", "-m",
    type=str,
    default=None,
    help="Path to significance mask file for guided merging"
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
def slerp(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    interpolation: float,
    mask_path: str,
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
            t=interpolation,
            mask_path=mask_path
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
    
@click.command("crop")
@click.argument("model_a_path", type=str)
@click.argument("model_b_path", type=str)
@click.argument("output_path", type=str)
@click.option(
    "--threshold", "-t",
    type=float,
    default=1e-5,
    help="Minimum difference threshold to keep"
)
def crop(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    threshold: float,
):
    """
    Crops delta weights between two models and saves them.
    
    Args:
        model_a_path: Path to the base model
        model_b_path: Path to the target model
        output_path: Path to save delta weights
        threshold: Minimum difference threshold
    """
    try:
        crop_model_deltas(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            output_path=output_path,
            threshold=threshold
        )

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

@click.command("splice")
@click.argument("base_model_path", type=str)
@click.argument("delta_path", type=str)
@click.argument("output_path", type=str)
@click.option(
    "--upload", "-u",
    is_flag=True,
    help="Upload model to Hugging Face Hub after splicing"
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
def splice(
    base_model_path: str,
    delta_path: str,
    output_path: str,
    upload: bool,
    repo_name: str,
    private: bool,
    commit_message: str,
):
    """
    Applies delta weights to a base model.
    
    Args:
        base_model_path: Path to the base model
        delta_path: Path to the delta weights file
        output_path: Path to save the resulting model
    """
    try:
        splice_model_deltas(
            base_model_path=base_model_path,
            delta_path=delta_path,
            output_path=output_path
        )
        click.echo(f"Successfully spliced model saved to: {output_path}")

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

def main():
    pass

if __name__ == '__main__':
    main()