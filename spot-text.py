#!/usr/bin/env -S uv run --script

"""
Run the MapReader text-spotting functionality against an image

Example:
    spot-text.py /path/to/image.jpg [/path/to/output.json]
"""

__version__ = "0.1"

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from src.inference import do_inference

cli = typer.Typer(add_completion=False, no_args_is_help=True)


@cli.callback(invoke_without_command=True)
def spot_cli(
    ctx: typer.Context,
    image_path: Path = typer.Argument(
        ..., help="Path to the image", show_default=False
    ),
    output_path: Path = typer.Argument(
        None, help="Output path (defaults to stdout)", show_default=False
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    version: bool = typer.Option(False, "--version"),
):
    if version:
        print(__version__)
        raise SystemExit

    log_level = logging.DEBUG if verbose else logging.INFO
    log_level = logging.CRITICAL if quiet else log_level
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(markup=True, console=Console(width=180, stderr=True))],
    )

    logging.getLogger("detectron2").setLevel(logging.WARNING)
    logging.getLogger("fvcore").setLevel(logging.WARNING)

    if not image_path.exists():
        logging.fatal("[red]Image does not exist")
        print(ctx.get_help())
        raise typer.Exit(code=1)

    predictions_df = do_inference(image_path)

    if output_path is None:
        predictions_df.T.to_json(sys.stdout, default_handler=str, indent=2)
    else:
        logging.info(f"[yellow]Saving output to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as _fh:
            predictions_df.T.to_json(_fh, default_handler=str, indent=2)

    logging.info("[yellow]...done!")


if __name__ == "__main__":
    cli()
