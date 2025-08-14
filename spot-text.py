#!/usr/bin/env python3

"""
Run the MapReader text-spotting functionality against an image

Example:
    spot-text.py /path/to/image.jpg [/path/to/output.json]
"""

__version__ = "0.1"

import logging
import sys
from pathlib import Path
from typing import cast

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler

from src.utils import capture_stdout

cli = typer.Typer(add_completion=False, no_args_is_help=True)


@cli.callback(invoke_without_command=True)
def spot(
    ctx: typer.Context,
    image_path: str = typer.Argument(..., help="Path to the image", show_default=False),
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

    if not Path(image_path).exists():
        logging.fatal("[red]Image does not exist")
        print(ctx.get_help())
        raise typer.Exit(code=1)

    logging.info("[yellow]Loading Dependencies...")
    with capture_stdout() as _get_value:
        from mapreader import MapTextRunner, loader

    logging.info("[yellow]Loading Image...")
    input_files = loader(image_path)

    logging.info("[yellow]Creating Patches...")
    with capture_stdout() as _get_value:
        # note: the overlap argument appears to be incorrectly typed
        input_files.patchify_all(patch_size=1024, overlap=0.2)  # type: ignore
    parent_df, patch_df = input_files.convert_images()

    logging.info("[yellow]Spotting Text...")
    with capture_stdout() as _get_value:
        runner = MapTextRunner(
            patch_df,
            parent_df,
            cfg_file="rumsey-finetune/final_rumsey.yaml",
            weights_file="rumsey-finetune/rumsey-finetune.pth",
            device="cuda",
        )
    runner.run_all(return_dataframe=True)

    logging.info("[yellow]Converting Output...")
    predictions_df: pd.DataFrame = cast(
        pd.DataFrame,
        runner.convert_to_parent_pixel_bounds(
            return_dataframe=True, deduplicate=True, min_ioa=0.7
        ),
    )

    if output_path is None:
        predictions_df.T.to_json(sys.stdout, default_handler=str, indent=2)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as _fh:
            predictions_df.T.to_json(_fh, default_handler=str, indent=2)


if __name__ == "__main__":
    cli()
