import logging
from pathlib import Path
from typing import cast

import pandas as pd
import torch

from .utils import capture_stdout

logger = logging.getLogger(__name__)


def merge_detectron2_split_weights(split_files):
    return {
        k: v
        for file_path in split_files
        for k, v in torch.load(file_path, map_location="cpu").items()
    }


def do_inference(image_path: Path) -> pd.DataFrame:
    logger.info("[yellow]Loading Dependencies...")
    with capture_stdout() as _get_value:
        from mapreader import MapTextRunner, loader

    logger.info("[yellow]Loading Weights...")
    if not Path("model/merged_weights.pth").exists():
        merged_weights = merge_detectron2_split_weights(
            [
                "model/backbone_weights.pth",
                "model/other_weights.pth",
            ]
        )
        torch.save(merged_weights, "model/merged_weights.pth")

    logger.info("[yellow]Loading Image...")
    input_files = loader(str(image_path))

    logger.info("[yellow]Creating Patches...")
    patch_cache_path = Path("patch_cache") / image_path.name
    patch_cache_path.mkdir(parents=True, exist_ok=True)
    with capture_stdout() as _get_value:
        # note: the overlap argument appears to be incorrectly typed
        input_files.patchify_all(
            patch_size=1024,
            overlap=0.2,  # type: ignore
            path_save=str(patch_cache_path),
        )
    parent_df, patch_df = input_files.convert_images()

    logger.info("[yellow]Spotting Text...")
    with capture_stdout() as _get_value:
        runner = MapTextRunner(
            patch_df,
            parent_df,
            cfg_file="model/detectron2_config.yaml",
            weights_file="model/merged_weights.pth",
            device="cuda",
        )
    runner.run_all(return_dataframe=True)

    logger.info("[yellow]Converting Output...")
    predictions_df: pd.DataFrame = cast(
        pd.DataFrame,
        runner.convert_to_parent_pixel_bounds(
            return_dataframe=True, deduplicate=True, min_ioa=0.7
        ),
    )

    return predictions_df
