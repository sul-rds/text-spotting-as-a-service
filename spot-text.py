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

import geopandas as gpd
import shapely
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
    output_dir: Path = typer.Argument(
        None, help="Output folder (defaults to stdout)", show_default=False
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

    image_name = image_path.name
    if output_dir is not None:
        json_out_path = output_dir / Path(image_name).with_suffix(".json")
        csv_out_path = output_dir / Path(image_name).with_suffix(".csv")

    predictions_df = do_inference(image_path)

    if "crs" in predictions_df:
        # Geo coordinates are available; find rectified midpoints and centroids
        gdf_utm = predictions_df.estimate_utm_crs()
        map_crs = str(predictions_df["crs"].values[0])
   
        # Might as well call a (georeferenced) spade a spade
        if output_dir is not None:
            json_out_path = output_dir / Path(image_name).with_suffix(".geojson")

        def get_projected_geometry_middle(
            geom: shapely.LineString | shapely.Polygon,
        ):

            line_not_polygon = isinstance(geom, shapely.LineString)
            in_gdf = gpd.GeoDataFrame(geometry=[geom], crs=map_crs)
            # Project the lat/lon coordinates into the best flattened coordinate system for the map
            proj_gdf = in_gdf.to_crs(epsg=gdf_utm.to_authority()[1])
            # Compute the midpoint or centroid, dependingon what type of geometry we're looking at
            if line_not_polygon:
                middle_geom = proj_gdf["geometry"].interpolate(0.5, normalized=True)
            else:
                middle_geom = proj_gdf["geometry"].centroid
            out_gdf = gpd.GeoDataFrame(geometry=middle_geom.values)
            # Convert the computed middle point back to lat/lon
            out_gdf = out_gdf.to_crs(map_crs)
            return out_gdf["geometry"].values[0]

        predictions_df["line_midpoint"] = predictions_df["line"].apply(
            get_projected_geometry_middle
        )
        predictions_df["polygon_centroid"] = predictions_df["geometry"].apply(
            get_projected_geometry_middle
        )

        predictions_df = predictions_df.rename(columns={"geometry": "polygon"})

        predictions_df["all_geoms"] = [
            shapely.GeometryCollection(
                [
                    shapely.Polygon(py),
                    shapely.Point(pc),
                    shapely.LineString(ln),
                    shapely.Point(pt),
                ]
            )
            for py, pc, ln, pt in zip(
                predictions_df["polygon"],
                predictions_df["polygon_centroid"],
                predictions_df["line"],
                predictions_df["line_midpoint"],
            )
        ]
        predictions_df = predictions_df.set_geometry(
            gpd.GeoSeries(predictions_df["all_geoms"], crs=map_crs)
        )
        predictions_df = predictions_df.drop(
            columns=["all_geoms", "line", "polygon", "pixel_geometry", "pixel_line"]
        )

        # Separate lat/lon of midpoints/centroids for CSV
        predictions_df["line_midpoint_lon"] = predictions_df["line_midpoint"].apply(
            lambda point: point.x
        )
        predictions_df["line_midpoint_lat"] = predictions_df["line_midpoint"].apply(
            lambda point: point.y
        )
        predictions_df["polygon_centroid_lon"] = predictions_df[
            "polygon_centroid"
        ].apply(lambda point: point.x)
        predictions_df["polygon_centroid_lat"] = predictions_df[
            "polygon_centroid"
        ].apply(lambda point: point.y)

        if output_dir is None:
            # If stdout, it's part of a pipeline, and only JSON is needed
            predictions_df.T.to_json(sys.stdout, default_handler=str, indent=2)
        else:
            logging.info(f"[yellow]Saving JSON and CSV to {output_dir}...")
            output_dir.mkdir(parents=True, exist_ok=True)
            # Output GeoJSON first
            # Ensure that these values remain in the GeoJSON output as non-geometries
            predictions_df["line_midpoint"] = gpd.GeoSeries(
                predictions_df["line_midpoint"]
            ).to_wkt()
            predictions_df["polygon_centroid"] = gpd.GeoSeries(
                predictions_df["polygon_centroid"]
            ).to_wkt()
            predictions_df.to_file(str(json_out_path), driver="GeoJSON")
            # Provide a CSV with only the info people would want
            predictions_df = predictions_df.drop(
                columns=["geometry", "line_midpoint", "polygon_centroid"]
            )
            predictions_df.to_csv(csv_out_path)

    else:
        # Only pixel coordinates are available
        predictions_df["pixel_line_midpoint"] = predictions_df["pixel_line"].apply(
            lambda l: l.interpolate(0.5, normalized=True)
        )
        predictions_df["pixel_polygon_centroid"] = predictions_df[
            "pixel_geometry"
        ].apply(lambda g: g.centroid)

        if output_dir is None:
            predictions_df.T.to_json(sys.stdout, default_handler=str, indent=2)
        else:
            logging.info(f"[yellow]Saving JSON and CSV to {output_dir}...")
            output_dir.mkdir(parents=True, exist_ok=True)
            predictions_df.T.to_json(json_out_path, default_handler=str, indent=2)
            predictions_df.to_csv(csv_out_path)

    logging.info("[yellow]...done!")


if __name__ == "__main__":
    cli()
