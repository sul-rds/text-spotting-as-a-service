# Text-Spotting-as-a-Service

This repo is intended to support deployment of the Text-Spotting code from the [MapReader](https://mapreader.readthedocs.io/) project in a simple and convenient manner.

The deployment includes the following packages from the [Maps as Data](https://github.com/maps-as-data) project:

* [mapreader](https://github.com/maps-as-data/mapreader)
* [deepsolo](https://github.com/maps-as-data/deepsolo)
* [dptext-detr](https://github.com/maps-as-data/dptext-detr)
* [maptextpipeline](https://github.com/maps-as-data/maptextpipeline)

Please see those repositories for more information.

At present there is a `uv`-based configuration and a `Dockerfile` which have been tested and confirmed to work for CUDA-enabled environments.  There may be a simple web service added at some point, such that images can be `POST`ed to an API endpoint and JSON returned.


## Installation

1. Ensure the appropriate NVIDIA drivers for your system are installed.
2. Clone the repository.


### `uv`

3. Ensure both `uv` and the NVIDIA CUDA Toolkit is installed (the config here expects CUDA >= 12)
4. Run inference:
   `uv run python spot-text.py /opt/maps/map.tiff`


### Docker

3. Ensure both docker and  the NVIDIA Container Toolkit are installed.

4. Build the image:

   `docker build -t text-spotting-as-a-service .`

5. Run inference:
   `docker run --rm --gpus all -v /path/to/maps:/opt/maps text-spotting-as-a-service:latest uv run python spot-text.py /opt/maps/map.tiff`


## Usage

```sh
❯ uv run python spot-text.py --help

 Usage: spot-text.py [OPTIONS] IMAGE_PATH [OUTPUT_PATH] COMMAND [ARGS]...

╭─ Arguments ───────────────────────────────────────────────────────────────────────────────╮
│ *    image_path       PATH           Path to the image [required]                         │
│      output_path      [OUTPUT_PATH]  Output path (defaults to stdout)                     │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────╮
│ --verbose  -v                                                                             │
│ --quiet    -q                                                                             │
│ --version                                                                                 │
│ --help               Show this message and exit.                                          │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```
