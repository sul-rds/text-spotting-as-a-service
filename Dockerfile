FROM python:3.12-slim-bookworm
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

LABEL author="Simon Wiles <simon.wiles@stanford.edu>"
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/textspotting-as-a-service
COPY . /opt/textspotting-as-a-service

RUN apt-get update \                                                        
  && apt-get upgrade -y \                                                    
  && apt-get install -y --no-install-recommends git-core g++ \
  && rm -rf /var/lib/apt/lists/*                     

RUN uv venv && uv pip install setuptools torch && uv sync --no-build-isolation
