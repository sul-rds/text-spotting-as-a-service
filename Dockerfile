FROM python:3.12-slim-bookworm
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

LABEL author="Simon Wiles <simon.wiles@stanford.edu>"
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/text-spotting-as-a-service
COPY . /opt/texts-potting-as-a-service

RUN apt-get update \                                                        
  && apt-get upgrade -y \                                                    
  && apt-get install -y --no-install-recommends git-core g++ libgl1 \
  && rm -rf /var/lib/apt/lists/*                     

RUN uv sync
