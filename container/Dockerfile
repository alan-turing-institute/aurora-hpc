# Use the NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set shell
SHELL ["/bin/bash", "-c"]

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working dir
WORKDIR /

# Install system-level deps
RUN apt update && apt-get install -y build-essential cmake wget git python3 python3-pip

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Set working dir
WORKDIR /

# clone aurora-hpc & install dependencies
RUN git clone --branch podman-hpc https://github.com/alan-turing-institute/aurora-hpc.git
WORKDIR /aurora-hpc
RUN pip3 install --no-cache-dir ".[bask,dev]"

# Define the default command to run when starting the container
CMD ["/bin/bash"]
