# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

ENV PYTHON_VERSION=3.8
ENV DEBIAN_FRONTEND="noninteractive"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Create workspace folder and use as working directory for all subsequent commands.
RUN mkdir -p /workspace/main
WORKDIR /workspace/main


# Install dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    sudo \
    pip \
    nano \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6

# Install dependencies for Python 3.8 virtual environment.
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes -y
RUN apt-get install python3.8-venv python3-pip python3.8-dev -y


# Create a virtual environment and append Python runtime to PATH.
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install packages in virtual environment.
# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt
# RUN rm -rf requirements.txt

CMD ["/bin/bash"]
