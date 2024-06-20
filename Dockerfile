# Use a base image from NVIDIA that includes CUDA and Ubuntu 22.04
FROM nvcr.io/nvidia/cuda:11.8.0-base-ubuntu22.04

# Define environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Installing Anaconda
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Installing  miipher requirements
RUN apt-get update; apt-get install -y ffmpeg espeak-ng git build-essential
RUN /root/miniconda3/bin/conda install python=3.10.11 -y
RUN /root/miniconda3/bin/conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
#RUN pip install scipy==1.11 nltk==3.8.1 singleton_decorator==1.0.0 phonemizer==3.2.1
#RUN pip install lightning~=2.0.5  speechbrain~=0.5.14 matplotlib~=3.7.2 pyroomacoustics~=0.7.3 hydra-core~=1.3.2 webdataset~=0.2.48 text2phonemesequence~=0.1.4 mecab-python3~=1.0.6 unidic~=1.1.0 wandb~=0.15.7 lightning_vocoders@git+https://github.com/Wataru-Nakata/ssl-vocoders llvmlite~=0.40.1 gradio~=3.45.2

# Set up the working directory
WORKDIR /app

# Copy the source code into the container
COPY . /app

# Install the application dependencies using pip
RUN pip install -e .