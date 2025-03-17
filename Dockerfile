ARG PYTORCH="2.3.1"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

WORKDIR /workspace

# fetch the key refer to https://forums.developer.nvidia.com/t/20-04-cuda-docker-image-is-broken/212892/9
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Pytorch Lighting
RUN pip install pytorch-lightning

#JDK
RUN apt update
RUN apt install -y default-jdk

# Other packets
RUN pip install opencv-python && \
    pip install pandas && \
    pip install albumentations && \
    pip install scikit-learn

# TORCHSEVER
RUN pip install torchserve torch-model-archiver nvgpu
