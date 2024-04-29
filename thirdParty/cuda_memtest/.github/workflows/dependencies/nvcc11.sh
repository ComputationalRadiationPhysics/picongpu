#!/usr/bin/env bash
#
# Authors: Axel Huebl

set -eu -o pipefail

sudo apt-get -qqq update
sudo apt-get install -y \
    build-essential     \
    ca-certificates     \
    cmake               \
    gnupg               \
    pkg-config          \
    wget

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

sudo apt-get update
sudo apt-get install -y          \
    cuda-command-line-tools-11-7 \
    cuda-compiler-11-7           \
    cuda-cupti-dev-11-7          \
    cuda-minimal-build-11-7      \
    cuda-nvml-dev-11-7
sudo ln -s cuda-11.0 /usr/local/cuda

