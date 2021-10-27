#!/bin/bash

# Install Dependencies to Run DDSP

sudo apt-get update
sudo apt-get install libblas-dev liblapack-dev liblapack-doc

sudo -H pip3 install -U pip testresources setuptools==49.6.0 protobuf
sudo -H pip3 install -U numpy==1.19.4 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 # h5py takes a while
sudo -H pip3 install -U keras_applications==1.0.8 gast==0.2.2 futures pybind11

sudo -H pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow
# this step takes a while

sudo -H npm install -g @bazel/bazelisk
# needed for `dm-tree` in DDSP

sudo -H pip3 install -U ddsp colorama

cd ~/Desktop ; touch launchJupyter.sh
echo "# The following line must be executed prior to launching Jupyter" | sudo tee -a launchJupyter.sh
echo | sudo tee -a launchJupyter.sh
echo "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1" | sudo tee -a launchJupyter.sh
echo | sudo tee -a launchJupyter.sh
echo "jupyter notebook" | sudo tee -a launchJupyter.sh
sudo chmod +x launchJupyter.sh

echo '--Installing DDSP for PyTorch--' ; echo ; echo ; sleep 2
cd ~/Desktop/raivShell/ddspShell/ddsp_pytorch-master/realtime ; mkdir build ; cd build
export Torch_DIR=~/opt/anaconda3/lib/python3.8/site-packages/torch/share/cmake/Torch
cmake ../ -DCMAKE_PREFIX_PATH=~/opt/anaconda3/lib/python3.8/site-packages/torch -DCMAKE_BUILD_TYPE=Release
make install