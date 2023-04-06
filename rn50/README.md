# collect RN50
### install anaconda
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh
```
### create conda env
```
conda create --name rn50 python=3.9
conda activate rn50
conda install intel-openmp jemalloc==5.2.1
```
### install Pytorch
```
pip install https://download.pytorch.org/whl/nightly/cpu-cxx11-abi/torch-2.1.0.dev20230313%2Bcpu.cxx11.abi-cp39-cp39-linux_x86_64.whl
```
### install IPEX
```
git clone https://github.com/zhuhaozhe/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout rn50
git submodule sync
git submodule update --init --recursive
python setup.py install
```

### Manually setting DIR
replace "/mnt/DP_disk2/resnet50/scripts/emon-config.txt" in https://github.com/zhuhaozhe/emon-collection/blob/main/rn50/scripts/resnet-conv.py#L41

replace "EMON_HOME" in https://github.com/zhuhaozhe/emon-collection/blob/main/rn50/scripts/rn50.sh#L6

replace "/mnt/DP_disk2/resnet50/results/" in https://github.com/zhuhaozhe/emon-collection/blob/main/rn50/scripts/rn50.sh#L14 and https://github.com/zhuhaozhe/emon-collection/blob/main/rn50/scripts/rn50.sh#L20
