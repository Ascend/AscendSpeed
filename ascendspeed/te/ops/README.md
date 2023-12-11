# How to run the te ops?

## previous installation
+ CANN
+ Transformer-Boost 
https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/260809541?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373
+ torch_npu

## compile and install
### 1. set the environment variables

+ export ASCEND_TOOLKIT_HOME = /usr/local/Ascend/latest/

### 2. include head files

+ newest torch_npu
+ newest cann

### 3. install scripts
```shell
python3 setup.py build
python3 setup.py bdist_wheel
pip3 install dist/*.whl --force-reinstall
```
