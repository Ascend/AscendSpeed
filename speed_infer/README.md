# ascend-speed-inference

#### 加速库安装

获取`Ascend-cann-atb_*_cxx11abi*_linux-{arch}.run`
```
chmox +x Ascend-cann-atb_*_cxx*abi*_linux-{arch}.run
./Ascend-cann-atb_*_cxx*abi*_linux-{arch}.run --install --install-path=YOUR_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source YOUR_PATH/atb/set_env.sh
```

#### 模型仓编译

##### 代码仓下载

```
git clone https://gitee.com/ascend/AscendSpeed.git
```

##### 代码编译

```
cd ascend-inference
bash scripts/build.sh
cd output/atb/
source set_env.sh
```

#### 环境变量参考

##### 日志打印

加速库日志

```
ATB_LOG_TO_FILE=1
ATB_LOG_TO_STDOUT=1
ATB_LOG_LEVEL=INFO
TASK_QUEUE_ENABLE=1
ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1
```

算子库日志

```
ASDOPS_LOG_TO_FILE=1
ASDOPS_LOG_TO_STDOUT=1
ASDOPS_LOG_LEVEL=INFO
```

性能提升（beta）
```
ATB_USE_TILING_CPY_STREAM=1
TASK_QUEUE_ENABLE=1
ATB_OPERATION_EXECUTE_ASYNC=1
ATB_OPSRUNNER_KERNEL_CACHE_GLOBAL_COUNT=40
```
