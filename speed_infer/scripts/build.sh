#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex
SCRIPT_DIR=$(cd $(dirname -- $0); pwd)
CURRENT_DIR=$(pwd)
cd $SCRIPT_DIR
TARGET_FRAMEWORK=torch
USE_CXX11_ABI=$(python3 get_cxx11_abi_flag.py -f "${TARGET_FRAMEWORK}")
cd ..
export CODE_ROOT=`pwd`
export CACHE_DIR=$CODE_ROOT/build
export OUTPUT_DIR=$CODE_ROOT/output
THIRD_PARTY_DIR=$CODE_ROOT/3rdparty
README_DIR=$CODE_ROOT
COMPILE_OPTIONS=""
INCREMENTAL_SWITCH=OFF
HOST_CODE_PACK_SWITCH=ON
DEVICE_CODE_PACK_SWITCH=ON
USE_VERBOSE=OFF
BUILD_OPTION_LIST="3rdparty download_testdata unittest unittest_and_run pythontest pythontest_and_run debug release help python_unittest_and_run"
BUILD_CONFIGURE_LIST=("--output=.*" "--cache=.*" "--verbose" "--incremental" "--gcov" "--no_hostbin" "--no_devicebin" "--use_cxx11_abi=0" 
    "--use_cxx11_abi=1" "--build_config=.*" "--optimize_off" "--use_torch_runner" "--use_lccl_runner" "--use_hccl_runner" "--doxygen" "--no_warn")


function fn_build_nlohmann_json()
{
    NLOHMANN_DIR=$THIRD_PARTY_DIR/nlohmannJson/include
    if [ ! -d "$NLOHMANN_DIR" ];then
        cd $CACHE_DIR
        rm -rf nlohmann
        mkdir nlohmann
        cd nlohmann
        wget --no-check-certificate https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/include.zip
        unzip include.zip
        mkdir -p $THIRD_PARTY_DIR/nlohmannJson
        cp -r ./include $THIRD_PARTY_DIR/nlohmannJson
        cd $CACHE_DIR
        rm -rf nlohmann
    fi
}

function fn_build_3rdparty()
{
    rm -rf $CACHE_DIR
    mkdir $CACHE_DIR
    cd $CACHE_DIR
    fn_build_nlohmann_json
    cd ..
}

function fn_init_pytorch_env()
{
    export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
    if [ -z "${PYTORCH_NPU_INSTALL_PATH}" ];then
        export PYTORCH_NPU_INSTALL_PATH="$(python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))')"
    fi
    echo "PYTHON_INCLUDE_PATH=$PYTHON_INCLUDE_PATH"
    echo "PYTHON_LIB_PATH=$PYTHON_LIB_PATH"
    echo "PYTORCH_INSTALL_PATH=$PYTORCH_INSTALL_PATH"
    echo "PYTORCH_NPU_INSTALL_PATH=$PYTORCH_NPU_INSTALL_PATH"

    COUNT=`grep get_tensor_npu_format ${PYTORCH_NPU_INSTALL_PATH}/include/torch_npu/csrc/framework/utils/CalcuOpUtil.h | wc -l`
    if [ "$COUNT" == "1" ];then
        echo "use get_tensor_npu_format"
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTORCH_GET_TENSOR_NPU_FORMAT_OLD=ON"
    else
        echo "use GetTensorNpuFormat"
    fi

    COUNT=`grep SetCustomHandler ${PYTORCH_NPU_INSTALL_PATH}/include/torch_npu/csrc/framework/OpCommand.h | wc -l`
    if [ $COUNT -ge 1 ];then
        echo "use SetCustomHandler"
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTORCH_SETCUSTOMHANDLER=ON"
    else
        echo "not use SetCustomHandler"
    fi
}

function fn_build()
{
    fn_build_3rdparty
    if [ ! -d "$OUTPUT_DIR" ];then
        mkdir -p $OUTPUT_DIR
    fi
    if [ "$INCREMENTAL_SWITCH" == "OFF" ];then
        rm -rf $CACHE_DIR
    fi
    if [ ! -d "$CACHE_DIR" ];then
        mkdir $CACHE_DIR
    fi
    cd $CACHE_DIR
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_INSTALL_PREFIX=$OUTPUT_DIR/atb_speed"

    cxx11_flag_str="--use_cxx11_abi"
    if [[ "$COMPILE_OPTIONS" == *$cxx11_flag_str* ]]
    then
    echo "compile_options contain cxx11_abi"
    else
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=${USE_CXX11_ABI}"
    fi


    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    cmake $CODE_ROOT $COMPILE_OPTIONS
    if [ "$INCREMENTAL_SWITCH" == "OFF" ];then
        make clean
    fi
    if [ "$USE_VERBOSE" == "ON" ];then
        VERBOSE=1 make -j
    else
        make -j
    fi
    make install
}

function fn_main()
{
    if [ -z $ATB_HOME_PATH ];then
        echo "env ATB_HOME_PATH not exist, please source atb's set_env.sh"
        exit -1
    fi

    PYTORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__)')"
    if [ ${PYTORCH_VERSION:0:5} == "1.8.0" ] || [ ${PYTORCH_VERSION:0:4} == "1.11" ];then
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTORCH_18=ON"
    fi

    if [[ "$BUILD_OPTION_LIST" =~ "$1" ]];then
        if [[ -z "$1" ]];then
            arg1="release"
        else
            arg1=$1
            shift
        fi
    else
        cfg_flag=0
        for item in ${BUILD_CONFIGURE_LIST[*]};do
            if [[ $1 =~ $item ]];then
                cfg_flag=1
                break 1
            fi
        done
        if [[ $cfg_flag == 1 ]];then
            arg1="release"
        else
            echo "argument $1 is unknown, please type build.sh help for more imformation"
            exit -1
        fi
    fi

    until [[ -z "$1" ]]
    do {
        arg2=$1
        case "${arg2}" in
        --output=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the output directory is not set. This should be set like --output=<outputDir>"
            else
                cd $CURRENT_DIR
                if [ ! -d "$arg2" ];then
                    mkdir -p $arg2
                fi
                export OUTPUT_DIR=$(cd $arg2; pwd)
            fi
            ;;
        --cache=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the cache directory is not set. This should be set like --cache=<cacheDir>"
            else
                cd $CURRENT_DIR
                if [ ! -d "$arg2" ];then
                    mkdir -p $arg2
                fi
                export CACHE_DIR=$(cd $arg2; pwd)
            fi
            ;;
        "--use_cxx11_abi=1")
            USE_CXX11_ABI=ON
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=ON"
            ;;
        "--use_cxx11_abi=0")
            USE_CXX11_ABI=OFF
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=OFF"
            ;;
        "--no_warn")
            ENABLE_WARNINGS=OFF
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DENABLE_WARNINGS=OFF"
            ;;
        "--verbose")
            USE_VERBOSE=ON
            ;;
        "--incremental")
            INCREMENTAL_SWITCH=ON
            ;;
        "--optimize_off")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_OPTIMIZE=OFF"
            ;;
        --link_python=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the python version is not set. This should be set like --link_python=python3.7|python3.8|python3.9"
            else
                COMPILE_OPTIONS="${COMPILE_OPTIONS} -DLINK_PYTHON=$arg2"
            fi
            ;;
        "--use_torch_runner")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_TORCH_RUNNER=ON"
            ;;
        esac
        shift
    }
    done

    fn_init_pytorch_env
    case "${arg1}" in
        "download_testdata")
            fn_download_testdata
            ;;
        "debug")
            COMPILE_OPTIONS="${COMPILE_OPTIONS}"
            fn_build
            ;;
        "release")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
            fn_build
            ;;
        "help")
            echo "build.sh 3rdparty|unittest|unittest_and_run|pythontest|pythontest_and_run|debug|release --incremental|--gcov|--no_hostbin|--no_devicebin|--output=<dir>|--cache=<dir>|--use_cxx11_abi=0|--use_cxx11_abi=1|--build_config=<path>"
            ;;
        *)
            echo "unknown build type:${arg1}";
            ;;
    esac
}

fn_main "$@"
