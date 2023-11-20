cmake_minimum_required(VERSION 3.8)
project("atb_speed")

#options, these can be pass in by cmake argument
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
option(USE_OPTIMIZE "USE_OPTIMIZE" ON)

# compile flag
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/output/atb"
        CACHE PATH "..." FORCE)
endif()
message(STATUS "CMAKE_INSTALL_PREFIX:${CMAKE_INSTALL_PREFIX}")
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-conversion-null")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-all")
if(ENABLE_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-equal")
else()
    add_definitions(-Wno-unused-value -Wno-unused-parameter -Wno-float-equal)
endif()
if(USE_OPTIMIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--build-id=none")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

IF(CMAKE_BUILD_TYPE STREQUAL "Release")
    ADD_LINK_OPTIONS(-s)
ELSE()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -g")
ENDIF()

set(LD_FLAGS_GLOBAL "-shared -rdynamic -ldl -Wl,-z,relro \
    -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} \
    ${LD_FLAGS_GLOBAL} -fexceptions")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} \
    -pie -fPIE")
if(TORCH_18)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_18")
endif()
if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
if(TORCH_GET_TENSOR_NPU_FORMAT_OLD)
    message(STATUS "define TORCH_GET_TENSOR_NPU_FORMAT_OLD")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_GET_TENSOR_NPU_FORMAT_OLD")
endif()
if(TORCH_SETCUSTOMHANDLER)
    message(STATUS "define TORCH_SETCUSTOMHANDLER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_SETCUSTOMHANDLER")
endif()
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/include
    ${PROJECT_SOURCE_DIR}/models
    ${PROJECT_SOURCE_DIR}/layers
    ${PROJECT_SOURCE_DIR}/3rdparty/nlohmannJson/include
    ${PROJECT_SOURCE_DIR}/3rdparty/lccl/include
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{PYTHON_INCLUDE_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    $ENV{PYTORCH_NPU_INSTALL_PATH}/include)

link_directories(
    ${PROJECT_SOURCE_DIR}/3rdparty/lccl/lib
    $ENV{ATB_HOME_PATH}/lib
    $ENV{PYTHON_LIB_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/lib
    $ENV{PYTORCH_NPU_INSTALL_PATH}/lib)

add_subdirectory(core)
add_subdirectory(models)
add_subdirectory(pytorch)
add_subdirectory(layers)

install(FILES ${PROJECT_SOURCE_DIR}/scripts/set_env.sh DESTINATION ./)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/pytorch/examples DESTINATION ./pytorch/)