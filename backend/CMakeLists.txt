# Builds backend/build/Release/tsgrad_native.node, which src/native.ts loads.
#
# From the repo root, after `npm install --save-dev node-addon-api`:
#   cmake -S backend -B backend/build
#   cmake --build backend/build --config Release
#
# To build without cuda
#   cmake -S backend -B backend/build -DTSGRAD_CUDA=OFF
#   cmake --build backend/build --config Release
#
# A local CUDA build targets the machine's GPU. Release builds pass an
# explicit list so one binary covers the published architectures:
#   cmake -S backend -B backend/build -DCMAKE_CUDA_ARCHITECTURES=75;80;86;89;90
#
# TSGRAD_CUDA=ON (the default) compiles src/cuda/*.cu when nvcc is installed.
# Those files define tsgrad::cuda::*. The CPU files define tsgrad::*.
# addon.cc calls the CPU entry points. A CPU function may call the matching
# tsgrad::cuda function once you write that handoff.

cmake_minimum_required(VERSION 3.30)
project(tsgrad_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(TSGRAD_CUDA "Compile the CUDA launchers in src/cuda" ON)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

execute_process(
    COMMAND node -p "require('path').resolve(require('node-addon-api').include_dir)"
    WORKING_DIRECTORY "${REPO_ROOT}"
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NODE_ADDON_API_RESULT
)
if(NOT NODE_ADDON_API_RESULT EQUAL 0)
    message(FATAL_ERROR "node-addon-api was not found. From the repo root: npm install --save-dev node-addon-api")
endif()

execute_process(
    COMMAND node -p "require('node-api-headers').include_dir"
    WORKING_DIRECTORY "${REPO_ROOT}"
    OUTPUT_VARIABLE NODE_API_HEADERS_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NODE_API_HEADERS_RESULT
)
if(NOT NODE_API_HEADERS_RESULT EQUAL 0)
    message(FATAL_ERROR "node-api-headers was not found. It is installed with node-addon-api.")
endif()

get_filename_component(NODE_API_HEADERS_PARENT "${NODE_API_HEADERS_DIR}" DIRECTORY)
set(NODE_API_DEF "${NODE_API_HEADERS_PARENT}/def/node_api.def")

add_library(tsgrad_native SHARED
    src/addon.cc
    src/storage.cpp
    src/cpu/elementwise.cpp
    src/cpu/reduce.cpp
    src/cpu/matmul.cpp
    src/cpu/conv.cpp
    src/cpu/optim.cpp
    src/cpu/gather.cpp
    src/cpu/scatter.cpp
    src/cpu/embedding.cpp
    src/cpu/layernorm.cpp
    src/cpu/softmax.cpp
    src/cpu/rope.cpp
    src/cpu/attention.cpp
)

# native.ts loads this exact file. Per-config output dirs are set so Visual Studio
# does not append a second Release/ directory.
set(TSGRAD_NODE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/Release")
set_target_properties(tsgrad_native PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    OUTPUT_NAME "tsgrad_native"
    RUNTIME_OUTPUT_DIRECTORY "${TSGRAD_NODE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TSGRAD_NODE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TSGRAD_NODE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TSGRAD_NODE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${TSGRAD_NODE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${TSGRAD_NODE_DIR}"
)

target_include_directories(tsgrad_native PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${NODE_ADDON_API_DIR}"
    "${NODE_API_HEADERS_DIR}"
)

target_compile_definitions(tsgrad_native PRIVATE
    NAPI_VERSION=8
    NAPI_CPP_EXCEPTIONS
)

if(MSVC)
    target_compile_options(tsgrad_native PRIVATE /EHsc)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(NODE_LIB_MACHINE "x64")
    else()
        set(NODE_LIB_MACHINE "x86")
    endif()
    set(NODE_LIB "${CMAKE_CURRENT_BINARY_DIR}/node.lib")
    add_custom_command(
        TARGET tsgrad_native PRE_LINK
        COMMAND "${CMAKE_AR}" /nologo /machine:${NODE_LIB_MACHINE} /def:${NODE_API_DEF} /out:${NODE_LIB}
        COMMENT "Creating node.lib from node_api.def"
    )
    target_link_libraries(tsgrad_native PRIVATE "${NODE_LIB}" delayimp.lib)
    target_link_options(tsgrad_native PRIVATE "/DELAYLOAD:node.exe")
elseif(APPLE)
    target_link_options(tsgrad_native PRIVATE "-undefined" "dynamic_lookup")
else()
    target_link_options(tsgrad_native PRIVATE "-Wl,--unresolved-symbols=ignore-all")
endif()

if(TSGRAD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(_tsgrad_cuda_arch "${CMAKE_CUDA_ARCHITECTURES}")
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        if(NOT _tsgrad_cuda_arch)
            set(_tsgrad_cuda_arch native)
        endif()
        target_sources(tsgrad_native PRIVATE
            src/cuda/context.cu
            src/cuda/elementwise.cu
            src/cuda/matmul.cu
            src/cuda/conv.cu
            src/cuda/optim.cu
            src/cuda/gather.cu
            src/cuda/scatter.cu
            src/cuda/embedding.cu
            src/cuda/layernorm.cu
            src/cuda/softmax.cu
            src/cuda/rope.cu
            src/cuda/attention.cu
        )
        set_target_properties(tsgrad_native PROPERTIES CUDA_ARCHITECTURES "${_tsgrad_cuda_arch}")
        target_include_directories(tsgrad_native PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(tsgrad_native PRIVATE CUDA::cudart CUDA::cublas)
        target_compile_definitions(tsgrad_native PRIVATE TSGRAD_HAS_CUDA)
    else()
        message(WARNING "nvcc was not found. The addon will contain the CPU entry points only. src/cuda/*.cu stays in the tree.")
    endif()
endif()

if(APPLE)
    set_target_properties(tsgrad_native PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX)
    set_target_properties(tsgrad_native PROPERTIES
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

if(MSVC)
    set(OpenMP_RUNTIME_MSVC "llvm")
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(tsgrad_native PRIVATE OpenMP::OpenMP_CXX)
