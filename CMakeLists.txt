cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)



# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 --cudart static --use_fast_math --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_37,code=compute_37 -gencode arch=compute_50,code=compute_50 -gencode arch=compute_52,code=compute_52 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52
    )



set(CMAKE_CXX_FLAGS_DISTRIBUTION --std=c++11)











# Specify include directories
#include_directories(
#    kernels
#    utility
#    )

# Specify library paths
#link_directories(
#    /opt/foobar/lib
#    /opt/joestuff/lib
#    )

# For compilation ...
# Specify target & source files to compile it from

set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_HOME_DIRECTORY}/bin
    )

cuda_add_executable(
     RicCorticalThicknessByNormal
     RicCorticalThicknessByNormal.cpp
     GM_Normal.cpp
     FindNormalDistThread_GPU.cu
     TexFill.cpp
    )




# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
      RicCorticalThicknessByNormal
       -L./
      -lRic
      -pthread
    )
