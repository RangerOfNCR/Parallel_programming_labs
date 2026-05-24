cmake_minimum_required(VERSION 3.12)

project(task4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME}
    main.cu
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE CUDA::cudart
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)

add_definitions(-DSRC_DIR="${CMAKE_SOURCE_DIR}")