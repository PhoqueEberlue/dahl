cmake_minimum_required (VERSION 3.30.5)
project(dahl)
 
set(CMAKE_C_STANDARD 23)

# TODO: chose the correct one: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch
set(CMAKE_CUDA_ARCHITECTURES all-major)
enable_language(CUDA)

# Shows compile command in ./build/compile_commands.json, useful for clangd language server
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------ RELEASE OPTIONS ------------
# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")

# ------------ DEBUG OPTIONS ------------
# Enable debug flags and avoid optimizations
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -Wall -fsanitize=undefined")

find_package(PkgConfig)
pkg_check_modules(STARPU REQUIRED starpu-1.4)
if (STARPU_FOUND)
    include_directories (${STARPU_INCLUDE_DIRS})
    link_directories    (${STARPU_STATIC_LIBRARY_DIRS})
    link_libraries      (${STARPU_STATIC_LIBRARIES})
else (STARPU_FOUND)
    message(FATAL_ERROR "StarPU not found")
endif()

# Example of linking cuda file to c
add_library(scal_obj STATIC kernels/scal.cu)

# Get every .h and .c in ./src ./include, and .cu in kernerls
file(GLOB_RECURSE dahl_includes CONFIGURE_DEPENDS "./include/*.h")
file(GLOB_RECURSE dahl_sources CONFIGURE_DEPENDS "./src/*.h" "./src/*.c")
file(GLOB_RECURSE dahl_kernels CONFIGURE_DEPENDS "./kernels/*.cu")

# Tests files
file(GLOB_RECURSE dahl_tests CONFIGURE_DEPENDS "./tests/*.c")

set(kernel_files
    kernels/scal.cu
)

add_library(dahl
    ${dahl_includes}
    ${dahl_sources}
    ${dahl_kernels}
)

add_executable(tests
    ${dahl_tests}
)

add_executable(basic_cnn
    "./examples/basic_cnn.c"
)

target_link_libraries(dahl PRIVATE scal_obj -fsanitize=undefined)
target_link_libraries(tests PUBLIC dahl -fsanitize=undefined)
target_link_libraries(basic_cnn PUBLIC dahl -fsanitize=undefined)
