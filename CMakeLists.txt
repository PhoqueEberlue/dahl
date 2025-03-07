cmake_minimum_required (VERSION 3.28)
project(dahl)
 
set(CMAKE_C_STANDARD 23)

# TODO: chose the correct one: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch
set(CMAKE_CUDA_ARCHITECTURES all-major)
enable_language(CUDA)

# Shows compile command in ./build/compile_commands.json, useful for clangd language server
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable debug flags and avoid optimizations
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -Wall")

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

# Get every .h and .c in ./src/ and put them in variable ${dahl_files}
file(GLOB_RECURSE dahl_files CONFIGURE_DEPENDS "./src/*.h" "./src/*.c")
file(GLOB_RECURSE test_files CONFIGURE_DEPENDS "./tests/*.c")
file(GLOB_RECURSE kernel_files CONFIGURE_DEPENDS "./kernels/*.cu")

set(kernel_files
    kernels/scal.cu
)

add_executable(tests
    ${dahl_files}
    ${kernel_files}
    ${test_files}
)

# TODO: add_library, with an include folder

target_link_libraries(tests PRIVATE czmq scal_obj)
