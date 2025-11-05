cmake_minimum_required (VERSION 3.30.5)
project(sandbox)
 
set(CMAKE_C_STANDARD 23)

# CUDA config
set(CMAKE_CUDA_ARCHITECTURES all-major)
enable_language(CUDA)

# Shows compile command in ./build/compile_commands.json, useful for clangd language server
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Also makes it easier for clangd to see cuda dependencies
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# ------------ RELEASE OPTIONS ------------
# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

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


# Get every .h and .c in ./src ./include, and .cu in kernerls
file(GLOB_RECURSE sandbox_includes CONFIGURE_DEPENDS "./include/*.h")
file(GLOB_RECURSE sandbox_sources CONFIGURE_DEPENDS "./src/*.h" "./src/*.c")
file(GLOB_RECURSE sandbox_cuda_sources CONFIGURE_DEPENDS "./src/*.cu")

# Creating a static library for cuda kernels
add_library(sandbox_cuda_codelets STATIC ${sandbox_cuda_sources})

add_executable(sandbox
    ${sandbox_includes}
    ${sandbox_sources}
)

target_link_libraries(sandbox PRIVATE sandbox_cuda_codelets -fsanitize=undefined)
