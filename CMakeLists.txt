cmake_minimum_required (VERSION 3.28)
project(dahl)
 
set(CMAKE_C_STANDARD 23)

# Shows compile command in ./build/compile_commands.json, useful for clangd language server
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable debug flags and avoid optimizations
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -Wall")
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")

SET(GCC_COVERAGE_LINK_FLAGS    "-lm")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

# Get every .h and .c in ./src ./include, and .cu in kernerls
file(GLOB_RECURSE dahl_includes CONFIGURE_DEPENDS "./include/*.h")
file(GLOB_RECURSE dahl_sources CONFIGURE_DEPENDS "./src/*.h" "./src/*.c")
file(GLOB_RECURSE dahl_tests CONFIGURE_DEPENDS "./tests/*.c")


add_library(dahl
    ${dahl_includes}
    ${dahl_sources}
    ${dahl_kernels}
)

add_executable(tests
    ${dahl_tests}
)

add_executable(example-mnist
    "./examples/fashion-mnist.c"
)

target_link_libraries(dahl PRIVATE czmq)
target_link_libraries(tests PUBLIC dahl)
target_link_libraries(example-mnist PUBLIC dahl)
