## Useful commands

Running on CPU only:
```bash
STARPU_NCUDA=0 ./executable
```

Running on GPU only:
```bash
STARPU_NCPU=0 ./executable
```

### Debugging

Make sure to enable starpu debug.
This can be done passing the flag "--enable-debug" when compiling starpu.
With the nix flake of the project this can be toggled in the arguments of the nix derivation:

```nix
starpu-master = starpu.packages.${system}.starpu-master.override {
  enableCUDA = enableCUDA;
  enableDebug = true;
};
```

Also enable debug for dahl in the `./CMakeLists.txt`:

```cmake
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -Wall -fsanitize=undefined")
```


### Traces

Generate traces (note that StarPU Debug mode should be enabled)
```bash
STARPU_TRACE_BUFFER_SIZE=1028 STARPU_GENERATE_TRACE=1 STARPU_FXT_TRACE=1 STARPU_FXT_PREFIX=./traces ./executable
```

Reduce trace file size
```bash
starpu_fxt_tool -i prof_file_andrew_0 -no-bus -no-flops -no-counter -no-events
```

Display GANTT diagram of the tasks (env -u for wayland users)
```bash
env -u WAYLAND_DISPLAY vite paje.trace & disown
```

Generate a graph of the tasks
```bash
dot -Tpdf dag.dot -o output.pdf
```
