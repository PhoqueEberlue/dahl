This little project shows how to reproduce a redux bug that occurs specifically on GPU.
On CPU, each task writing to a handle with redux mode will accumulate the results, however on GPU,
only the first task will be accumulated (with itself), so the result will be equal to the first task
output.

Tried with StarPU at the following commits `c8c9c9c473c09e2654642d1ce8da76fb59ac9871` (2 weeks ago) and 
`0acb729335f742fab3e3bf6bb3d58925c7a7a94a` (2 days ago).

```bash
andrew@rammus ~/d/sandbox (cuda) [nix] > rm -rf build && mkdir build && cd build && cmake ..
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is GNU 13.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /nix/store/zx71vq7s1v840wqsrw2m2ckmxn413a2b-gcc-wrapper-13.3.0/bin/gcc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /nix/store/zx71vq7s1v840wqsrw2m2ckmxn413a2b-gcc-wrapper-13.3.0/bin/g++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- The CUDA compiler identification is NVIDIA 12.4.99
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /nix/store/g5hdvh2jc4kbma1g1y77hjzrwayf8jzm-cuda_nvcc-12.4.99/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Found PkgConfig: /nix/store/dvqr74scragfh4mgxpa03npa57xkdqdg-pkg-config-wrapper-0.29.2/bin/pkg-config (found version "0.29.2")
-- Checking for module 'starpu-1.4'
--   Found starpu-1.4, version 1.4.99
-- Configuring done (2.8s)
-- Generating done (0.0s)
-- Build files have been written to: /home/andrew/dahl/sandbox/build
andrew@rammus ~/d/s/build (cuda) [nix] > make
[ 25%] Building CUDA object CMakeFiles/sandbox_cuda_codelets.dir/src/kernel.cu.o
[ 50%] Linking CUDA static library libsandbox_cuda_codelets.a
[ 50%] Built target sandbox_cuda_codelets
[ 75%] Building C object CMakeFiles/sandbox.dir/src/main.c.o
In file included from /nix/store/dc9s2cqvwslmx5lsfidnn60v9af044zw-glibc-2.40-66-dev/include/bits/libc-header-start.h:33,
                 from /nix/store/dc9s2cqvwslmx5lsfidnn60v9af044zw-glibc-2.40-66-dev/include/stdio.h:28,
                 from /nix/store/ccg9zgy7ayr2zgaiqwc3kgnm5i4p24dg-starpu-master/include/starpu/1.4/starpu_util.h:20,
                 from /nix/store/ccg9zgy7ayr2zgaiqwc3kgnm5i4p24dg-starpu-master/include/starpu/1.4/starpu_data.h:21,
                 from /home/andrew/dahl/sandbox/src/main.c:1:
/nix/store/dc9s2cqvwslmx5lsfidnn60v9af044zw-glibc-2.40-66-dev/include/features.h:422:4: warning: #warning _FORTIFY_SOURCE requires compiling with optimization (-O) [-Wcpp]
  422 | #  warning _FORTIFY_SOURCE requires compiling with optimization (-O)
      |    ^~~~~~~
[100%] Linking CUDA executable sandbox
[100%] Built target sandbox
andrew@rammus ~/d/s/build (cuda) [nix] > STARPU_NCUDA=0 ./sandbox
[starpu][starpu_initialize] Warning: StarPU was configured with --enable-debug (-O0), and is thus not optimized
[starpu][starpu_initialize] Warning: StarPU was configured with --enable-spinlock-check, which slows down a bit
[starpu][_starpu_init_topology] Warning: there are several kinds of CPU on this system. For now StarPU assumes all CPU are equal
[starpu][_starpu_initialize_workers_bindid] Warning: hwloc reported 28 logical CPUs for 20 cores, this is not homogeneous, will assume 1 logical CPUs per core
called cuda accumulate
called cuda accumulate # <--- Called two times, make sense
15.000000 20.000000 15.000000 20.000000  # <------------ Correct result when running on CPU
andrew@rammus ~/d/s/build (cuda) [nix] > STARPU_NCPU=0 ./sandbox
[starpu][starpu_initialize] Warning: StarPU was configured with --enable-debug (-O0), and is thus not optimized
[starpu][starpu_initialize] Warning: StarPU was configured with --enable-spinlock-check, which slows down a bit
[starpu][_starpu_init_topology] Warning: there are several kinds of CPU on this system. For now StarPU assumes all CPU are equal
[starpu][_starpu_initialize_workers_bindid] Warning: hwloc reported 28 logical CPUs for 20 cores, this is not homogeneous, will assume 1 logical CPUs per core
[starpu][initialize_lws_policy] Warning: you are running the default lws scheduler, which is not a very smart scheduler, while the system has GPUs or several memory nodes. Make sure to read the StarPU documentation about adding performance models in order to be able to use the dmda or dmdas scheduler instead.
[starpu][_starpu_cuda_driver_init] Warning: reducing STARPU_CUDA_PIPELINE to 0 because blocking drivers are enabled (and simgrid is not enabled)
called cuda accumulate # <--- Called only on time for the first task
10.000000 15.000000 10.000000 15.000000  # <------------ Wrong result when running on GPU
andrew@rammus ~/d/s/build (cuda) [nix] >
```

I also tried to disable blocking drivers just in case, but the behavior stays the same.
