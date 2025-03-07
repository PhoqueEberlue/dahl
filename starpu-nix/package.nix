{
  # derivation dependencies
  lib,
  fetchzip,
  stdenv,

  # starpu dependencies
  cudaPackages,
  hwloc,
  libuuid,
  libX11,
  fftw, 
  fftwFloat, # Same than previous but with float precision
  pkg-config,
  libtool,
  autoconf,
  automake,
  simgrid ? null, # Can be null because Simgrid is optional
  mpi ? null, # Can be null because MPI support is optional
  cuda ? null, # Can be null because MPI support is optional

  # Options
  enableSimgrid ? false,
  enableMPI ? false,
  enableCUDA ? false,
}:

stdenv.mkDerivation (finalAttrs: rec {
  pname = "starpu";
  version = "1.4.7";

  inherit enableSimgrid;
  inherit enableMPI;
  inherit enableCUDA;

  src = fetchzip {
    url = "https://files.inria.fr/starpu/starpu-${finalAttrs.version}/starpu-${finalAttrs.version}.tar.gz";
    hash = "sha256-6AjQr+2nSJ/uYjJ6II4vJgxj5nHuvlsLvPGZZv/cU8M=";
  };

  # Runtime build dependencies
  nativeBuildInputs = [
    pkg-config
    hwloc
  ]
  ++ lib.optional finalAttrs.enableSimgrid simgrid
  ++ lib.optional finalAttrs.enableMPI mpi
  ++ lib.optional finalAttrs.enableCUDA cuda;

  buildInputs =
  [
    libuuid
    libX11
    fftw
    fftwFloat
    pkg-config
    libtool
    autoconf
    automake
    hwloc
  ]
  ++ lib.optional finalAttrs.enableSimgrid simgrid
  ++ lib.optional finalAttrs.enableMPI mpi
  ++ lib.optional finalAttrs.enableCUDA cuda;

  configureFlags = [ ]
  ++ lib.optional finalAttrs.enableSimgrid "--enable-simgrid"
  ++ lib.optional finalAttrs.enableMPI [ "--enable-mpi" "--enable-mpi-check" "--disable-shared" ]; # Last arg enables static linking which is mandatory for smpi
  # No need to add flags for CUDA, it should be detected by ./configure

  # Some installation scripts use /bin/bash which isn't available in nix
  patches = [ ./starpu-nix-shebang.patch ];

  enableParallelBuilding = true;
  # doCheck = true;

  meta = {
    homepage = "https://starpu.gitlabpages.inria.fr/index.html";
    changelog = "https://files.inria.fr/starpu/starpu-${finalAttrs.version}/log.txt";
    description = "Task programming library for hybrid architectures";
    longDescription = ''
      StarPU is a task programming library for hybrid architectures

      - The application provides algorithms and constraints
        - CPU/GPU implementations of tasks
        - A graph of tasks, using either StarPU’s rich C/C++/Fortran/Python API, or OpenMP pragmas.
      - StarPU internally deals with the following aspects:
        - Task dependencies
        - Optimized heterogeneous scheduling
        - Optimized data transfers and replication between main memory and discrete memories
        - Optimized cluster communications
        - Fully asynchronous execution without spurious waits

      Rather than handling low-level issues, programmers can concentrate on algorithmic aspects!
    '';
    license = lib.licenses.lgpl21;
    # TODO: add myself to the maintainers if publishing on nixpkgs (see https://github.com/NixOS/nixpkgs/tree/master/maintainers)
    maintainers = [ lib.maintainers.PhoqueEberlue ];
  };
})
