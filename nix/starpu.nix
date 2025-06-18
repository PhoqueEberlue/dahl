{
  # derivation dependencies
  lib,
  # fetchzip,
  fetchgit,
  stdenv,

  # starpu dependencies
  hwloc,
  libuuid,
  libX11,
  fftw,
  fftwFloat, # Same than previous but with float precision
  pkg-config,
  libtool,
  autoconf,
  automake,
  simgrid,
  mpi,
  cudaPackages,
  fxt,
  pajeng,
  python3,

  # Options
  enableSimgrid ? false,
  enableMPI ? false,
  enableCUDA ? false,
  enableTracing ? false,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "starpu";
  version = "master";

  inherit enableSimgrid;
  inherit enableMPI;
  inherit enableCUDA;
  inherit enableTracing;

  # src = fetchzip {
  #   url = "https://files.inria.fr/starpu/starpu-${finalAttrs.version}/starpu-${finalAttrs.version}.tar.gz";
  #   hash = "sha256-6AjQr+2nSJ/uYjJ6II4vJgxj5nHuvlsLvPGZZv/cU8M=";
  # };

  src = fetchgit {
    url = "https://gitlab.inria.fr/starpu/starpu.git";
    rev = "bdaf1ea9b55099d8163be086743fec8327475629";
    hash = "sha256-BgUUEps3nLi4nPRoTBYmztO2bLwtJw2Bp5ZKurccG+4=";
  };

  # Runtime build dependencies
  nativeBuildInputs =
    [
      pkg-config
      hwloc
    ]
    ++ lib.optional finalAttrs.enableSimgrid simgrid
    ++ lib.optional finalAttrs.enableMPI mpi
    ++ lib.optional finalAttrs.enableCUDA cudaPackages.cudatoolkit
    ++ lib.optional finalAttrs.enableTracing [ fxt ];

  nativeCheckInputs = 
    [
    ]
    ++ lib.optional finalAttrs.enableTracing [ pajeng python3 fxt ];

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
    ++ lib.optional finalAttrs.enableCUDA cudaPackages.cudatoolkit;

  configureFlags =
    [
      "--enable-debug"
      "--enable-quick-check"
      "--disable-build-examples"
      "--enable-blocking-drivers"
    ]
    ++ lib.optional finalAttrs.enableSimgrid "--enable-simgrid"
    ++ lib.optional finalAttrs.enableMPI [
      "--enable-mpi"
      "--enable-mpi-check"
      "--disable-shared"
    ];
  # Last arg enables static linking which is mandatory for smpi
  # No need to add flags for CUDA, it should be detected by ./configure

  preConfigure = ''
    ./autogen.sh
  '';

  postConfigure = ''
    # Patch shebangs recursively because a lot of scripts are used
    shopt -s globstar
    patchShebangs --build **/*.sh tools/*

    # this line removes a bug where value of $HOME is set to a non-writable /homeless-shelter dir
    export HOME=$(pwd)
    '';

  enableParallelBuilding = true;
  doCheck = true;

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
    maintainers = [ lib.maintainers.PhoqueEberlue ];
  };
})
