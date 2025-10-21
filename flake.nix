{
  description = "Nix flake for DAHL's development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-parts.inputs.nixpkgs.follows = "nixpkgs";

    # Required to run CUDA because of problems between cuda drivers (usually handled by the machine OS) and
    # the runtime cuda dependencies, here handled by this nix flake.
    nixglhost-src = {
      url = "github:numtide/nix-gl-host/main";
      flake = false;
    };

    # Alumet is a power measuring software
    alumet.url = "path:./nix/alumet/";

    # Using our starpu flake as input
    starpu.url = "path:./nix/starpu/";

    # Allow us to build starpu-local with the git submodule at `nix/starpu/src`
    self.submodules = true;
  };

  outputs = { self, nixpkgs, flake-parts, nixglhost-src, alumet, starpu }@inputs: 
  flake-parts.lib.mkFlake { inherit inputs; } {
    systems = [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ];

    # Creates a development environment for each system
    perSystem =
      { system, pkgs, ... }:
      let
        # Only enable CUDA on linux
        enableCUDA = if system == "x86_64-linux" || system == "aarch64-linux" then true else false;

        #############################################
        ### Packages produced by the StarPU flake ###
        # Local version of StarPU using sources in `nix/starpu/src`
        starpu-local = (starpu.packages.${system}.starpu-local.override {
          enableCUDA = enableCUDA;
          enableDebug = true;
        }).overrideAttrs { doCheck = false; }; # Disable make check for the local version

        # Gitlab version of StarPU using master branch
        starpu-master = starpu.packages.${system}.starpu-master.override {
          enableCUDA = enableCUDA;
          enableDebug = true;
        };

        # Vite trace visualizer software
        vite = starpu.packages.${system}.vite;
        # Kernel/User trace recording library
        fxt = starpu.packages.${system}.fxt;
        pajeng = starpu.packages.${system}.pajeng;
        # Getting the exact cudaPackages used by StarPU.
        # This is important because we need to match CUDA version at compile time (when compiling
        # StarPU) and CUDA version at runtime (when using it with DAHL).
        cudaPackages = starpu.legacyPackages.${system}.cudaPackages;
        #############################################

        ## TODO: we could also get hwloc from the starpu flake?
        hwloc = pkgs.hwloc.override { enableCuda = enableCUDA; };
        # nixgl is only needed for cuda executions
        nixglhost = pkgs.callPackage "${nixglhost-src}/default.nix" { };

        # Alumet power measuring software
        alumet-0_9_0 = alumet.packages.${system}.default;
        
      in
      {
      # Configuring nixpkgs options
      _module.args.pkgs = import self.inputs.nixpkgs {
        inherit system;
        # Required for cudaPackages, as it is proprietary software
        config.allowUnfree = true;
      };
      devShells =
        {
        # Actually defining the dev environment
        default = pkgs.mkShell
          {
            packages = with pkgs; [
              # Debugging
              valgrind
              pajeng
              flamegraph 
              vite
              fxt

              # StarPU itself with runtime dependencies
              starpu-master
              hwloc

              # DAHL misc
              clang
              scc
              gzip
              gnuplot
              alumet-0_9_0

              python312
              python312Packages.termcolor
              python312Packages.pandas
              python312Packages.matplotlib
              python312Packages.seaborn
              python312Packages.numpy
              python312Packages.ipython

              # tmp from nix-shell python version
              python312Packages.scipy
              python312Packages.torch
              python312Packages.torchvision
              python312Packages.scikit-learn
              python312Packages.matplotlib
              python312Packages.seaborn
            ] ++ lib.optionals enableCUDA [
                cudaPackages.cuda_cudart cudaPackages.cuda_nvcc 
                cudaPackages.cudatoolkit nixglhost ];

            nativeBuildInputs = with pkgs; [
              cmake
              # Required to find starpu with cmake
              pkg-config
            ];
          };
        };
      };
    };
}
