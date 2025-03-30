{
  description = "Nix flake for a development shell with CUDA, StarPU, and nixGLHost";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    nixglhost-src = {
      url = "github:numtide/nix-gl-host/main";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, nixglhost-src }: 
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { system = system; config.allowUnfree = true; };

      cuda = pkgs.cudaPackages.cudatoolkit;
      hwloc = pkgs.hwloc.override { enableCuda = true; };
      starpu = pkgs.callPackage ./starpu-nix/package.nix { 
        cuda = cuda; 
        hwloc = hwloc; 
        enableCUDA = true; 
      };
      nixglhost = pkgs.callPackage "${nixglhost-src}/default.nix" { };
    in
      {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          starpu
          nixglhost
          cudaPackages.cuda_cudart
          cudaPackages.cuda_nvcc
          cuda
          clang
          valgrind
          hwloc
          czmq
        ];

        nativeBuildInputs = with pkgs; [
          cmake
          pkg-config
        ];
      };
    };
}

