let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-24.11.tar.gz") { config.allowUnfree = true; };

  cuda = pkgs.cudaPackages.cudatoolkit;
  hwloc = (pkgs.hwloc).override { enableCuda = true; };
  starpu = pkgs.callPackage ./starpu-nix/package.nix { cuda = cuda; hwloc = hwloc ; enableCUDA = true; };

  nixglhost-sources = pkgs.fetchFromGitHub {
    owner = "numtide";
    repo = "nix-gl-host";
    rev = "main";
    hash = "sha256-QZCKJoypcwgS3tDNSWMjlxEBZtOYPW3eXV24rMzKsac=";
  };

  nixglhost = pkgs.callPackage "${nixglhost-sources}/default.nix" { };

in 
  pkgs.mkShell {
    imports = [ ];

    packages = with pkgs; [
      starpu
      nixglhost

      # Maybe unify cuda sources
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
  }
