{
  description = "Nix flake for a development shell with CUDA, StarPU, and nixGLHost";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

    # Required to run CUDA because of problems between cuda drivers (usually handled by the machine OS) and
    # the runtime cuda dependencies, here handled by this nix flake.
    nixglhost-src = {
      url = "github:numtide/nix-gl-host/main";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, nixglhost-src }: 
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { 
          inherit system; 
          config = {
            allowUnfree = true; 
            # Apparently valgrind is broken on MacOS so we need this
            allowBroken = true;
          };
        };
      }); 
    in
      {
      # Generate a devShell for each platform
      devShells = forEachSupportedSystem ({ pkgs }: 
        let 
          system = pkgs.system;

          # Only enable CUDA on linux
          enableCUDA = if system == "x86_64-linux" || system == "aarch64-linux" then true else false;
          cudaPackages = if enableCUDA then pkgs.cudaPackages else null;
          hwloc = pkgs.hwloc.override { enableCuda = enableCUDA; };
          # nixgl is only needed for cuda executions
          nixglhost = if enableCUDA then pkgs.callPackage "${nixglhost-src}/default.nix" { } else null;

          # Building from my local derivation of StarPU until it is available on nixpkgs
          starpu = pkgs.callPackage ./starpu.nix { 
            cudaPackages = cudaPackages;
            hwloc = hwloc;
            enableCUDA = enableCUDA;
          };
        in
          {
          # Actually defining the dev environment
          default = pkgs.mkShell
            {
              packages = with pkgs; [
                starpu 
                clang
                valgrind
                scc
                hwloc
                czmq
              ] ++ (if enableCUDA then [
                  cudaPackages.cuda_cudart cudaPackages.cuda_nvcc cudaPackages.cudatoolkit nixglhost] else []);

              nativeBuildInputs = with pkgs; [
                cmake
                pkg-config
              ];
            };
        });
    };
}
