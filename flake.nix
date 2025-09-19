{
  description = "Nix flake for DAHL's development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

    # Required to run CUDA because of problems between cuda drivers (usually handled by the machine OS) and
    # the runtime cuda dependencies, here handled by this nix flake.
    nixglhost-src = {
      url = "github:numtide/nix-gl-host/main";
      flake = false;
    };

    alumet.url = "path:./nix/alumet/";
  };

  outputs = { self, nixpkgs, nixglhost-src, alumet }: 
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
          fxt = pkgs.callPackage ./nix/fxt.nix {};
          pajeng = pkgs.callPackage ./nix/pajeng.nix {};
          vite = pkgs.callPackage ./nix/vite.nix { qtPackages = pkgs.libsForQt5.qt5; };

          # Building from my local derivation of StarPU until it is available on nixpkgs
          starpu = pkgs.callPackage ./nix/starpu.nix { 
            cudaPackages = cudaPackages;
            enableCUDA = enableCUDA;
            hwloc = hwloc;

            # Tracing libs
            fxt = fxt;
            pajeng = pajeng;
            enableDebug = true;
          };
        in
          {
          # Actually defining the dev environment
          default = pkgs.mkShell
            {
              packages = with pkgs; [
                starpu 
                alumet.packages.${system}.default
                clang
                valgrind
                scc
                flamegraph
                hwloc
                czmq
                fxt
                pajeng
                vite
                gzip
                riffdiff
                icdiff
                diff-so-fancy
                python312
                python312Packages.termcolor
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
