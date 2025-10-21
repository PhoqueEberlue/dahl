{
  description = "Starpu flake that provides both dev environment, the starpu package itself, and
    associated debugging tools.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-parts.inputs.nixpkgs.follows = "nixpkgs";
    # Required when build starpu-local
    self.submodules = true;
  };

  outputs = { self, nixpkgs, flake-parts }@inputs: 
  flake-parts.lib.mkFlake { inherit inputs; } {
    systems = [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ];

    # This will generate outputs for each supported systems
    perSystem =
      { system, pkgs, ... }:
      let
        # Cuda dependencies
        enableCUDA = if system == "x86_64-linux" || system == "aarch64-linux" then true else false;
        cudaPackages = if enableCUDA then pkgs.cudaPackages else null;

        # Starpu dependencies
        fxt = pkgs.callPackage ./dependencies/fxt.nix {};
        pajeng = pkgs.callPackage ./dependencies/pajeng.nix {};
        hwloc = pkgs.hwloc.override { enableCuda = enableCUDA; };
        
        # Visualization tool
        vite = pkgs.callPackage ./dependencies/vite.nix { qtPackages = pkgs.libsForQt5.qt5; };

        # Starpu itself
        starpu = pkgs.callPackage ./starpu.nix {
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
        # Configuring nixpkgs options
        _module.args.pkgs = import self.inputs.nixpkgs {
          inherit system;
          # Required for cudaPackages, as it is proprietary software
          config.allowUnfree = true;
        };
         
        # Outpts of the flake: here we return a starpu and vite package
        packages = {
          starpu-master = starpu;
          starpu-local = starpu.overrideAttrs {
            version = "local";
            src = ./src;
          };
          inherit vite fxt pajeng;
        };

        # Expose cudaPackages so external flakes can reuse them
        legacyPackages = {
          inherit cudaPackages enableCUDA;
        };
      };
  };
}
