let
    pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-unstable.tar.gz") {};
in 
pkgs.mkShell {
  packages = with pkgs.python312Packages; [
      numpy
      scipy
      ipython
      tensorflow
      keras
      torch
      torchvision
      scikit-learn
      matplotlib
      seaborn
  ];
}
