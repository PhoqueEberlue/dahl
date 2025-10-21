## Building StarPU and associated packages:

```bash
nix build .#starpu-local
```

Building starpu from the git submodule:

```bash
nix build .#starpu-local
```

This will perform the exact same building instructions, but using `./src/` instead of pulling the
version available on gitlab.

## Using the packages

Of course we can use the produced packages via `nix run` command:

```bash
# This will run 'vite' trace visualizer
nix run .#vite
```

But obviously starpu is a library, not a binary, so we want to use it in a nix dev environment
instead. It can be imported into other flakes as such:

```nix
# In the inputs:
inputs = {
  # Use the right path
  starpu.url = "path:./nix/starpu/";
};

# ...
# Then import it in the packages: 
starpu.packages.${system}.starpu-master
# or
starpu.packages.${system}.starpu-local
```

It is also possible to toggle modes defined in the derivation `./starpu.nix` using `override`
function:

```nix
starpu-master = starpu.packages.${system}.starpu-master.override {
  enableCUDA = false;
  enableDebug = true;
};
```


## Developping StarPU

```bash
nix develop .#starpu-local
cd ./src
configurePhase
buildPhase
checkPhase
```
