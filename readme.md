# DAHL

Distributed, Arbitrary and Heterogeneous Learning

## Goals

Providing a Distributed Learning runtime, that can run on heterogenous platforms (a myriad of different machines with multiples CPU and/or GPU), and with arbitrary execution order of the operations.

This will let us room to develop energy-aware load balancing algorithms.


## Architecture

Under the hood, [StarPU](https://starpu.gitlabpages.inria.fr/) is used to support heterogenous computing at a single machine level.

## Project structure

- `examples/` contains ML training examples
- `include/` expose public headers of DAHL
- `kernels/` CUDA kernels
- `tests/` unit testing for DAHL
- `design-talk/` here I discuss about the problems faced during the implementation
- `flake.nix` Nix flake with every needed dependency for developping DAHL

## Building

This project can be built with Nix

[Download Nix on whatever OS](https://nixos.org/download/#multi-user-installation-recommended)

Enter the development environment
```bash
nix develop
```

Compiling
```bash
mkdir build
cd build
cmake ..
make
```

Try the tests to see if it works with `./tests/`

## Running

The first available example is a CNN trained with the fashion mnist dataset

Download the dataset, see `./datasets/`

```bash
cd ./datasets
./download_fashion_mnist.bash
```

Train the CNN
```bash
./example-mnist ../fashion-mnist/train-images-idx3-ubyte ../fashion-mnist/train-labels-idx1-ubyte
```

## Acknowledgements

Thanks tsoding for the [Arena implementation](https://github.com/tsoding/arena).
