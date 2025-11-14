# DAHL

Distributed, Arbitrary and Heterogeneous Learning

## Goals

DAHL is a machine learning framework, developed as part of my thesis, which aims for the reduction
of AI training electricity consumption.

The current tendency in deep learning (or big models in general), seems to be based
on massive investments on enormous clusters with homogeneous machines, because it makes 
parallelizing, indeed very efficient, easy to implement, but has a critical ecological impact.

On the other hand, supporting heterogeneous environments takes a lot more effort in parallelizing
and scheduling, but we think it could provide non-negligible improvements in terms of ecological
impact and economical costs.
In fact, if we break the algorithmic homogeneous constraint, the options to train deep learning
models won't be limited to "buying the same machine with the latest Nvidia GPU in hundred of copies"
anymore but could open the door to "maybe I will only buy fifty of those and combine their power
with existing machines available in my lab/company".

That's why we believe that leveraging the heterogeneity of machines and clusters is the key for more
sustainable AI.

The first step is to implement a machine learning framework that is able to handle heterogeneity
at the machine level. We also want to support arbitrary execution of operations in order to develop
energy-aware load balancing algorithms.

In a second time, we would like to expand our framework to multiple machines.

## Architecture

Under the hood, [StarPU](https://starpu.gitlabpages.inria.fr/) is used to support heterogeneous computing at a single machine level.

## Project structure

```
├── dahl/
│   ├── datasets/    Datasets directory with download scripts
│   ├── examples/    Examples of machine leraning models
│   ├── include/     DAHL public headers
│   ├── src/         DAHL sources
│   └── tests/       Unit testing
├── nix/
│   ├── alumet/      Alumet power measuring software
│   └── starpu/      StarPU flake and dependencies
├── notes/
│   ├── archives/
│   ├── design/      Discussions concerning problems faced during implementation
│   ├── flamegraphs/
│   └── lab/         My laboratory notebook
├── python-version/  Pytorch equivalent of CNN implemented with DAHL
├── scripts/
├── flake.lock
└── flake.nix        Nix flake that provide a reproducible development environment
```

## Building

This project can be built with Nix

[Download Nix on whatever OS](https://nixos.org/download/#multi-user-installation-recommended)

Enter the development environment
```bash
nix develop
```

Compiling
```bash
cd dahl
mkdir build
cd build
cmake ..
make
```

Try the tests to see if it works with `./tests/`

## Running

The first available example is a CNN trained with the fashion mnist dataset

Download the dataset, see `./dahl/datasets/`

```bash
cd ./datasets
./download_fashion_mnist.bash
```

Train the CNN
```bash
cd build
./basic_cnn ../fashion-mnist/train-images-idx3-ubyte ../fashion-mnist/train-labels-idx1-ubyte
```

## Acknowledgements

Thanks tsoding for the [Arena implementation](https://github.com/tsoding/arena).
