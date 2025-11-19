---
tldr: Trying a bigger dataset, loading data takes a lot of time, parallelizing could be better.
commit: `2b39d66551e95d1cc86ba2180d7e8940322afa31`
---

Commands:
```bash
STARPU_SCHED=dmda STARPU_TRACE_BUFFER_SIZE=1028 STARPU_GENERATE_TRACE=1 STARPU_FXT_TRACE=1 STARPU_FXT_PREFIX=./traces ./basic_cnn
```
(hard coded dataset path in this commit)

In these experiments I used big fashion dataset (https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data).
However images were too big to be loaded in memory:
```
#sample* nx   * ny   * nz * sizeof(double)
44441  * 1440 * 1080 * 3  * 8              * 10^-9 ~= 1658 Gigabyte.
```

So I cropped them to 270x360*3.

When running on my laptop (14 cpu) vs rammus (28 cpu + rtx 4060), we notice that things are getting
better parallelized on my laptop.
We notice strange things on GPU execution, while most task are effectivitly 10 times faster than
CPU, some of them take as much time.

We still notice too much dependency constraints on the conv2d backward filters because it produces a
checkerboard (./rammus-cpu-gpu-big-fashion.trace).

On the gpu only run (./rammus-gpu-only-big-fashion.trace), everything is greatly
parallelized (well there's only one GPU :))
We again notice drastic task lenght differences, even on same codelets.For example, conv2d has one
execution of 29milliseconds, but most of the others are around 0.03 milliseconds

On the cpu only run (./rammus-cpu-only-big-fashion.trace), parallelization is better, but
we still notice drawbacks of checkerboard and waiting in between the layers.

--- 
All the previous experiments were ran with batch_size = 60.
Now we'll compare different batch sizes.

(./rammus-cpu-gpu-120-batch-size-big-fashion.trace) here we see solid blocks when scheduling
especially for the forward passes, however we still see the same issues during the backward.

We get the same behavior when reducing to batch size 30 (./rammus-cpu-gpu-30-batch-size-big-fashion.trace).

Work should be done to tackle the different issues in the backward pass.
