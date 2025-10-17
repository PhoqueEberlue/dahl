---
tldr: I compared the impact of the image size on starpu's scheduling and found out that, indeed small tasks heavly degrade performances.
commit: `900d3e51e33f71ac4a1bad59c1580d214ca4502d`
---

Openning the traces in this folder (using the nix flake ):

```bash
vite ./28x28x3-images-scheduling-result.trace
vite ./512x512x3-images-scheduling-result.trace
# For wayland users
env -u WAYLAND_DISPLAY vite ./28x28x3-images-scheduling-result.trace
env -u WAYLAND_DISPLAY vite .5512x512x3-images-scheduling-result.trace
```

When using 28x28x3 images the tasks are very small which make it way hard for starpu to schedule tasks on multiple cpu.
We even notice a lot of overhead and scheduling on the traces.
I knew that it was probably linked to the task size but wanted to improve more my parallelizion before trying this.
So we notice that 512x512 images lead to way better scheduling, e.g. tasks a executed (almost) continously and are spread on every cpus.  
We still notice some sleep time in red, this is caused by the gather functions (block_sum_z_axis, tensor_sum_t_axis...) that reduces the batch results in order to update weights and biases.
-> this could probably be fixed by using built-in starpu reduce capabilities, because right now I simply looping through the datas and summing "by hand" at the end of each "layer batch".

This is a pretty interesting result because it showcases the limitations of using a TGCS.
Of course, using bigger dataset that requires more computation will make DAHL shine because it will parallelize well.
Nevetherless, we should probably investigate more about the task granularity: it could be really interesting to have ways to "cut" or "gather" tasks together depending on the size of the input data.
Maybe for the 28x28 images we should do data parallelism? and simply pack the tasks together into a big "per batch task".

So we have many ideas to try, however they will be limited by the code.
~Doing this kind of idea heavily changes the user API, which is not particularly a problem in terms of research, but then if we go too far, maybe we are just showing that we can do better if we hardcode everything for our specific task?~
-> not sure with this argument, anyways we shouldn't limit ourselves.
Though its still complicated code wise :)
