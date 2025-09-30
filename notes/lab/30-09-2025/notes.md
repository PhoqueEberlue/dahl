---
tldr: Integrated StarPU redux system and changed starpu perf model to get better parallelization.
commit: `d496ae94d294a95e9d50dd52e37fbb1770db279c`
---

Openning the traces

```bash
vite ./28x28x3-images-scheduling-result.trace
vite ./512x512x3-images-scheduling-result.trace
# For wayland users
env -u WAYLAND_DISPLAY vite ./28x28x3-images-scheduling-result.trace
env -u WAYLAND_DISPLAY vite .5512x512x3-images-scheduling-result.trace
```

Experiments were ran with in debug mode (both StarPU and DAHL)
```bash
STARPU_SCHED=dmda STARPU_NCPU=14 STARPU_CALIBRATE=0 STARPU_TRACE_BUFFER_SIZE=1028 STARPU_GENERATE_TRACE=1 STARPU_FXT_TRACE=1 STARPU_FXT_PREFIX=./traces ./example-mnist
```

> [../24-09-2025/notes.md]
> We still notice some sleep time in red, this is caused by the gather functions (block_sum_z_axis, tensor_sum_t_axis...) that reduces the batch results in order to update weights and biases.
> -> this could probably be fixed by using built-in starpu reduce capabilities, because right now I simply looping through the datas and summing "by hand" at the end of each "layer batch".

So I integrated StarPU's redux system into DAHL and indeed it works flawlessly.
The accumulator tasks are getting parallelized effectively, which indeed reduces synchronism.
-> `./512x512x3-images-batch-size-14-history-model.trace`

However we still notice that some tasks are struggling to be properly parallelized (backward pooling for example).
In fact StarPU is complaining a lot that some tasks were taking longer than what was estimated by the scheduler which causes a lot of sleep time.
Simply changing to a regression model helps a lot!
-> `./512x512x3-images-batch-size-14-regression-model.trace`

Also I tried to play a bit with the batch size and I noticed that when I set batch_size > NCPU the parallelization struggles a bit: 
At the end of some layers every tasks gets queued up to the same core? Pretty weird.
-> `./512x512x3-images-batch-size-32-regression-model.trace`

On the other hand, a batch size < NCPU struggles to fill all CPU cores, which seems pretty fair.
-> `./512x512x3-images-batch-size-4-regression-model.trace`

So all of these changes are really interesting and helps a lot parallelization, however we can clearly notice the synchronization barriers between each layers.
We should find a way to prevent that. One very interesting idea would be to implement a PipeDream-like algorithm, so we could fill all cores and reduce the
synchronization constraints.
But I wonder if we would end up with similar results, i.e. some tasks still running on one cpu at the end of each layer, which slows down the whole training?
The advantage of PipeDream is that we will be able to run multiple different layers pass at the same time, which could permit StarPU to fill the empty spots.
