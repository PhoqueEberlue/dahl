# Memory managment

Problem:

Currently (mar. 29 avril 2025 15:07:57 CEST) my API is implemented as follows:
dahl_data.h contains functions that are able to allocate memory dynamically e.g. matrix_init, block_init etc., and have their
corresponding finalize functions.
the codelets themselves are not allocating data at all, just using the provided buffers.
However the dahl_tasks.h layer is able to allocate data by calling functions from dahl_data.h.
e.g. `task_vector_diag` takes a vector, and instanciates a new matrix with vector lenght * vector lenght as its dimensions.

Those functions are also used in the different layers of my CNN.
Which means that every epoch, and every sample, data is allocated.
This is not good for multiple reasons:
- First it can be difficult to track the data dependencies, and where a buffer should actually be finalized (lifetime issue).
- furthermore, starpu work asynchronously so I should await the data to be available (i.e. all tasks related to it ended) in order to free it.
- even if I manage to free my memory in the correct order, it would still be highly inneficient as the allocations require system calls.

(
mer. 30 avril 2025 15:07:50 CEST remark a posteriori:
See the [flamegraph](../flamegraphs/dahl-malloc-over-and-over.svg) of my program, a lot of time is spent in malloc + we have a lot of page faults
because we are always changing our buffers.
)

What we could do instead, would be to reuse the buffers.
At each epoch/sample iteration, we perform exactly the same operations, with exactly the same needs in terms of memory.
So in theory, the maximal memory size can be known at compile time.

One strategy to allocate memory as such is called [Linear allocation/Arena/Region-based allocator].
It makes reasonning in terms of grouped elements instead of single elements.
And you can free all the elements in the group at the same time, without worrying about the data dependency order, pointers etc.

Some readings on that:
https://www.rfleury.com/p/untangling-lifetimes-the-arena-allocator
https://www.gingerbill.org/article/2019/02/01/memory-allocation-strategies-001/
About Rust and Cpp RAII:
"[...] Such languages also have the tendency to couple the concept of ownership with the concept of lifetime, **which are not necessarily linked.**"
I shouldn't forget to align my memory

Let's get back to my example.
I could use my arena in multiple ways:

```md
1.
create my arena with MEM_MAX bytes, actual memory allocation by calling malloc, nmap, windows virtual memory whatever.

for epoch in epochs
    for sample in samples
        allocate using the arena -> O(1)
        forward pass
        backward pass
        reset my arena -> fill the buffer with 0

delete my arena, actual memory deallocation

2.

create my arena with max(MEM_MAX_FORWARD, MEM_MAX_BACKWARD) bytes.

for epoch in epochs
    for sample in samples
        allocate using the arena
        forward pass

        reset my arena

        allocate using the arena
        backward pass

        reset my arena

delete my arena, actual memory deallocation
```

In 1., I allocate and deallocate at the beginnig and the end of the program respectively.
In fact, I could even omit deallocating the memory, the OS would do it for me anyways.
Yet it is not equivalent to malloc'ing everywhere and not freeing the memory because it's harder to manage (like my actual implem).
Here we allocate one buffer, and reuse it for all the iterations.
The memory consumption is constant, let's say MEM_MAX.
The program will be at MEM_MAX from beginning to end.
This obviously constrasts with my current version where the memory is continuously growing (huge memory leak).
So this is way better.

But this solution is not limited to a constant MEM_MAX, which could be problematic when your machine's memory is inferior than that. 
In solution 2. we show that the arena can be resetted between the forward and the backward pass.
This way, the minimum amount of memory is now the max(MEM_MAX_FORWARD, MEM_MAX_BACKWARD).
The memory granualarity can be chosen.
Another solution could also consists of declaring context arenas, e.g. one for the forward, one for the backward, or even lower in granularity.

Oh also the advantage of the arena is that it is then very obvious if a function actually allocate memory or not.
It also forces the developper to think about the lifetime of its object (In which arena do I put this object?).

Sadly, it is harder to do with starpu.
Apparently I can override `starpu_malloc` (and free) with `starpu_malloc_set_hooks` like in this [example](https://gitlab.inria.fr/starpu/starpu/-/blob/master/examples/basic_examples/hooks.c?ref_type=heads).
But it seems the data handles cannot be allocated freely wherever I want.
This is not a big deal in terms of performances because the handles are just the wrapper to the actual data (which is could allocate with arenas).
However it still complexify the data management, I still need to know when a buffer can be freed.
Because at some point I need to call starpu_data_unregister.

In theory I could implement an arena with the global buffer (everything is stored inside) + keeping track of every starpu_handles that have some data
in this buffer.
I could unregister all of that in the arena reset.
So it would imply actual memory dealloction (which is not dramatic for handles).
But, it makes my API more heterogenous, which I don't really like.
It means the GPU allocations will be treated differently, with starpu's care.

Do I need to manage GPU allocations myself? 
I think knowing the memory used be my program is very important in order to optimize it.
Especially for distributing computation on heterogeneous machines: I want to distribute the task in order to
decrease the energy consumption but I need to take into considerations limits of each machines.
This obviously includes the processing unit speed, number, and types (gpu, cpu), the bandwidth,
but also maximum memory of each processing unit (gpu, cpu).
As a side note I think memory managment is also important to reduce for energy consumption.
Also it should lead to better usage of the cache, thus speeding up the execution speed.

[Mon Jun 16 04:24:21 PM CEST 2025]

Arena having been finally implemented.

Concerning previous task, at the moment it seems "ok" to me to let StarPU manage GPU memory.
It's still a pretty homogeneous way to handle memory from the user perspective.

I decided to use an existing arena implemetation and wrap it so I can store the handles that points to data in the arena.
I also included `starpu_memory_pin()` calls in the arena to permit asynchronous transfers.
However we should check later if this really works, because instead of pinning data one by one (which mean I should store they position in the arena), I pin directly the arena buffer.
I hope I will be ok for StarPU. 
I could also do `starpu_malloc()` but it feels like its done for singular data elements (a bit like pin in fact?).

Anyways it works really really well in terms of memory consumption.
The arena implementation I am using can grow, which means it will grow at the first iteration then remain the same for every other.

Surprisingly, it did not improve runtime.

Results with the arena version

```bash
andrew@nixos ~/l/i/d/build (arena) [nix] > time ./example-mnist ../fashion-mnist/train-images-idx3-ubyte ../fashion-mnist/train-labels-idx1-ubyte
[starpu][_starpu_init_topology] Warning: there are several kinds of CPU on this system. For now StarPU assumes all CPU are equal
[starpu][_starpu_initialize_workers_bindid] Warning: hwloc reported 14 logical CPUs for 12 cores, this is not homogeneous, will assume 1 logical CPUs per core
Loaded 60000 images of size 28x28 from ../fashion-mnist/train-images-idx3-ubyte
Loaded 60000 labels from ../fashion-mnist/train-labels-idx1-ubyte
Epoch 0
Average loss: 0.140936 - Accuracy: 52.619999%
Epoch 1
Average loss: 0.087665 - Accuracy: 68.540001%
Epoch 2
Average loss: 0.077776 - Accuracy: 72.339996%
Epoch 3
Average loss: 0.071970 - Accuracy: 74.299995%
Epoch 4
Average loss: 0.067910 - Accuracy: 75.919998%

________________________________________________________
Executed in  156.48 secs    fish           external
   usr time  585.66 secs    0.97 millis  585.66 secs
   sys time  364.38 secs    1.75 millis  364.38 secs
```

With the malloc version
```bash
[starpu][_starpu_init_topology] Warning: there are several kinds of CPU on this system. For now StarPU assumes all CPU are equal
[starpu][_starpu_initialize_workers_bindid] Warning: hwloc reported 14 logical CPUs for 12 cores, this is not homogeneous, will assume 1 logical CPUs per core
Loaded 60000 images of size 28x28 from ../fashion-mnist/train-images-idx3-ubyte
Loaded 60000 labels from ../fashion-mnist/train-labels-idx1-ubyte
Epoch 0
Average loss: 0.140936 - Accuracy: 52.619999%
Epoch 1
Average loss: 0.087665 - Accuracy: 68.540001%
Epoch 2
Average loss: 0.077776 - Accuracy: 72.339996%
Epoch 3
Average loss: 0.071970 - Accuracy: 74.299995%
Epoch 4
Average loss: 0.067910 - Accuracy: 75.919998%

________________________________________________________
Executed in  160.40 secs    fish           external
   usr time  544.30 secs    0.00 millis  544.30 secs
   sys time  284.45 secs    1.74 millis  284.45 secs
```

Reproducing these results:

Arena version commit `3251a3bfd03b2a954b30787c853a5135e88abfa9`.
Malloc version commit `2f85eab08278250346d29a47da9c70f091c4b3fd`.

Make sure that the fashion mnist dataset have been downloaded.

```bash
git checkout <commit>
nix develop
mkdir build && cd build && cmake .. && make
time ./example-mnist ../fashion-mnist/train-images-idx3-ubyte ../fashion-mnist/train-labels-idx1-ubyte
```
