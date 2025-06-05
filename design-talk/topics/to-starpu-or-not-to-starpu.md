# To StarPU or not to StarPU

The last point discuss solutions where starpu would be left to give me more control over the memory operations.

This ask a real debate for the further direction of the thesis, first here are my first thought about pros and cons of leaving starpu:

What do I lose from leaving starpu?
- Asynchronous handling of tasks for threads and gpu executions (which is an enormous point -> ⚠️ mutex/semaphore ⚠️)
- automatic perf models (which can be reimplemented pretty quickly I would say)
- Auto-managment of memory between ram and GPU integrated memory (but the point is to manage it ourselves)
- More work on developping with CUDA probably, however I think I still had to do most of stuff (other than  memory managment) with starpu.
- Also data access and concurrency between threads!!!

What do I gain?
- I develop my own scheduler: I want to create an optimization software that takes a ML Model, a platform and return the theoritical optimum 
  for training this model with the minimum energy consumption (partition). With starpu the task scheduling will be done at runtime, this can be great
  but it will hurt reproducibility (I'm not sure I can simulate it), and I won't be able to execute my partition like the optimization software told us 
  to do. *But maybe this is a bad idea to do that from the beginning?*
- the code becomes more "deterministic" in terms of execution flow, which will lead to an easier time with simulating the runtime. (We could even try to separate the "execution flow" (So the ML API wrapper in fact?) from the implem, network and executions to integrate it in Simgrid)
- full control of GPU allocations, I can take into account the GPU allocation in the optimizing software
- I control my data structures myself, my three wrappers (block, matrix, vector) are no longer all pointing to a `starpu_block` just because the sub data of a block is a block. 
  I can specialize and decide how the data is stored. I can take every optimization in terms of data storing and use my sturctures cleanly
  in the codelets for cpu or CUDA.

=> In fact StarPU is colliding with what I had in mind for the thesis.
I want to make a software that statically optimizes a distributed workload and planify the best execution order.
However starpu is a runtime scheduler, the execution will always be different because it will adapt dynamically.

[mar. 06 mai 2025 11:35:09 CEST] Update, hybrid aproach?

We could chose an in-between, static workload/data distribution inter-node, and runtime scheduled tasks for intra-node.
In fact for inter-node distribution, we would have an idea of the computation power for each node (+ the data transfer cost/time) and based on that
we would split the workloads so that for every synchronization point, everyone involved "almost" finishes at the same time.

From the intra-node point of view, it would receive a partition (ordered tasks graph to accomplish, computation or data sending) and based on that the scheduler tries to do his best
to execute it fast. Or we could also imagine using an energy-based scheduler.

One problem is that we could estimate that a machine will perform some task very fast thanks to the GPU, however the scheduler could decide not to use it, leading to huge slow downs.
Here the gap between estimations and reality could make our partition very bad. 
However we could have similar problems with a "fully static" approach, if we meticulously assign a task to a core/gpu in particular we could end up having problems or simply under-use our nodes.

[Thu Jun  5 09:29:17 PM CEST 2025]
See my [drawio graphs](../illustrating-graph-formalism.drawio) for further talks, globally I is perfectly correct to continue using StarPU, we keep on sailing.
