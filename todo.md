# Todo

## Absolutely required for publication

### Track #1:

TODO:
- Implement the GPU version
  -> we probably ignore BLAS/cuBLAS for now, will take too much time
- Use multiple datasets;
  - What nice with we already have (CIFAR-10 and Fashion-Mnist) is that we can compare
    parallelization between a CNN that deals with multi-channel images and black and white images.
    This can be great if I want to showcase a simple example of parallelization, and/or show the
    differences when using more channels.
  - In any ways, we will need one or more datasets with bigger images to show the impact of task
    granularity on the scheduling.
- Make sure to always compare with Pytorch, and measure energy consumption on both.
  -> I probably also want to get the %average (or curve) of the CPU/GPU usage.
- We should probably try at least two different architectures of CNN?
  -> get inspiration from CNNExplainer

Content of the article:
- Introduce the problematic of heterogeneity
- TGCS in general, StarPU
- Task granularity tradeoffs
- Parallelizing
- Benchmarks comparison with Pytorch, comparing energy consumption
- Do we have an advantage using GPU + CPU?

### Track #2:

TODO:
- Make the tool very clean
  - add some documentation
  - add some examples
- provide even more reproducible features
  - My current Pytorch-result-driven unit tests do the job, but I think they might not be perfectly
    rigorous, I should probably implement better floating point testing (I don't remember how its
    called).
  - implement a clean and reproducible environment
    -> an idea would be to improve my existing workflow in notes/lab
    create scripts to re-launch the experiments?
    inspire from qanat?
    At least we could do something (nix script?) that does bundle the full entry parameters (source
    code, ref of the datasets, command and its argument) and store the results (cmd out, traces etc)

Content of the article:
- Introduce the problematic of heterogeneity
- TGCS in general, StarPU, the codelets systems
- Using Nix for reproducible development/building
- Unit testing following Pytorch results
- Memory management with Arenas: explain theory, pseudo implementation, show graphs of memory usage,
  emphasize how cleaner it makes the API.
~~- Maybe talk about data structure traits/_Generic, although I'm not sure its "scientifically interesting."~~
- talk about `()`, `_self()`, and `_init()`:  the advantage of reusing buffers, using masks...
- optimizations to remove padding but still use "valid" convolution
- Some preliminary results

## Important

> [!CAUTION]
> MANDATORY IN THE NEXT WEEKS:
> - Fix loss? It seems that it is pretty different than pytorch's loss when iterating over multiple epochs
> - Implement a new layer for relu, so that we can store the previous mask of negative values index,
>   so that it can be reused during the backward.
> - There's somehow a memory leak? Somewhere...
> - Implement and use datasets with bigger images
> 
>                                   - Really need to do that -
> - Find a easy way to get the average execution time of a codelet and document it (probably in a file
>   with a bunch of "useful debugging, and performance anylizing commands").
>   This should be easy to do with starpu builtin commands.

- Investigate why a codelet can have various execution time: different input size? different work
  load?
- Why does the scheduler sometimes struggles to parallelize tasks when using big batch sizes?

## Refactors

- make relu a layer in itself, same for flatten layer. Could pontentially refactor how we
  write/assemble layers in the user's code.
- refactor similar codelets (any_scal, sub, power etc) to use an inlined function?
- Add __restrict__ for functions that does not implement _self mode
  -> in fact we will be able to add it in a lot of places. Altough, we might check where it makes
  sense to do so.
- Remove diff.py and simply implement that in C so we have the exact digit by digit comparison, because right now it's more a char to char comparison,
  which doesn't work anymore when one value is shifted, e.g.: 12.001 vs 1.001

## Later

- Implement PipeDream algorithm, and optionnaly implement the 3 other parallelizable dimensions
- standardize error messages? -> At least be able to print values, the line where it crashed, etc
- refactor partitionning functions with an Axis argument?
- separate cleanly private functions in dahl data structure files
- Add compilation condition to enable/disable debugg asserts
- Investigate on that: if I partition a starpu block, and send tasks with the sub blocks on the GPU, does StarPU copies the whole block on the GPU, or
  every single sub block one by one?
- My API let the opportunity to acquire a data multiple times which leads to deadlocks and could be hard to debug,
  maybe we should hide the data acquire and instead give accessor to fill the data. See [hiding-acquire-release](./design-talk/topics/hiding-acquire-release.md)
- the padding function is very unefficient
- We might still have a neglibeable memory leak even after the arena introduction.
  -> it seems that `starpu_codelet_pack_args` causes memory leak
- it seems that `*_get_value()` functions are superrrrrrrrrrrrr slow. What do we do about that?
  -> we could propose two ways of accessing values: the existing one, and another one that uses a
  macro to avoid function calls?

## Clearly not important

- Add dahl prefix to every public functions/macros -> questionable decision? Honestly I think this is not needed, 
  we already have the prefix for the types, which is sufficient I guess. The only place where it is mandatory is for
  the arena, because we need to differentiate our arena wrapper with tsoding's arena.
- use starpu redux system?
- Investigate why filling arenas buffer in block/matrix/vector works and not with a memset directly inside the arena_alloc function
- cmake if macos ignore cuda
