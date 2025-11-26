# Todo

## Absolutely required for publication

### Track #1:

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
    -> See if we can do it in time. Otherwise its fine.
  - write a clean and reproducible experimenting environment
    use scripts to launch experiments, ensure git commit version, nix env etc.

Content of the article:
- Introduce the problematic of heterogeneity
- TGCS in general, StarPU, the codelets systems
- Using Nix for reproducible development/building
- Unit testing following Pytorch results
- Memory management with Arenas: explain theory, pseudo implementation, show graphs of memory usage,
  emphasize how cleaner it makes the API.
  -> probably should include benchmarks on that, compare the Arena to a simple Malloc/Free.
- talk about `()`, `_self()`, and `_init()`:  the advantage of reusing buffers, using masks...
- optimizations to remove padding but still use "valid" convolution
  -> Not so sure, it seems that the operation could already be done without padding, in a
  mathematically equivalent way.
- Some preliminary results

## Important

> [!CAUTION]
> MANDATORY IN THE NEXT WEEKS:
> - Fix loss? It seems that it is pretty different than pytorch's loss when iterating over multiple
>   epochs. Maybe I confused batch size and num classes somewhere in the code?
> - There's somehow a memory leak? Somewhere...
> 
>                                   - Really need to do that -
> - Find a easy way to get the average execution time of a codelet and document it (probably in a 
>   file with a bunch of "useful debugging, and performance anylizing commands").
>   This should be easy to do with starpu builtin commands.

- Add a parameter to choose how much images we load in a dataset

- we probably need to use `REACTIVATE_PARTITION` on objects stored in the network_arena, cause it
  may risks of allocating more memory over and over otherwise. Or simply never unpartition those
  ones, for example the masks in relu/pooling don't need to be unpartitioned each time.
- Investigate why a codelet can have various execution time: different input size? different work
  load?
  - Why there is a huge convolution on GPU? -> try execution a lot of convolution in tests

- Why does the scheduler sometimes struggles to parallelize tasks when using big batch sizes?
- Maybe some layers have a better performance on a given processing unit?
- tensor sum xyt: why is it slower on gpu than in cpu?
- Improve const correctness with the new typestate partitioning system
- standardize error messages? -> At least be able to print values, the line where it crashed, etc.

## Later

- Implement PipeDream algorithm

- Functions with underscore prefix are supposed to be reserved for compiler and std.
- Add dahl prefix to every public functions/macros -> Will be required at some point, though we may
  reduce prefix to `dl` instead of `dahl`.

- Benchmarks comparing different float types, could be interesting but also requires more
  state-of-the-art to see what added value I can bring. Also we'll need to use _Generic cause some
  functions we're using might be limited to double, the actual type of dahl_fp.
- Add __restrict__ for functions that does not implement _self mode
  -> in fact we will be able to add it in a lot of places. Altough, we might check where it makes
  sense to do so.
- Remove diff.py and simply implement that in C so we have the exact digit by digit comparison,
  because right now it's more a char to char comparison, which doesn't work anymore when one value
  is shifted, e.g.: 12.001 vs 1.001. Honestly doing that in C would be "fun" and not particularly
  complicated.
- Investigate on that: if I partition a starpu block, and send tasks with the sub blocks on the GPU,
  does StarPU copies the whole block on the GPU, or every single sub block one by one?
- My API let the opportunity to acquire a data multiple times which leads to deadlocks and could be
  hard to debug, maybe we should hide the data acquire and instead give accessor to fill the data. 
  See [hiding-acquire-release](./design-talk/topics/hiding-acquire-release.md)
- We might still have a neglibeable memory leak even after the arena introduction.
  -> it seems that `starpu_codelet_pack_args` causes memory leak
- it seems that `*_get_value()` functions are superrrrrrrrrrrrr slow. What do we do about that?
  -> we could propose two ways of accessing values: the existing one, and another one that uses a
  macro to avoid function calls?

- cmake if macos ignore cuda
