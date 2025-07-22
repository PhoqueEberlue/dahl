# Todo

## Important

- fix layers unit tests
  - hard code bias and weights values to prevent rng problems
- fix cifar-10 training

- Try to do shuffling to fix the accuracy?
- benchmark with different task granularity?
- benchmark with different batch sizes?
- Implementing the 3 other parallelizable dimensions
- Implement PipeDream algorithm
- One problem with the current acquire is that  it cannot be used if the actual data is partitionned. 
  However it does work if we require a read only data.

## Later

- Make dahl_temporary_arena and persistent arena pointer const. This requires being able to instanciate the arenas on the stack.
  -> I think I won't be possible because I need (or want) to hide the arena implementation behind an opaque type.
- Replace the context arena mechanism by an explicit-argument-passing implement: this will make function signature indicate if the function
  actually allocates data

- Investigate why filling arenas buffer in block/matrix/vector works and not with a memset directly inside the arena_alloc function
- propagate precision passed in the asserts to the block/matirx/vector prints
- Add compilation condition to enable/disable debugg asserts
- change the strings of the STARPU_CHECK_RETURN_VALUE()
- cmake if macos ignore cuda
- Investigate on that: if I partition a starpu block, and send tasks with the sub blocks on the GPU, does StarPU copies the whole block on the GPU, or
  every single sub block one by one?
- My API let the opportunity to acquire a data multiple times which leads to deadlocks and could be hard to debug,
  maybe we should hide the data acquire and instead give accessor to fill the data. See [hiding-acquire-release](./design-talk/topics/hiding-acquire-release.md)
- the padding function is very unefficient
- We might still have a neglibeable memory leak even after the arena introduction.
  -> it seems that `starpu_codelet_pack_args` causes memory leak
- make the dataset loader more generic

## Questionable

- Add dahl prefix to every public functions/macros -> questionable decision?
- less important but print always the same numbers of character in pretty print e.g. "42.00", " 8.00"... -> can be made easy with scientific notation
- use starpu redux system?
