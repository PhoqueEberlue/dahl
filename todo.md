# Todo

## Important

- Fix mini-batch: we should not update weights per sample but rather accumulate dl_dw and dl_dy to update the weights once.
  Also note that we could simply fill dl_dw_batch then do tensor_sum_axis_t but:
  - the advantage of doing that is that we can fully go parallel because the buffers are not shared as we use sub partitions
    BUT this makes us store those partial result and leads to useless memory access
  - instead we could leverage starpu redux system, however I wonder how it deals with parallelism? Especially if the whole batch
    does try to increment the same buffer at the same time
    -> Should try this in a separate project

- What I can do is defining multiple implementations for a single codelet, for example I could keep my actual version of the cross correlation,
  but I could also write a vectorized version. This way I keep the naive, and perf oriented implem, and StarPU will be able to chose between the two.
- Think about implementing a mini batch mechanism
  -> need to add a batch argument to my conv, pooling and dense layers
  -> it implies adding a 4th data structure, so we should refactor cleanly the codelet api to reuse cleanly the basic functions
- benchmark with different task granularity?
- benchmark with different batch sizes?
- Implementing the 3 other parallelizable dimensions
- Implement PipeDream algorithm
- One problem with the current acquire is that  it cannot be used if the actual data is partitionned. 
  However it does work if we require a read only data.

## Later

- Make dahl_temporary_arena and persistent arena pointer const. This requires being able to instanciate the arenas on the stack.
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
- make the dataset loader more generic

## Questionable

- Add dahl prefix to every public functions/macros -> questionable decision?
- less important but print always the same numbers of character in pretty print e.g. "42.00", " 8.00"... -> can be made easy with scientific notation
