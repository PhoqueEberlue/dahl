# Todo

## Important

- refactor partitionning functions with an Axis argument?
- separate cleanly private functions in dahl data structure files
- standardize error messages? -> At least be able to print values, the line where it crashed, etc


- benchmark with different task granularity?
- implement a clean and reproducible environment
- benchmark with different batch sizes?
- Implementing the 3 other parallelizable dimensions
- Implement PipeDream algorithm

## Later

- Add compilation condition to enable/disable debugg asserts
- change the strings of the STARPU_CHECK_RETURN_VALUE()
- Investigate on that: if I partition a starpu block, and send tasks with the sub blocks on the GPU, does StarPU copies the whole block on the GPU, or
  every single sub block one by one?
- My API let the opportunity to acquire a data multiple times which leads to deadlocks and could be hard to debug,
  maybe we should hide the data acquire and instead give accessor to fill the data. See [hiding-acquire-release](./design-talk/topics/hiding-acquire-release.md)
- the padding function is very unefficient
- We might still have a neglibeable memory leak even after the arena introduction.
  -> it seems that `starpu_codelet_pack_args` causes memory leak

## Clearly not important

- Add dahl prefix to every public functions/macros -> questionable decision? Honestly I think this is not needed, 
  we already have the prefix for the types, which is sufficient I guess. The only place where it is mandatory is for
  the arena, because we need to differentiate our arena wrapper with tsoding's arena.
- less important but print always the same numbers of character in pretty print e.g. "42.00", " 8.00"... -> can be made easy with scientific notation
- use starpu redux system?
- Investigate why filling arenas buffer in block/matrix/vector works and not with a memset directly inside the arena_alloc function
- cmake if macos ignore cuda
