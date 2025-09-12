# Todo

## Important

- Unify data access in data types? For now I access elements with `[(ld * ny) + nx]` which is error prone, especially when we have many dimensions.
  ~Also my equals and print functions does not take into account `ld`s which produce different result than starpu codelets.~
  -> this should be fixed, however some function may still miss ld, beware.
  > [!WARNING]
  > This behavior is not fixed in the equals functions -> refactor that
  I think giving access to elements via a function would be too slow, so we could probably create a macro?
  -> my opinion right now is: we let the `[(ld * ny) + nx]` access inside the codelets, but for blocking functions (that are not performance oriented) we can
  use a macro or a function. So the API user doesn't make any mistake, and the developper of codelets should be aware of `ld`s anyway.
  It should probably be a function because we have to call starpu to get dimensions and padding anyways.

- propagate precision passed in the asserts to the block/matirx/vector prints
  - Optionnaly create a global parameter to manage floating point display
  - in the asserts fix the precision displaying, for example if I ask a precision of 15 and actually
    display 15 digit after comma, the last digit could be round up to the same number, meaning
    that the difference won't be noticable in the diff display.
- for random functions, provide a min/max argument instead of hard coding the values

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
  -> I think it won't be possible because I need (or want) to hide the arena implementation behind an opaque type.

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
