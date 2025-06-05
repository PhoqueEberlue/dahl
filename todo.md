# Todo

## Important

- Homogeneize `x_to_y`, `x_as_y` etc functions. Maybe actually add a view functionnality?
- check memory leaks, don't forget to call the finalize() at some point :) -> Arena?
- make the partition function work with const data?? -> obviously lead to problems bc we need to create the views and change some values to toggle the partition boolean.

## Later

- propagate precision passed in the asserts to the block/matirx/vector prints
- Add compilation condition to enable/disable debugg asserts
- change the strings of the STARPU_CHECK_RETURN_VALUE()
- cmake if macos ignore cuda
- Investigate on that: if I partition a starpu block, and send tasks with the sub blocks on the GPU, does StarPU copies the whole block on the GPU, or
  every single sub block one by one?

## Questionable

- Add dahl prefix to every public functions/macros -> questionable decision?
- less important but print always the same numbers of character in pretty print e.g. "42.00", " 8.00"... -> can be made easy with scientific notation
