- Make a wrapper for starpu matrix/block to provide nice accesors such as get(x, y, z), maybe we can even hide data acquiring under the hood, it can be nice.
    => This is trickier than it seems:
    First solution is to do a function, it works very well and can even check with asserts if the index is oob, however it means that in CUDA
    I should be able to import this function... Which may be possible?
    ```c
    dahl_fp block_get(starpu_data_handle_t handle, size_t x, size_t y, size_t z)
    {
        dahl_fp* block = (dahl_fp*)starpu_block_get_local_ptr(handle);
        size_t ldy = starpu_block_get_local_ldy(handle);
        size_t ldz = starpu_block_get_local_ldz(handle);

        size_t index = (z*ldz)+(y*ldy)+x;

        // TODO: add debug flags macros
        size_t nx = starpu_block_get_nx(handle);
        size_t ny = starpu_block_get_ny(handle);
        size_t nz = starpu_block_get_nz(handle);

        assert(index < nx * ny * nz);

        return block[index];
    }
    ```

    With macros we wouldn't have the problem however this looks very bad and its harder to get an assert in there
    ```c
    #define block_get(p, x, y, z, ldy, ldz) (p[((z * ldz) + (y * ldy) + x)])

    #define block_get(handle, x, y, z) (\
        ((dahl_fp*)starpu_block_get_local_ptr(handle))\
            [ (z * starpu_block_get_local_ldz(handle))\
            + (y * starpu_block_get_local_ldy(handle))\
            +  x]\
    )
    ```
- lun. 17 mars 2025 11:48:23 CET -> With my new wrapper (dahl_matrix, dahl_block) this is handled, however we still need to separate matrix and block functions in the codelets functions, even if the implementation could be the same. I'm not sure this is a problem though.
  -> everything can be registered as blocks under the hood?
  -> seems like a good idea, so we can route corresponding functions e.g.:
  ```
  task_relu_block(dahl_block) -> calls cl_relu_block()
  task_relu_matrix(dahl_matrix) -> calls cl_relu_block() // Same because dahl_matrix is under the hood a block and also because relu implementation is looping through all elements one by one so the dimensions does not matter
  ```
  And probably that even for add functions, block and matrix functions could be the same in fact?

- Make dahl_matrix / dahl_block generic? they could use the type we want? -> seems hard, need to think about it.
- less important but print always the same numbers of character in pretty print e.g. "42.00", " 8.00"...
- Should filter values be negative?
- is `type const* const` really useful? typically when defining a parameter, obviously the pointer is const and won't be changed no? idk
