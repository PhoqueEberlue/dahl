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

- less important but print always the same numbers of character in pretty print e.g. "42.00", " 8.00"...
- Should filter values be negative?
