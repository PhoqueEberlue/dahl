-------------------------------------------------------------------------------
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
    => THIS IS WHAT HAVE BEEN CHOSEN
-------------------------------------------------------------------------------
- Make dahl_matrix / dahl_block generic? they could use the type we want? -> seems hard, need to think about it.
    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #define N 500000000

    enum type {
        type_int,
        type_double,
    };

    void* int_get(void* array, size_t index)
    {
        return (void*) &( (int*)array )[index];
    }

    void int_add(void* a, void* b)
    {
        *(int*)a += *(int*)b;
    }

    int main()
    {
        int* array = malloc(N * sizeof(int));

        for (size_t i = 0; i < N; i++)
        {
            array[i] = 42;
        }

        void* array_v = (void*)array;
        enum type ty = type_int;
        
        // Solution 1: no genericity | avg: 1.50sc runtime
        for (size_t i = 0; i < N - 1; i++)
        {
            array[i+1] += array[i];
        }

        // Solution 2: function pointers | avg: 2.63sc runtime
        // Here we can do those assignements in if/else to chose the correct type
        if (ty == type_int)
        {
            void* (*get)(void*, size_t) = &int_get;
            void (*add)(void*, void*) = &int_add;
        }

        for (size_t i = 0; i < N - 1; i++)
        {
            void* val_a = get(array_v, i);
            void* val_b = get(array_v, i+1);

            int_add(val_b, val_a);
        }

        // Solution 3: enum and if else in the loop | avg: 1.62sc runtime
        // It seems that the branch predictor is doing a good job, yet it's still bellow the optimal.
        for (size_t i = 0; i < N - 1; i++)
        {
            if (ty == type_int)
            {
                ((int*)array_v)[i+1] += ((int*)array_v)[i];
            }
            else if (ty == type_double)
            {
                ((double*)array_v)[i+1] += ((double*)array_v)[i];
            }
        }
    }
    ```
    Solution 1 is the actual one, this is the fastest because only one type can be used at runtime, so everything is perfectly tailor-made.
    Solution 2 keeps the code in a readable state, however the performance for calling the functions, passing by address and doing the good casts
    does severely degrade the performance.
    Solution 3 makes the code awful but it is performing almost as great as solution 1, probably thanks to branch prediction.
    Solution 4 involves using C++ templates, however it is still mandatory to link everything together by hand for each codelet for each type... 
    Nntile does that, the code is indeed less readable.
    -> lun. 24 mars 2025 14:57:44 CET, for now, the use case of generic types is just to store indexes into a block. Instead we could create a data
    structure "view" which makes sense to have it's own implementation?

-------------------------------------------------------------------------------
- less important but print always the same numbers of character in pretty print e.g. "42.00", " 8.00"...
- Should filter values be negative?
- is `type const* const` really useful? typically when defining a parameter, obviously the pointer is const and won't be changed no? idk
    => Yes it is!
