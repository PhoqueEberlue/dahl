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
    - Solution 1 is the actual one, this is the fastest because only one type can be used at runtime, so everything is perfectly tailor-made.
    - Solution 2 keeps the code in a readable state, however the performance for calling the functions, passing by address and doing the good casts
    does severely degrade the performance.
    - Solution 3 makes the code awful but it is performing almost as great as solution 1, probably thanks to branch prediction.
    - Solution 4 involves using C++ templates, however it is still mandatory to link everything together by hand for each codelet for each type... 
    Nntile does that, the code is indeed less readable.

    -> lun. 24 mars 2025 14:57:44 CET, for now, the use case of generic types is just to store indexes into a block. Instead we could create a data
    structure "view" which makes sense to have it's own implementation?

-------------------------------------------------------------------------------
- ven. 28 mars 2025 08:26:17 CET
    ~ Macro nightmare ~
    ```c
    #define DEFINE_STARPU_CODELET(func_name, num_buffers, ...)                         \
        void func_name(void *buffers[num_buffers], void *cl_arg);                      \
                                                                                       \
        static struct starpu_perfmodel perf_model_##func_name = {                      \
            .type = STARPU_HISTORY_BASED,                                              \
            .symbol = "perf_model_" #func_name                                         \
        };                                                                             \
                                                                                       \
        static struct starpu_codelet cl_##func_name = {                                \
            .cpu_funcs = { func_name },                                                \
            .nbuffers = num_buffers,                                                   \
            .modes = { __VA_ARGS__ },                                                  \
            .model = &perf_model_##func_name                                           \
        };                                                                             \
                                                                                       \
        void call_##func_name(SELECT_ARGS(num_buffers, starpu_data_handle_t))          \
        {                                                                              \
            int ret = starpu_task_insert(&cl_##func_name,                              \
                                         SELECT_ACCESS_HANDLE(num_buffers, __VA_ARGS__)\
                                         0);                                           \
                                                                                       \
            STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");                \
        }


    #define GET_ARG_1(A, ...) A
    #define GET_ARG_2(_1, A, ...) A
    #define GET_ARG_3(_1, _2, A, ...) A
    #define GET_ARG_4(_1, _2, _3, A, ...) A
    #define GET_ARG_5(_1, _2, _3, _4, A, ...) A

    #define ARG(N, T) T a##N
    #define ARG_LIST_1(T) ARG(1, T)
    #define ARG_LIST_2(T) ARG_LIST_1(T), ARG(2, T)
    #define ARG_LIST_3(T) ARG_LIST_2(T), ARG(3, T)
    #define ARG_LIST_4(T) ARG_LIST_3(T), ARG(4, T)
    #define ARG_LIST_5(T) ARG_LIST_4(T), ARG(5, T)
    #define SELECT_ARGS(N, T) ARG_LIST_##N(T)

    #define ACCESS_HANDLE(N, ...) GET_ARG_##N(__VA_ARGS__), a##N,
    #define ACCESS_HANDLE_LIST_1(...) ACCESS_HANDLE(1, __VA_ARGS__)
    #define ACCESS_HANDLE_LIST_2(...) ACCESS_HANDLE_LIST_1(__VA_ARGS__) ACCESS_HANDLE(2, __VA_ARGS__)
    #define ACCESS_HANDLE_LIST_3(...) ACCESS_HANDLE_LIST_2(__VA_ARGS__) ACCESS_HANDLE(3, __VA_ARGS__)
    #define SELECT_ACCESS_HANDLE(N, ...) ACCESS_HANDLE_LIST_##N(__VA_ARGS__)

    DEFINE_STARPU_CODELET(matrix_cross_correlation, 3, STARPU_R, STARPU_R, STARPU_W)
    DEFINE_STARPU_CODELET(matrix_max_pooling, 3, STARPU_R, STARPU_W, STARPU_W)
    DEFINE_STARPU_CODELET(matrix_backward_max_pooling, 3, STARPU_R, STARPU_R, STARPU_W)
    DEFINE_STARPU_CODELET(relu, 2, STARPU_R, STARPU_W)
    DEFINE_STARPU_CODELET(block_sum_z_axis, 2, STARPU_R, STARPU_W)
    DEFINE_STARPU_CODELET(scal, 2, STARPU_R, STARPU_W)
    DEFINE_STARPU_CODELET(sub, 3, STARPU_R, STARPU_R, STARPU_W)
    DEFINE_STARPU_CODELET(add, 3, STARPU_R, STARPU_R, STARPU_W)
    ```
    The first part is actually pretty nice to define perf models, codelets struct and function, but the call_function() is too much. 

-------------------------------------------------------------------------------
- Simplifying the api by providing a common type "any" that would let you pass anything (block, matrix, vector) so that the functions that performs elementary operations could be called by the same function (e.g. ADD is the same implementation for every data types).
    Actualy if I did a simple function to convert a type, let's say block, to the any type, it would complexify the user code by adding a lot of local variables in their code for each "type casting".
    Another solution is to only use any type as a stack allocated object, thus it can be passed directly by value, so we can call `operator(block_as_any())` where operator takes a any.
    However it again adds functions with long names (block_as_any, vector_as_any etc.), and if an operator has a lot of parameters it can become a mess.
    Instead we can use macros to handle that, as type is knowned at compilation anyways, giving us more syntax sugar
    ```c
    struct a
    {
        int i;
    };

    struct b
    {
        int i;
    };

    struct any
    {
        union data
        {
            struct a* a;
            struct b* b;
        } data;

        enum type
        {
            type_a,
            type_b,
        } type;
    };

    #define AS_ANY(x) _Generic((x),              \
        struct a*:                               \
            (struct any)                         \
            {                                    \
                .data = { .a = (struct a*)(x) }, \
                .type = type_a                   \
            },                                   \
        struct b*:                               \
            (struct any)                         \
            {                                    \
                .data = { .b = (struct b*)(x) }, \
                .type = type_b                   \
            }                                    \
    )

    void fn(struct any an)
    {
        // actually checks with value any is
    }

    int main()
    {
        struct a a = { .i = 42 };
        struct b b = { .i = 12 };

        fn(AS_ANY(&a));
    }
    ```

-------------------------------------------------------------------------------
- less important but print always the same numbers of character in pretty print e.g. "42.00", " 8.00"...
- Should filter values be negative?
- is `type const* const` really useful? typically when defining a parameter, obviously the pointer is const and won't be changed no? idk
    => Yes it is!
- Check discarded const
- test backward pass
