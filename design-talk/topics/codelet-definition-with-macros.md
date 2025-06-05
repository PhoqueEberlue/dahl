# Codelet definition with macros

[ven. 28 mars 2025 08:26:17 CET]
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


//--- Used to generate a call function, the solution was abandoned ---
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
//--------------------------------------------------------------------

DEFINE_STARPU_CODELET(matrix_cross_correlation, 3, STARPU_R, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(matrix_max_pooling, 3, STARPU_R, STARPU_W, STARPU_W)
DEFINE_STARPU_CODELET(matrix_backward_max_pooling, 3, STARPU_R, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(relu, 2, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(block_sum_z_axis, 2, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(scal, 2, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(sub, 3, STARPU_R, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(add, 3, STARPU_R, STARPU_R, STARPU_W)
```

Problem: for each codelet function, I need to write a codelet struct, a perfmodel struct and the signature of the function.
This can be easly generated using a macro.

[jeu. 24 avril 2025 17:12:02 CEST]
I also tried to do a wrapper function that actually makes the starpu task insert of the codelet but this is way too much because
we need to handle multiple arguments for the buffers and it becomes messy real quick.
Also in the current state of dahl, many of those functions hide some specific work in this wrapper function, e.g. if I do a matrix matrix
multiplication, my wrapper can automatically instanciate an output matrix with the right dimensions.


