#ifndef DAHL_CODELETS_H
#define DAHL_CODELETS_H

#include <starpu.h>

// Helper macro to generate the three essentials codelets definitions:
// - The actual codelet function signature which is always the same, `void(void**, void*)`
//   but here we also specify the number of buffers
// - starpu_perfmodel
// - starpu_codelet, referencing the function, number of buffers, their access modes and
//   the perfmodel
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
    };

DEFINE_STARPU_CODELET(matrix_cross_correlation, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_max_pooling, 3, STARPU_R, STARPU_W, STARPU_W);
DEFINE_STARPU_CODELET(matrix_backward_max_pooling, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(relu, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(block_sum_z_axis, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(scal, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(sub, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(add, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_softmax, 2, STARPU_R, STARPU_W);

#endif //!DAHL_CODELETS_H
