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
    void func_name(void* buffers[num_buffers], void* cl_arg);                      \
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
DEFINE_STARPU_CODELET(matrix_matrix_product, 3, STARPU_R, STARPU_R, STARPU_W);

DEFINE_STARPU_CODELET(block_relu, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_relu, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_relu, 2, STARPU_R, STARPU_W);

DEFINE_STARPU_CODELET(block_sum_z_axis, 2, STARPU_R, STARPU_W);

DEFINE_STARPU_CODELET(block_scal, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_scal, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_scal, 2, STARPU_R, STARPU_W);

DEFINE_STARPU_CODELET(block_sub, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_sub, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_sub, 3, STARPU_R, STARPU_R, STARPU_W);

DEFINE_STARPU_CODELET(block_add, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_add, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_add, 3, STARPU_R, STARPU_R, STARPU_W);

DEFINE_STARPU_CODELET(vector_softmax, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_dot_product, 2, STARPU_R, STARPU_R);
DEFINE_STARPU_CODELET(vector_diag, 2, STARPU_R, STARPU_W);

DEFINE_STARPU_CODELET(block_add_value, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_add_value, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_add_value, 2, STARPU_R, STARPU_W);

DEFINE_STARPU_CODELET(block_sub_value, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_sub_value, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_sub_value, 2, STARPU_R, STARPU_W);

DEFINE_STARPU_CODELET(matrix_vector_product, 3, STARPU_R, STARPU_R, STARPU_W);

DEFINE_STARPU_CODELET(block_clip, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_clip, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_clip, 2, STARPU_R, STARPU_W);

DEFINE_STARPU_CODELET(vector_cross_entropy_loss, 2, STARPU_R, STARPU_R);
DEFINE_STARPU_CODELET(vector_cross_entropy_loss_gradient, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_transpose, 2, STARPU_R, STARPU_W);


#endif //!DAHL_CODELETS_H
