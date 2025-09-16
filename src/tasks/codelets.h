#ifndef DAHL_CODELETS_H
#define DAHL_CODELETS_H

#include "../../include/dahl_tasks.h"
#include "../data_structures/data_structures.h"
#include <starpu.h>

// Helper macro to generate the three essentials codelets definitions:
// - The actual codelet function signature which is always the same, `void(void**, void*)`
//   but here we also specify the number of buffers
// - starpu_perfmodel
// - starpu_codelet, referencing the function, number of buffers, their access modes and
//   the perfmodel
#define DEFINE_STARPU_CODELET(func_name, num_buffers, ...)    \
    void func_name(void* buffers[num_buffers], void* cl_arg); \
                                                              \
    static struct starpu_perfmodel perf_model_##func_name = { \
        .type = STARPU_HISTORY_BASED,                         \
        .symbol = #func_name                                  \
    };                                                        \
                                                              \
    static struct starpu_codelet cl_##func_name = {           \
        .cpu_funcs = { func_name },                           \
        .nbuffers = num_buffers,                              \
        .modes = { __VA_ARGS__ },                             \
        .model = &perf_model_##func_name                      \
    };

// Utility codelet to switch/refresh/synchronize buffers.
// See manual partitioning: design-talk/topics/data-structure-wrappers.md#getting-the-right-types-with-manual-partitionning
static struct starpu_codelet cl_switch =
{
	.where = STARPU_NOWHERE,
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
};

// ---------------------------------------- TENSOR ----------------------------------------
DEFINE_STARPU_CODELET(tensor_sum_t_axis, 2, STARPU_R, STARPU_W);

// ---------------------------------------- BLOCK ----------------------------------------
DEFINE_STARPU_CODELET(block_sum_z_axis, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(block_sum_y_axis, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(block_sum_xy_axes, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(block_add_padding, 2, STARPU_R, STARPU_W);

// ---------------------------------------- MATRIX ----------------------------------------
DEFINE_STARPU_CODELET(matrix_cross_correlation, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_max_pooling, 3, STARPU_R, STARPU_W, STARPU_W);
DEFINE_STARPU_CODELET(matrix_backward_max_pooling, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_matrix_product, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_sum_y_axis, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_vector_product, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_transpose, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(matrix_resize, 1, STARPU_W);
DEFINE_STARPU_CODELET(matrix_rotate_180, 2, STARPU_R, STARPU_W);

// ---------------------------------------- VECTOR ----------------------------------------
DEFINE_STARPU_CODELET(vector_softmax, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_dot_product, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_diag, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_to_matrix, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_outer_product, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(vector_shuffle, 1, STARPU_RW);

// ---------------------------------------- ANY ----------------------------------------
// Codelets that can be used with any type
DEFINE_STARPU_CODELET(relu, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(relu_backward, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(scal, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(power, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(sub, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(add, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(add_value, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(clip, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(sum, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(mean, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(fill, 1, STARPU_W);
DEFINE_STARPU_CODELET(wait, 1, STARPU_W);
DEFINE_STARPU_CODELET(copy, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(min, 2, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(max, 2, STARPU_R, STARPU_W);

// ---------------------------------------- ML Related ----------------------------------------
DEFINE_STARPU_CODELET(check_predictions_batch, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(cross_entropy_loss_batch, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(cross_entropy_loss_gradient, 3, STARPU_R, STARPU_R, STARPU_W);
DEFINE_STARPU_CODELET(convolution_2d, 3, STARPU_R, STARPU_R, STARPU_W);

#endif //!DAHL_CODELETS_H
