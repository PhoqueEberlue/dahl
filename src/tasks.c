#include "codelets.h"
#include "starpu_data.h"

// Including data.h and not include/dahl_data.h so we have access to the private functions
#include "data.h"

void task_matrix_cross_correlation(dahl_matrix const* const in, dahl_matrix const* const kernel, dahl_matrix* const out)
{
    int ret = starpu_task_insert(&cl_matrix_cross_correlation,
                                 STARPU_R, matrix_get_handle(in),
                                 STARPU_R, matrix_get_handle(kernel),
                                 STARPU_W, matrix_get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_matrix_max_pooling(dahl_matrix const* const in, dahl_matrix* const out, dahl_matrix* const mask, size_t const pool_size)
{
    int ret = starpu_task_insert(&cl_matrix_max_dahl_pooling,
                             STARPU_VALUE, &pool_size, sizeof(&pool_size),
                             STARPU_R, matrix_get_handle(in),
                             STARPU_W, matrix_get_handle(out), 
                             STARPU_W, matrix_get_handle(mask), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_matrix_backward_max_pooling(dahl_matrix const* const in, dahl_matrix const* const mask, dahl_matrix* const out, size_t const pool_size)
{
    int ret = starpu_task_insert(&cl_matrix_backward_max_dahl_pooling,
                             STARPU_VALUE, &pool_size, sizeof(&pool_size),
                             STARPU_R, matrix_get_handle(in),
                             STARPU_R, matrix_get_handle(mask), 
                             STARPU_W, matrix_get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_matrix_backward_max_pooling_self(dahl_matrix const* const in, dahl_matrix* const mask_self, size_t const pool_size)
{
    task_matrix_backward_max_pooling(in, mask_self, mask_self, pool_size);
}

void task_relu(dahl_any const in, dahl_any out)
{
    int ret = starpu_task_insert(&cl_relu, 
                                 STARPU_R, any_get_handle(in), 
                                 STARPU_W, any_get_handle(out), 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_matrix* task_block_sum_z_axis(dahl_block const* const in)
{
    dahl_shape3d in_shape = block_get_shape(in);
    dahl_shape2d out_shape = { .x = in_shape.x, .y = in_shape.y };
    dahl_matrix* out = matrix_init(out_shape);

    int ret = starpu_task_insert(&cl_block_sum_z_axis,
                                 STARPU_R, block_get_handle(in),
                                 STARPU_W, matrix_get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    return out;
}

void task_scal(dahl_any const in, dahl_any out, dahl_fp const factor)
{
    int ret = starpu_task_insert(&cl_scal,
                             STARPU_VALUE, &factor, sizeof(&factor),
                             STARPU_R, any_get_handle(in),
                             STARPU_W, any_get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_sub(dahl_any const a, dahl_any const b, dahl_any c)
{ 
    int ret = starpu_task_insert(&cl_sub,
                                 STARPU_R, any_get_handle(a),
                                 STARPU_R, any_get_handle(b),
                                 STARPU_W, any_get_handle(c), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_add(dahl_any const a, dahl_any const b, dahl_any c)
{ 
    int ret = starpu_task_insert(&cl_add,
                                 STARPU_R, any_get_handle(a),
                                 STARPU_R, any_get_handle(b),
                                 STARPU_W, any_get_handle(c), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}
