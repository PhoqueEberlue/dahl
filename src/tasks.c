#include "codelets.h"

// Including data.h and not include/dahl_data.h so we have access to the private functions
#include "data.h"

#include "../include/dahl_tasks.h"
#include <stdio.h>

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
    int ret = starpu_task_insert(&cl_matrix_max_pooling,
                             STARPU_VALUE, &pool_size, sizeof(&pool_size),
                             STARPU_R, matrix_get_handle(in),
                             STARPU_W, matrix_get_handle(out), 
                             STARPU_W, matrix_get_handle(mask), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_matrix_backward_max_pooling(dahl_matrix const* const in, dahl_matrix const* const mask, dahl_matrix* const out, size_t const pool_size)
{
    int ret = starpu_task_insert(&cl_matrix_backward_max_pooling,
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

void task_matrix_matrix_product(dahl_matrix const* const a, dahl_matrix const* const b, dahl_matrix* const c)
{
    int ret = starpu_task_insert(&cl_matrix_matrix_product,
                             STARPU_R, matrix_get_handle(a),
                             STARPU_R, matrix_get_handle(b), 
                             STARPU_W, matrix_get_handle(c), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_matrix* task_matrix_matrix_product_init(dahl_matrix const* const a, dahl_matrix const* const b)
{
    dahl_shape2d a_shape = matrix_get_shape(a);
    dahl_shape2d b_shape = matrix_get_shape(b);

    dahl_shape2d c_shape = { .x = a_shape.x, .y = b_shape.y };
    dahl_matrix* const c = matrix_init(c_shape);

    task_matrix_matrix_product(a, b, c);

    return c;
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

// Note: do not implement a self function (in and out being the same buffers), as 
// out buffer is used to store partial computations this would mess the results.
void task_vector_softmax(dahl_vector const* const in, dahl_vector* const out)
{
    int ret = starpu_task_insert(&cl_vector_softmax,
                                 STARPU_R, vector_get_handle(in),
                                 STARPU_W, vector_get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_vector* task_vector_softmax_init(dahl_vector const* const in)
{
    size_t len = vector_get_len(in);
    dahl_vector* out = vector_init(len);

    task_vector_softmax(in, out); 
    return out;
}

dahl_fp task_vector_dot_product(dahl_vector const* const a, dahl_vector const* const b)
{
    dahl_fp res = 0;
    dahl_fp* res_p = &res;

    struct starpu_task* task = starpu_task_create();
    task->cl = &cl_vector_dot_product;

    // Initialize argument buffer to obtain the return value with a pointer pointer
    char *arg_buffer;
    size_t arg_buffer_size;
    starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                        STARPU_VALUE, &res_p, sizeof(&res_p), 0);

    task->cl_arg = arg_buffer;
    task->cl_arg_size = arg_buffer_size;
    task->nbuffers = 2;
    task->handles[0] = vector_get_handle(a);
    task->handles[1] = vector_get_handle(b);
    task->detach = 0;

    int ret = starpu_task_submit(task); 
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    ret = starpu_task_wait(task);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    return res;
}

dahl_matrix* task_vector_diag(dahl_vector const* const in)
{
    size_t vec_len = vector_get_len(in);

    dahl_shape2d out_shape = { .x = vec_len, .y = vec_len };
    dahl_matrix* out = matrix_init(out_shape);

    int ret = starpu_task_insert(&cl_vector_diag,
                                 STARPU_R, vector_get_handle(in),
                                 STARPU_W, matrix_get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    return out;
}

void task_add_value(dahl_any const in, dahl_any out, dahl_fp const value)
{
    int ret = starpu_task_insert(&cl_add_value,
                             STARPU_VALUE, &value, sizeof(&value),
                             STARPU_R, any_get_handle(in),
                             STARPU_W, any_get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_sub_value(dahl_any const in, dahl_any out, dahl_fp const value)
{
    int ret = starpu_task_insert(&cl_sub_value,
                             STARPU_VALUE, &value, sizeof(&value),
                             STARPU_R, any_get_handle(in),
                             STARPU_W, any_get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

// TODO: for coherency maybe it should be a codelet on its own? like the basic softmax derivative.
dahl_matrix* task_vector_softmax_derivative(dahl_vector const* const in)
{
    dahl_matrix* result = task_vector_diag(in);
    dahl_fp value = task_vector_dot_product(in, in);

    TASK_SUB_VALUE_SELF(result, value);

    return result;
}

void task_matrix_vector_product(dahl_matrix const* const mat, dahl_vector const* const vec, dahl_vector* const out)
{
    int ret = starpu_task_insert(&cl_matrix_vector_product,
                             STARPU_R, matrix_get_handle(mat),
                             STARPU_R, vector_get_handle(vec),
                             STARPU_W, vector_get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_vector* task_matrix_vector_product_init(dahl_matrix const* const mat, dahl_vector const* const vec)
{
    dahl_shape2d mat_shape = matrix_get_shape(mat);
    size_t vec_len = vector_get_len(vec);
    dahl_vector* out;
    if (mat_shape.x == vec_len)
    {
        out = vector_init(mat_shape.y);
    }
    else if (mat_shape.y == vec_len)
    {
        out = vector_init(mat_shape.x);
    }
    else
    {
        printf("The input matrix doesn't match any of the vector's dimension mat(%lu, %lu) vec(%lu)", 
               mat_shape.x, mat_shape.y, vec_len);
        abort();
    }

    task_matrix_vector_product(mat, vec, out); 

    return out;
}

void task_clip(dahl_any const in, dahl_any const out, dahl_fp const min, dahl_fp const max)
{
    int ret = starpu_task_insert(&cl_clip,
                             STARPU_VALUE, &min, sizeof(&min),
                             STARPU_VALUE, &max, sizeof(&max),
                             STARPU_R, any_get_handle(in),
                             STARPU_W, any_get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_fp task_vector_cross_entropy_loss(dahl_vector const* const predictions, dahl_vector const* const targets)
{
    dahl_fp const epsilon = 1e-7F;
    size_t const n_classes = vector_get_len(predictions);

    dahl_vector* tmp = vector_init(n_classes);

    // TODO: '-- why do I need to cast here
    TASK_CLIP((dahl_vector*)predictions, tmp, epsilon, 1 - epsilon);

    dahl_fp res = 0;
    dahl_fp* res_p = &res;

    struct starpu_task* task = starpu_task_create();
    task->cl = &cl_vector_cross_entropy_loss;

    // Initialize argument buffer to obtain the return value with a pointer pointer
    char *arg_buffer;
    size_t arg_buffer_size;
    starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                        STARPU_VALUE, &res_p, sizeof(&res_p), 0);

    task->cl_arg = arg_buffer;
    task->cl_arg_size = arg_buffer_size;
    task->nbuffers = 2;
    task->handles[0] = vector_get_handle(tmp);
    task->handles[1] = vector_get_handle(targets);
    task->detach = 0;

    int ret = starpu_task_submit(task); 
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    ret = starpu_task_wait(task);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    return res;
}

dahl_vector* task_vector_cross_entropy_loss_gradient(dahl_vector const* const predictions, dahl_vector const* const targets)
{
    size_t len = vector_get_len(predictions);
    dahl_vector* out = vector_init(len);

    int ret = starpu_task_insert(&cl_vector_cross_entropy_loss_gradient,
                             STARPU_R, vector_get_handle(predictions),
                             STARPU_R, vector_get_handle(targets),
                             STARPU_W, vector_get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    return out;
}
