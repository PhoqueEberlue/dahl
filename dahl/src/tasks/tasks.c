
#include "codelets.h"

// Including data.h and not include/dahl_data.h so we have access to the private functions
#include "../data_structures/data_structures.h"
#include "starpu_data.h"
#include "starpu_task.h"
#include "sys/types.h"
#include <stdint.h>
#include <stdio.h>

// ---------------------------------------- TENSOR ----------------------------------------
void task_tensor_sum_t_axis(dahl_tensor const* in, dahl_block* out)
{
    int ret = starpu_task_insert(&cl_tensor_sum_t_axis,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_block* task_tensor_sum_t_axis_init(dahl_arena* arena, dahl_tensor const* in)
{
    dahl_shape4d in_shape = tensor_get_shape(in);

    dahl_shape3d out_shape = {
        .x = in_shape.x,
        .y = in_shape.y,
        .z = in_shape.z,
    };

    dahl_block* out = block_init(arena, out_shape);
    
    task_tensor_sum_t_axis(in, out);

    return out;
}
// ---------------------------------------- BLOCK ----------------------------------------
void task_block_sum_z_axis(dahl_block const* in, dahl_matrix* out)
{
    int ret = starpu_task_insert(&cl_block_sum_z_axis,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_matrix* task_block_sum_z_axis_init(dahl_arena* arena, dahl_block const* in)
{
    dahl_shape3d in_shape = block_get_shape(in);

    dahl_shape2d out_shape = {
        .x = in_shape.x,
        .y = in_shape.y,
    };

    dahl_matrix* out = matrix_init(arena, out_shape);
    
    task_block_sum_z_axis(in, out);

    return out;
}

void task_block_sum_y_axis(dahl_block const* in, dahl_matrix* out)
{
    int ret = starpu_task_insert(&cl_block_sum_y_axis,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_matrix* task_block_sum_y_axis_init(dahl_arena* arena, dahl_block const* in)
{
    dahl_shape3d in_shape = block_get_shape(in);

    // Here we obtain the x, z dimension because y is summed up
    dahl_shape2d out_shape = {
        .x = in_shape.x,
        .y = in_shape.z,
    };

    dahl_matrix* out = matrix_init(arena, out_shape);
    
    task_block_sum_y_axis(in, out);

    return out;
}

void task_block_sum_xy_axes(dahl_block const* in, dahl_vector* out)
{
    int ret = starpu_task_insert(&cl_block_sum_xy_axes,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_vector* task_block_sum_xy_axes_init(dahl_arena* arena, dahl_block const* in)
{
    // Here we obtain a vector of length z because x and y will be summed up
    dahl_vector* out = vector_init(arena, block_get_shape(in).z);
    task_block_sum_xy_axes(in, out);
    return out;
}

void task_block_add_padding(dahl_block const* in, dahl_block* out)
{
    int ret = starpu_task_insert(&cl_block_add_padding,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_block* task_block_add_padding_init(dahl_arena* arena, dahl_block const* in, dahl_shape3d const new_shape)
{
    dahl_block* out = block_init(arena, new_shape);
    task_block_add_padding(in, out);
    return out;
}

// ---------------------------------------- MATRIX ----------------------------------------
void task_matrix_cross_correlation(dahl_matrix const* in, dahl_matrix const* kernel, dahl_matrix* out)
{
    int ret = starpu_task_insert(&cl_matrix_cross_correlation,
                                 STARPU_R, in->handle,
                                 STARPU_R, kernel->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_matrix_max_pooling(dahl_matrix const* in, dahl_matrix* mask, dahl_matrix* out, size_t pool_size)
{
    int ret = starpu_task_insert(&cl_matrix_max_pooling,
                             STARPU_VALUE, &pool_size, sizeof(pool_size),
                             STARPU_R, in->handle,
                             STARPU_W, mask->handle, 
                             STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_matrix_backward_max_pooling(dahl_matrix const* in, dahl_matrix const* mask, dahl_matrix* out, size_t pool_size)
{
    int ret = starpu_task_insert(&cl_matrix_backward_max_pooling,
                             STARPU_VALUE, &pool_size, sizeof(pool_size),
                             STARPU_R, in->handle,
                             STARPU_R, mask->handle, 
                             STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_matrix_backward_max_pooling_self(dahl_matrix const* in, dahl_matrix* mask_self, size_t pool_size)
{
    task_matrix_backward_max_pooling(in, mask_self, mask_self, pool_size);
}

void task_matrix_matrix_product(dahl_matrix const* a, dahl_matrix const* b, dahl_matrix* c)
{
    int ret = starpu_task_insert(&cl_matrix_matrix_product,
                             STARPU_R, a->handle,
                             STARPU_R, b->handle, 
                             STARPU_W, c->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_matrix* task_matrix_matrix_product_init(dahl_arena* arena, dahl_matrix const* a, dahl_matrix const* b)
{
    dahl_shape2d a_shape = matrix_get_shape(a);
    dahl_shape2d b_shape = matrix_get_shape(b);

    dahl_shape2d c_shape = { .x = b_shape.x, .y = a_shape.y };
    dahl_matrix* c = matrix_init(arena, c_shape);

    task_matrix_matrix_product(a, b, c);

    return c;
}

void task_matrix_sum_y_axis(dahl_matrix const* in, dahl_vector* out)
{
    int ret = starpu_task_insert(&cl_matrix_sum_y_axis,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_vector* task_matrix_sum_y_axis_init(dahl_arena* arena, dahl_matrix const* in)
{
    dahl_shape2d in_shape = matrix_get_shape(in);
    dahl_vector* out = vector_init(arena, in_shape.x);
    
    task_matrix_sum_y_axis(in, out);

    return out;
}

void task_matrix_vector_product(dahl_matrix const* mat, dahl_vector const* vec, dahl_vector* out)
{
    int ret = starpu_task_insert(&cl_matrix_vector_product,
                             STARPU_R, mat->handle,
                             STARPU_R, vec->handle,
                             STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_vector* task_matrix_vector_product_init(dahl_arena* arena, dahl_matrix const* mat, dahl_vector const* vec)
{
    dahl_shape2d mat_shape = matrix_get_shape(mat);
    size_t vec_len = vector_get_len(vec);
    
    assert(mat_shape.x == vec_len);

    dahl_vector* out = vector_init(arena, mat_shape.y);

    task_matrix_vector_product(mat, vec, out); 

    return out;
}

void task_matrix_transpose(dahl_matrix const* in, dahl_matrix* out)
{
    int ret = starpu_task_insert(&cl_matrix_transpose,
                             STARPU_R, in->handle,
                             STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_matrix* task_matrix_transpose_init(dahl_arena* arena, dahl_matrix const* in)
{
    dahl_shape2d in_shape = matrix_get_shape(in);
    dahl_shape2d out_shape = { .x = in_shape.y, .y = in_shape.x };
    dahl_matrix* out = matrix_init(arena, out_shape);

    task_matrix_transpose(in, out);

    return out;
}

void task_matrix_resize(dahl_matrix* mat, dahl_shape2d shape)
{
    struct starpu_task *task = starpu_task_create();

    size_t new_nx = shape.x;
    size_t new_ny = shape.y;
    size_t new_ld = new_nx * new_ny;
 
    task->cl = &cl_matrix_resize;
    task->synchronous = 1; // Set to synchronous to prevent any problems.
    
    char *arg_buffer;
    size_t arg_buffer_size;
    starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                         STARPU_VALUE, &new_nx, sizeof(new_nx),
                         STARPU_VALUE, &new_ny, sizeof(new_ny),
                         STARPU_VALUE, &new_ld, sizeof(new_ld), 0);

    task->cl_arg = arg_buffer;
    task->cl_arg_size = arg_buffer_size;
    task->nbuffers = 1;
    task->handles[0] = mat->handle;
 
    /* submit the task to StarPU */
    starpu_task_submit(task);
}

void task_matrix_as_flat_row(dahl_matrix* mat)
{
    dahl_shape2d shape = matrix_get_shape(mat);
    shape.x = shape.x * shape.y;
    shape.y = 1;

    task_matrix_resize(mat, shape);
}

void task_matrix_as_flat_col(dahl_matrix* mat)
{
    dahl_shape2d shape = matrix_get_shape(mat);
    shape.y = shape.x * shape.y;
    shape.x = 1;

    task_matrix_resize(mat, shape);
}

void task_matrix_rotate_180(dahl_matrix const* in, dahl_matrix* out)
{
    int ret = starpu_task_insert(&cl_matrix_rotate_180,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_matrix* task_matrix_rotate_180_init(dahl_arena* arena, dahl_matrix const* in)
{
    dahl_matrix* out = matrix_init(arena, matrix_get_shape(in));
    task_matrix_rotate_180(in, out);
    return out;
}

// ---------------------------------------- VECTOR ----------------------------------------
// Note: do not implement a self function (in and out being the same buffers), as 
// out buffer is used to store partial computations this would mess the results.
void task_vector_softmax(dahl_vector const* in, dahl_vector* out)
{
    int ret = starpu_task_insert(&cl_vector_softmax,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_vector* task_vector_softmax_init(dahl_arena* arena, dahl_vector const* in)
{
    size_t len = vector_get_len(in);
    dahl_vector* out = vector_init(arena, len);

    task_vector_softmax(in, out); 
    return out;
}

void task_vector_dot_product(dahl_vector const* a, dahl_vector const* b, dahl_scalar* c)
{
    int ret = starpu_task_insert(&cl_vector_dot_product,
                                 STARPU_R, a->handle, 
                                 STARPU_R, b->handle, 
                                 STARPU_W, c->handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_matrix_submit");
}

dahl_scalar* task_vector_dot_product_init(dahl_arena* arena, dahl_vector const* a, dahl_vector const* b)
{
    dahl_scalar* res = scalar_init(arena);
    task_vector_dot_product(a, b, res);
    return res;
}

void task_vector_diag(dahl_vector const* in, dahl_matrix* out)
{
    int ret = starpu_task_insert(&cl_vector_diag,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_matrix* task_vector_diag_init(dahl_arena* arena, dahl_vector const* in)
{
    size_t vec_len = vector_get_len(in);

    dahl_shape2d out_shape = { .x = vec_len, .y = vec_len };
    dahl_matrix* out = matrix_init(arena, out_shape);
    task_vector_diag(in, out);
    return out;
}

// TODO: for coherency maybe it should be a codelet on its own? like the basic softmax derivative.
void task_vector_softmax_derivative(dahl_arena* scratch_arena, dahl_vector const* in, dahl_matrix* out)
{
    task_vector_diag(in, out);
    dahl_matrix* in_col = task_vector_to_column_matrix_init(scratch_arena, in);
    dahl_matrix* in_row = task_vector_to_row_matrix_init(scratch_arena, in);
    dahl_matrix* partial_res = task_matrix_matrix_product_init(scratch_arena, in_col, in_row);

    TASK_SUB_SELF(out, partial_res);
}

dahl_matrix* task_vector_softmax_derivative_init(dahl_arena* arena, dahl_arena* scratch_arena, dahl_vector const* in)
{
    size_t vec_len = vector_get_len(in);
    dahl_shape2d out_shape = { .x = vec_len, .y = vec_len };
    dahl_matrix* result = matrix_init(arena, out_shape);
    task_vector_softmax_derivative(scratch_arena, in, result);
    return result;
}

void task_vector_to_matrix(dahl_vector const* in, dahl_matrix* out)
{
    int ret = starpu_task_insert(&cl_vector_to_matrix,
                                 STARPU_R, in->handle, 
                                 STARPU_W, out->handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_matrix_submit");
}

dahl_matrix* task_vector_to_matrix_init(dahl_arena* arena, dahl_vector const* vector, dahl_shape2d new_shape)
{
    dahl_matrix* res = matrix_init(arena, new_shape);
    task_vector_to_matrix(vector, res);
    return res;
}

dahl_matrix* task_vector_to_column_matrix_init(dahl_arena* arena, dahl_vector const* vector)
{
    dahl_shape2d new_shape = { .x = 1, .y = vector_get_len(vector) };
    return task_vector_to_matrix_init(arena, vector, new_shape);
}

dahl_matrix* task_vector_to_row_matrix_init(dahl_arena* arena, dahl_vector const* vector)
{
    dahl_shape2d new_shape = { .x = vector_get_len(vector), .y = 1 };
    return task_vector_to_matrix_init(arena, vector, new_shape);
}

void task_vector_outer_product(dahl_vector const* a, dahl_vector const* b, dahl_matrix* c)
{
    int ret = starpu_task_insert(&cl_vector_outer_product,
                                 STARPU_R, a->handle, 
                                 STARPU_R, b->handle, 
                                 STARPU_W, c->handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_matrix_submit");
}

dahl_matrix* task_vector_outer_product_init(dahl_arena* arena, dahl_vector const* a, dahl_vector const* b)
{
    dahl_shape2d shape = { .x = vector_get_len(a), .y = vector_get_len(b) };
    dahl_matrix* c = matrix_init(arena, shape);
    task_vector_outer_product(a, b, c);
    return c;
}

void task_vector_shuffle(dahl_vector* vec)
{
    int ret = starpu_task_insert(&cl_vector_shuffle,
                                 STARPU_RW, vec->handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_matrix_submit");
}

// ---------------------------------------- TRAITS ----------------------------------------
void task_relu(void const* in, void* out, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(out);
    int ret = starpu_task_insert(&cl_any_relu,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_R, traits->get_handle(in), 
                                 STARPU_W, traits->get_handle(out), 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_matrix_submit");
}

void task_relu_backward(void const* input, void const* gradients, void* out, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(out);
    int ret = starpu_task_insert(&cl_any_relu_backward,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_R, traits->get_handle(input), 
                                 STARPU_R, traits->get_handle(gradients), 
                                 STARPU_W, traits->get_handle(out), 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_matrix_submit");
}

void task_scal(void const* in, void* out, dahl_fp factor, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(out);
    int ret = starpu_task_insert(&cl_any_scal,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_VALUE, &factor, sizeof(factor),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, traits->get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_power(void const* in, void* out, dahl_fp power, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(out);
    int ret = starpu_task_insert(&cl_any_power,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_VALUE, &power, sizeof(power),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, traits->get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_sub(void const* a, void const* b, void* c, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(c);
    int ret = starpu_task_insert(&cl_any_sub,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_R, traits->get_handle(a),
                                 STARPU_R, traits->get_handle(b),
                                 STARPU_W, traits->get_handle(c), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_add(void const* a, void const* b, void* c, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(c);
    int ret = starpu_task_insert(&cl_any_add,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_R, traits->get_handle(a),
                                 STARPU_R, traits->get_handle(b),
                                 STARPU_W, traits->get_handle(c), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_add_value(void const* in, void* out, dahl_fp value, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(out);
    int ret = starpu_task_insert(&cl_any_add_value,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_VALUE, &value, sizeof(value),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, traits->get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_clip(void const* in, void* out, dahl_fp min, dahl_fp max, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(out);
    int ret = starpu_task_insert(&cl_any_clip,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_VALUE, &min, sizeof(min),
                                 STARPU_VALUE, &max, sizeof(max),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, traits->get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_sum(void const* in, dahl_scalar* out, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(in);
    int ret = starpu_task_insert(&cl_any_sum,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_scalar* task_sum_init(dahl_arena* arena, void const* object, dahl_traits* traits)
{
    dahl_scalar* res = scalar_init(arena);
    task_sum(object, res, traits);
    return res;
}

void task_mean(void const* in, dahl_scalar* out, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(in);
    int ret = starpu_task_insert(&cl_any_mean,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_scalar* task_mean_init(dahl_arena* arena, void const* object, dahl_traits* traits)
{
    dahl_scalar* res = scalar_init(arena);
    task_mean(object, res, traits);
    return res;
}

void task_fill(void* object, dahl_fp value, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(object);
    int ret = starpu_task_insert(&cl_any_fill,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_VALUE, &value, sizeof(value),
                                 STARPU_W, traits->get_handle(object), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_wait(void const* object, unsigned int duration, dahl_traits* traits)
{
    int ret = starpu_task_insert(&cl_any_wait,
                                 STARPU_VALUE, &duration, sizeof(duration),
                                 STARPU_W, traits->get_handle(object), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_copy(void const* in, void* out, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(in);

    int ret = starpu_task_insert(&cl_any_copy,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, traits->get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_min(void const* in, dahl_scalar* out, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(in);
    int ret = starpu_task_insert(&cl_any_min,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_scalar* task_min_init(dahl_arena* arena, void const* object, dahl_traits* traits)
{
    dahl_scalar* res = scalar_init(arena);
    task_min(object, res, traits);
    return res;
}

void task_max(void const* in, dahl_scalar* out, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(in);
    int ret = starpu_task_insert(&cl_any_max,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_scalar* task_max_init(dahl_arena* arena, void const* object, dahl_traits* traits)
{
    dahl_scalar* res = scalar_init(arena);
    task_max(object, res, traits);
    return res;
}

void task_round(void const* in, void* out, int8_t precision, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(in);
    int ret = starpu_task_insert(&cl_any_round,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_VALUE, &precision, sizeof(precision),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, traits->get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

// ---------------------------------------- ML Related ----------------------------------------
void task_check_predictions_batch(dahl_matrix const* prediction_batch, dahl_matrix const* target_batch, dahl_scalar* good_predictions)
{
    int ret = starpu_task_insert(&cl_check_predictions_batch,
                                 STARPU_R, prediction_batch->handle,
                                 STARPU_R, target_batch->handle,
                                 STARPU_W, good_predictions->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_scalar* task_check_predictions_batch_init(dahl_arena* arena, dahl_matrix const* prediction_batch, dahl_matrix const* target_batch)
{
    dahl_scalar* res = scalar_init(arena);
    task_check_predictions_batch(prediction_batch, target_batch, res);
    return res;
}

void task_cross_entropy_loss_batch(dahl_matrix const* prediction_batch, dahl_matrix const* target_batch, dahl_scalar* out)
{
    int ret = starpu_task_insert(&cl_cross_entropy_loss_batch,
                                 STARPU_R, prediction_batch->handle,
                                 STARPU_R, target_batch->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_scalar* task_cross_entropy_loss_batch_init(dahl_arena* arena, dahl_matrix const* prediction_batch, dahl_matrix const* target_batch)
{
    dahl_scalar* res = scalar_init(arena);
    task_cross_entropy_loss_batch(prediction_batch, target_batch, res);
    return res;
}

void task_cross_entropy_loss_gradient_batch(dahl_matrix const* predictions, dahl_matrix const* targets, dahl_matrix* gradients)
{
    int ret = starpu_task_insert(&cl_cross_entropy_loss_gradient,
                             STARPU_R, predictions->handle,
                             STARPU_R, targets->handle,
                             STARPU_W, gradients->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_matrix* task_cross_entropy_loss_gradient_batch_init(dahl_arena* arena, dahl_matrix const* prediction_batch, 
                                                                dahl_matrix const* target_batch)
{
    dahl_matrix* gradient_batch = matrix_init(arena, matrix_get_shape(prediction_batch));
    task_cross_entropy_loss_gradient_batch(prediction_batch, target_batch, gradient_batch);
    return gradient_batch;
}

void task_convolution_2d(dahl_block const* in, dahl_block const* kernel, dahl_matrix* out)
{
    int ret = starpu_task_insert(&cl_convolution_2d,
                                 STARPU_R, in->handle,
                                 STARPU_R, kernel->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_convolution_2d_backward_filters(dahl_block const* in, dahl_matrix const* kernel, dahl_block* out)
{
    int ret = starpu_task_insert(&cl_convolution_2d_backward_filters,
                                 STARPU_R, in->handle,
                                 STARPU_R, kernel->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}
