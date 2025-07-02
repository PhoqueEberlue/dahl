
#include "codelets.h"

// Including data.h and not include/dahl_data.h so we have access to the private functions
#include "../data_structures/data_structures.h"
#include "starpu_data.h"
#include "starpu_task.h"

// ---------------------------------------- BLOCK ----------------------------------------
void task_block_sum_z_axis(dahl_block const* in, dahl_matrix* out)
{
    int ret = starpu_task_insert(&cl_block_sum_z_axis,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_matrix* task_block_sum_z_axis_init(dahl_block const* in)
{
    dahl_shape3d in_shape = block_get_shape(in);

    dahl_shape2d out_shape = {
        .x = in_shape.x,
        .y = in_shape.y,
    };

    dahl_matrix* out = matrix_init(out_shape);
    
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

dahl_matrix* task_block_sum_y_axis_init(dahl_block const* in)
{
    dahl_shape3d in_shape = block_get_shape(in);

    // Here we obtain the x, z dimension because y is summed up
    dahl_shape2d out_shape = {
        .x = in_shape.x,
        .y = in_shape.z,
    };

    dahl_matrix* out = matrix_init(out_shape);
    
    task_block_sum_y_axis(in, out);

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

void task_matrix_max_pooling(dahl_matrix const* in, dahl_matrix* out, dahl_matrix* mask, size_t pool_size)
{
    int ret = starpu_task_insert(&cl_matrix_max_pooling,
                             STARPU_VALUE, &pool_size, sizeof(&pool_size),
                             STARPU_R, in->handle,
                             STARPU_W, out->handle, 
                             STARPU_W, mask->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_matrix_backward_max_pooling(dahl_matrix const* in, dahl_matrix const* mask, dahl_matrix* out, size_t pool_size)
{
    int ret = starpu_task_insert(&cl_matrix_backward_max_pooling,
                             STARPU_VALUE, &pool_size, sizeof(&pool_size),
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

dahl_matrix* task_matrix_matrix_product_init(dahl_matrix const* a, dahl_matrix const* b)
{
    dahl_shape2d a_shape = matrix_get_shape(a);
    dahl_shape2d b_shape = matrix_get_shape(b);

    dahl_shape2d c_shape = { .x = b_shape.x, .y = a_shape.y };
    dahl_matrix* c = matrix_init(c_shape);

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

dahl_vector* task_matrix_sum_y_axis_init(dahl_matrix const* in)
{
    dahl_shape2d in_shape = matrix_get_shape(in);
    dahl_vector* out = vector_init(in_shape.x);
    
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

dahl_vector* task_matrix_vector_product_init(dahl_matrix const* mat, dahl_vector const* vec)
{
    dahl_shape2d mat_shape = matrix_get_shape(mat);
    size_t vec_len = vector_get_len(vec);
    
    assert(mat_shape.x == vec_len);

    dahl_vector* out = vector_init(mat_shape.y);

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

dahl_matrix* task_matrix_transpose_init(dahl_matrix const* in)
{
    dahl_shape2d in_shape = matrix_get_shape(in);
    dahl_shape2d out_shape = { .x = in_shape.y, .y = in_shape.x };
    dahl_matrix* out = matrix_init(out_shape);

    task_matrix_transpose(in, out);

    return out;
}

void task_matrix_resize(dahl_matrix* mat, size_t new_nx, size_t new_ny, size_t new_ld)
{
    struct starpu_task *task = starpu_task_create();
 
    task->cl = &cl_matrix_resize;
    task->synchronous = 1; // Set to synchronous to prevent any problems.
    
    char *arg_buffer;
    size_t arg_buffer_size;
    starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                         STARPU_VALUE, &new_nx, sizeof(&new_nx),
                         STARPU_VALUE, &new_ny, sizeof(&new_ny),
                         STARPU_VALUE, &new_ld, sizeof(&new_ld), 0);

    task->cl_arg = arg_buffer;
    task->cl_arg_size = arg_buffer_size;
    task->nbuffers = 1;
    task->handles[0] = mat->handle;
 
    /* submit the task to StarPU */
    starpu_task_submit(task);
}

void task_matrix_to_flat_row(dahl_matrix* mat)
{
    dahl_shape2d shape = matrix_get_shape(mat);
    size_t new_nx = shape.x * shape.y;
    size_t new_ny = 1;
    size_t new_ld = new_nx;

    task_matrix_resize(mat, new_nx, new_ny, new_ld);
}

void task_matrix_to_flat_col(dahl_matrix* mat)
{
    dahl_shape2d shape = matrix_get_shape(mat);
    size_t new_nx = 1;
    size_t new_ny = shape.x * shape.y;
    size_t new_ld = 1;

    task_matrix_resize(mat, new_nx, new_ny, new_ld);
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

dahl_vector* task_vector_softmax_init(dahl_vector const* in)
{
    size_t len = vector_get_len(in);
    dahl_vector* out = vector_init(len);

    task_vector_softmax(in, out); 
    return out;
}

dahl_fp task_vector_dot_product(dahl_vector const* a, dahl_vector const* b)
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
    task->handles[0] = a->handle;
    task->handles[1] = b->handle;
    task->detach = 0;

    int ret = starpu_task_submit(task); 
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    ret = starpu_task_wait(task);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    return res;
}

dahl_matrix* task_vector_diag(dahl_vector const* in)
{
    size_t vec_len = vector_get_len(in);

    dahl_shape2d out_shape = { .x = vec_len, .y = vec_len };
    dahl_matrix* out = matrix_init(out_shape);

    int ret = starpu_task_insert(&cl_vector_diag,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    return out;
}

// TODO: for coherency maybe it should be a codelet on its own? like the basic softmax derivative.
void task_vector_softmax_derivative(dahl_vector const* in, dahl_matrix* out)
{
    // Init in the temporary arena 
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_temporary_arena;

    dahl_matrix* in_col = vector_to_column_matrix(in);
    dahl_matrix* in_row = vector_to_row_matrix(in);
    dahl_matrix* tmp = task_matrix_matrix_product_init(in_col, in_row);

    // Then switch to previous context.
    dahl_context_arena = save_arena;

    TASK_SUB_SELF(out, tmp);
}

dahl_matrix* task_vector_softmax_derivative_init(dahl_vector const* in)
{
    dahl_matrix* result = task_vector_diag(in);
    task_vector_softmax_derivative(in, result);
    return result;
}

dahl_fp task_vector_cross_entropy_loss(dahl_vector const* predictions, dahl_vector const* targets)
{
    dahl_fp const epsilon = 1e-7F;
    size_t const n_classes = vector_get_len(predictions);

    // Init in the temporary arena 
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_temporary_arena;

    dahl_vector* tmp = vector_init(n_classes);

    // Then switch to previous context.
    dahl_context_arena = save_arena;

    TASK_CLIP(predictions, tmp, epsilon, 1 - epsilon);

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
    task->handles[0] = tmp->handle;
    task->handles[1] = targets->handle;
    task->detach = 0;

    int ret = starpu_task_submit(task); 
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    ret = starpu_task_wait(task);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    return res;
}

void task_vector_cross_entropy_loss_gradient(dahl_vector const* predictions, dahl_vector const* targets, dahl_vector* gradients)
{
    int ret = starpu_task_insert(&cl_vector_cross_entropy_loss_gradient,
                             STARPU_R, predictions->handle,
                             STARPU_R, targets->handle,
                             STARPU_W, gradients->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_vector* task_vector_cross_entropy_loss_gradient_init(dahl_vector const* predictions, dahl_vector const* targets)
{
    size_t len = vector_get_len(predictions);

    // Init in the temporary arena 
    dahl_arena* const save_arena = dahl_context_arena;
    dahl_context_arena = dahl_temporary_arena;

    dahl_vector* gradients = vector_init(len);

    // Then switch to previous context.
    dahl_context_arena = save_arena;

    task_vector_cross_entropy_loss_gradient(predictions, targets, gradients);

    return gradients;
}

// ---------------------------------------- TRAITS ----------------------------------------
// Tasks that can be applied to any data type are defined with traits.
// Each trait links a type with the right codelets.
dahl_traits dahl_traits_tensor = {
    .get_handle = _tensor_get_handle,
    .get_nb_elem = _tensor_get_nb_elem,
};

dahl_traits dahl_traits_block = {
    .get_handle = _block_get_handle,
    .get_nb_elem = _block_get_nb_elem,
};

dahl_traits dahl_traits_matrix = {
    .get_handle = _matrix_get_handle,
    .get_nb_elem = _matrix_get_nb_elem,
};

dahl_traits dahl_traits_vector = {
    .get_handle = _vector_get_handle,
    .get_nb_elem = _vector_get_nb_elem,
};

void task_relu(void const* in, void* out, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(out);
    int ret = starpu_task_insert(&cl_relu,
                                 STARPU_VALUE, &nb_elem, sizeof(&nb_elem),
                                 STARPU_R, traits->get_handle(in), 
                                 STARPU_W, traits->get_handle(out), 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_matrix_submit");
}

void task_scal(void const* in, void* out, dahl_fp factor, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(out);
    int ret = starpu_task_insert(&cl_scal,
                                 STARPU_VALUE, &nb_elem, sizeof(&nb_elem),
                                 STARPU_VALUE, &factor, sizeof(&factor),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, traits->get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_sub(void const* a, void const* b, void* c, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(c);
    int ret = starpu_task_insert(&cl_sub,
                                 STARPU_VALUE, &nb_elem, sizeof(&nb_elem),
                                 STARPU_R, traits->get_handle(a),
                                 STARPU_R, traits->get_handle(b),
                                 STARPU_W, traits->get_handle(c), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_add(void const* a, void const* b, void* c, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(c);
    int ret = starpu_task_insert(&cl_add,
                                 STARPU_VALUE, &nb_elem, sizeof(&nb_elem),
                                 STARPU_R, traits->get_handle(a),
                                 STARPU_R, traits->get_handle(b),
                                 STARPU_W, traits->get_handle(c), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_add_value(void const* in, void* out, dahl_fp value, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(out);
    int ret = starpu_task_insert(&cl_add_value,
                                 STARPU_VALUE, &nb_elem, sizeof(&nb_elem),
                                 STARPU_VALUE, &value, sizeof(&value),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, traits->get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_clip(void const* in, void* out, dahl_fp min, dahl_fp max, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(out);
    int ret = starpu_task_insert(&cl_clip,
                                 STARPU_VALUE, &nb_elem, sizeof(&nb_elem),
                                 STARPU_VALUE, &min, sizeof(&min),
                                 STARPU_VALUE, &max, sizeof(&max),
                                 STARPU_R, traits->get_handle(in),
                                 STARPU_W, traits->get_handle(out), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_fp task_sum(void const* object, dahl_traits* traits)
{
    dahl_fp res = 0.0F;
    dahl_fp* res_p = &res;

    size_t nb_elem = traits->get_nb_elem(object);

    struct starpu_task* task = starpu_task_create();
    task->cl = &cl_sum;

    // Initialize argument buffer to obtain the return value with a pointer pointer
    char *arg_buffer;
    size_t arg_buffer_size;
    starpu_codelet_pack_args((void**)&arg_buffer, &arg_buffer_size,
                        STARPU_VALUE, &nb_elem, sizeof(&nb_elem),
                        STARPU_VALUE, &res_p, sizeof(&res_p), 0);

    task->cl_arg = arg_buffer;
    task->cl_arg_size = arg_buffer_size;
    task->nbuffers = 1;
    task->handles[0] = traits->get_handle(object);
    task->detach = 0;

    int ret = starpu_task_submit(task); 
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    ret = starpu_task_wait(task);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    return res;
}

void task_fill(void const* object, dahl_fp value, dahl_traits* traits)
{
    size_t nb_elem = traits->get_nb_elem(object);
    int ret = starpu_task_insert(&cl_fill,
                                 STARPU_VALUE, &nb_elem, sizeof(&nb_elem),
                                 STARPU_VALUE, &value, sizeof(&value),
                                 STARPU_W, traits->get_handle(object), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}
