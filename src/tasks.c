#include "codelets.h"
#include "starpu_data.h"
#include "types.h"
#include "tasks.h"

void task_matrix_cross_correlation(dahl_matrix const* const in, dahl_matrix const* const kernel, dahl_matrix* const out)
{
    int ret = starpu_task_insert(&cl_matrix_cross_correlation,
                                 STARPU_R, in->handle,
                                 STARPU_R, kernel->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_matrix_max_pooling(dahl_matrix const* const in, dahl_matrix* const out, dahl_matrix* const mask, size_t const pool_size)
{
    int ret = starpu_task_insert(&cl_matrix_max_pooling,
                             STARPU_VALUE, &pool_size, sizeof(&pool_size),
                             STARPU_R, in->handle,
                             STARPU_W, out->handle, 
                             STARPU_W, mask->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

// ------------------ BACKWARD MAX POOLING ------------------
void call_backward_max_pooling(starpu_data_handle_t in_handle, starpu_data_handle_t mask_handle, starpu_data_handle_t out_handle, size_t const pool_size)
{
    int ret = starpu_task_insert(&cl_matrix_backward_max_pooling,
                             STARPU_VALUE, &pool_size, sizeof(&pool_size),
                             STARPU_R, in_handle,
                             STARPU_R, mask_handle, 
                             STARPU_W, out_handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_matrix_backward_max_pooling(dahl_matrix const* const in, dahl_matrix const* const mask, dahl_matrix* const out, size_t const pool_size)
{
    call_backward_max_pooling(in->handle, mask->handle, out->handle, pool_size);
}

void task_matrix_backward_max_pooling_self(dahl_matrix const* const in, dahl_matrix* const mask_self, size_t const pool_size)
{
    call_backward_max_pooling(in->handle, mask_self->handle, mask_self->handle, pool_size);
}

// ------------------ BACKWARD MAX POOLING ------------------
void call_relu(starpu_data_handle_t in_handle, starpu_data_handle_t out_handle)
{
    int ret = starpu_task_insert(&cl_relu, 
                                 STARPU_R, in_handle, 
                                 STARPU_W, out_handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_any_relu(dahl_any const in, dahl_any out)
{
    starpu_data_handle_t in_handle = any_get_handle(in);
    starpu_data_handle_t out_handle = any_get_handle(out);
    call_relu(in_handle, out_handle);
}

void task_any_relu_self(dahl_any self)
{
    starpu_data_handle_t self_handle = any_get_handle(self);
    call_relu(self_handle, self_handle);
}

// ------------------ SUM Z AXIS ------------------
dahl_matrix* task_block_sum_z_axis(dahl_block const* const in)
{
    shape3d in_shape = block_get_shape(in);
    shape2d out_shape = { .x = in_shape.x, .y = in_shape.y };
    dahl_matrix* out = matrix_init(out_shape);

    int ret = starpu_task_insert(&cl_block_sum_z_axis,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    return out;
}

// ------------------ SCAL SELF ------------------
// TODO: make it return to another buffer by default to match call_sub behaviour, then it call be called with the same buffers to scal self if needed.
void call_scal(starpu_data_handle_t in_handle, starpu_data_handle_t out_handle, dahl_fp const factor)
{
    int ret = starpu_task_insert(&cl_scal,
                             STARPU_VALUE, &factor, sizeof(&factor),
                             STARPU_R, in_handle, 
                             STARPU_W, out_handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_any_scal(dahl_any const in, dahl_any const out,  dahl_fp const factor)
{
    call_scal(
        any_get_handle(in),
        any_get_handle(out),
        factor
    );
}

void task_any_scal_self(dahl_any self, dahl_fp const factor)
{
    starpu_data_handle_t self_handle = any_get_handle(self);
    call_scal(self_handle, self_handle, factor);
}

// ------------------ SUB ------------------
void call_sub(starpu_data_handle_t a_handle, starpu_data_handle_t b_handle, starpu_data_handle_t c_handle)
{ 
    int ret = starpu_task_insert(&cl_sub, 
                                 STARPU_R, a_handle, 
                                 STARPU_R, b_handle, 
                                 STARPU_W, c_handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_any task_any_sub(dahl_any const a, dahl_any const b)
{
    dahl_any c = any_zeros_like(a);

    call_sub(
        any_get_handle(a), 
        any_get_handle(b), 
        any_get_handle(c)
    );

    return c;
}

// Same as `task_any_sub` but here the result is stored into the same buffer a
void task_any_sub_self(dahl_any a_self, dahl_any const b)
{
    starpu_data_handle_t a_handle = any_get_handle(a_self);
    starpu_data_handle_t b_handle = any_get_handle(b);

    call_sub(a_handle, b_handle, a_handle);
}

// ------------------ ADD ------------------
void call_add(starpu_data_handle_t a_handle, starpu_data_handle_t b_handle, starpu_data_handle_t c_handle)
{ 
    int ret = starpu_task_insert(&cl_add, 
                                 STARPU_R, a_handle, 
                                 STARPU_R, b_handle, 
                                 STARPU_W, c_handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

dahl_any task_any_add(dahl_any const a, dahl_any const b)
{
    dahl_any c = any_zeros_like(a);

    call_add(
        any_get_handle(a), 
        any_get_handle(b), 
        any_get_handle(c)
    );

    return c;
}

void task_any_add_self(dahl_any a_self, dahl_any const b)
{
    starpu_data_handle_t a_handle = any_get_handle(a_self);
    starpu_data_handle_t b_handle = any_get_handle(b);

    call_add(a_handle, b_handle, a_handle);
}
