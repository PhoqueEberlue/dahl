#include "codelets.h"
#include "types.h"
#include "tasks.h"

void task_cross_correlation_2d(dahl_matrix const* const a, dahl_matrix const* const b, dahl_matrix* const c)
{
    int ret = starpu_task_insert(&cl_cross_correlation_2d,
                                 STARPU_R, a->handle,
                                 STARPU_R, b->handle,
                                 STARPU_W, c->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

void task_relu(dahl_block* const in)
{
    int ret = starpu_task_insert(&cl_relu, STARPU_RW, in->handle);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

dahl_matrix* task_sum_z_axis(dahl_block const* const in)
{
    shape3d in_shape = block_get_shape(in);
    shape2d out_shape = { .x = in_shape.x, .y = in_shape.y };
    dahl_matrix* out = matrix_init(out_shape);

    int ret = starpu_task_insert(&cl_sum_z_axis,
                                 STARPU_R, in->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

    return out;
}

void task_scal(dahl_block* const in, dahl_fp const factor)
{
    //TODO: Factor may die before the task finishes? Or is it copied?
    int ret = starpu_task_insert(&cl_relu,
                             STARPU_VALUE, &factor, sizeof(&factor),
                             STARPU_RW, in->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

dahl_block* task_sub(dahl_block const* const a, dahl_block const* const b)
{
    // Output shape is the same as input's one
    shape3d c_shape = block_get_shape(a);
    dahl_block* c = block_init(c_shape);

    int ret = starpu_task_insert(&cl_sub, 
                                 STARPU_R, a->handle, 
                                 STARPU_R, b->handle, 
                                 STARPU_W, c->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    return c;
}

dahl_block* task_add(dahl_block const* const a, dahl_block const* const b)
{
    // Output shape is the same as input's one
    shape3d c_shape = block_get_shape(a);
    dahl_block* c = block_init(c_shape);

    int ret = starpu_task_insert(&cl_add, 
                                 STARPU_R, a->handle, 
                                 STARPU_R, b->handle, 
                                 STARPU_W, c->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

    return c;
}
