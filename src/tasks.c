#include "codelets.h"
#include "tasks.h"

void task_cross_correlation_2d(const starpu_data_handle_t a, const starpu_data_handle_t b, const starpu_data_handle_t c)
{
    int ret = starpu_task_insert(&cl_cross_correlation_2d, 
                                 STARPU_R, a, 
                                 STARPU_R, b, 
                                 STARPU_W, c, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

void task_relu(starpu_data_handle_t in)
{
    int ret = starpu_task_insert(&cl_relu, STARPU_RW, in);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

void task_scal(starpu_data_handle_t in, dahl_fp factor)
{
    // Factor may die before the task finishes?
    int ret = starpu_task_insert(&cl_relu,
                             STARPU_VALUE, &factor, sizeof(&factor),
                             STARPU_RW, in, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

void task_sub(starpu_data_handle_t a, const starpu_data_handle_t b)
{
    int ret = starpu_task_insert(&cl_sub, 
                                 STARPU_RW, a, 
                                 STARPU_R, b, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

void task_add(const starpu_data_handle_t a, const starpu_data_handle_t b)
{
    int ret = starpu_task_insert(&cl_add, 
                                 STARPU_RW, a, 
                                 STARPU_R, b, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

void task_sum_z_axis(const starpu_data_handle_t in, starpu_data_handle_t out)
{
    int ret = starpu_task_insert(&cl_sum_z_axis,
                                 STARPU_R, in,
                                 STARPU_W, out, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}
