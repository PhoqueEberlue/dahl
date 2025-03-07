#include "convolution.h"
#include "codelets.h"
#include "utils.h"

convolution create_convolution(shape2d input_shape, size_t filter_size, size_t num_filters)
{
    shape3d filter_shape = {
        .x = num_filters,
        .y = filter_size,
        .z = filter_size,
    };
    
    starpu_data_handle_t filters_handle = block_init(filter_shape);
    block_fill_random(filters_handle);

    shape3d output_shape = {
        .x = num_filters,
        .y = input_shape.x - filter_size + 1,
        .z = input_shape.y - filter_size + 1,
    };

    starpu_data_handle_t biases_handle = block_init(output_shape);
    block_fill_random(biases_handle);

    convolution conv = {
        .input_shape = input_shape,
        .num_filters = num_filters,
        .filter_size = filter_size,
        .filter_shape = filter_shape,
        .output_shape = output_shape,
        .filters_handle = filters_handle,
        .biases_handle = biases_handle,
    };

    return conv;
}

starpu_data_handle_t forward_pass(convolution conv, const starpu_data_handle_t input_handle)
{
    starpu_data_handle_t output_handle = block_init(conv.output_shape);

    matrix_print_from_handle(input_handle);
    block_print_from_handle(conv.filters_handle);

    struct starpu_task* task = starpu_task_create();
    task->cl = &cl_cross_correlation_2d;                      /* Pointer to the codelet defined below */
    task->handles[0] = input_handle;    /* First parameter of the codelet */
    task->handles[1] = conv.filters_handle;    /* Second parameter of the codelet */
    task->handles[2] = output_handle;    /* Third parameter of the codelet */
    task->synchronous = 1;
 
    int ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

    starpu_task_wait_for_all();

    return output_handle;
}
