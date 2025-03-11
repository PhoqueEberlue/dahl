#include "convolution.h"
#include "codelets.h"
#include "starpu_task_util.h"
#include "utils.h"

convolution create_convolution(shape2d input_shape, size_t filter_size, size_t num_filters)
{
    shape3d filter_shape = {
        .x = filter_size,
        .y = filter_size,
        .z = num_filters,
    };
    
    starpu_data_handle_t filters_handle = block_init(filter_shape);
    block_fill_random(filters_handle);

    shape3d output_shape = {
        .x = input_shape.x - filter_size + 1,
        .y = input_shape.y - filter_size + 1,
        .z = num_filters,
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

    struct starpu_task* task = starpu_task_create();
    task->cl = &cl_cross_correlation_2d;
    task->handles[0] = input_handle;
    task->handles[1] = conv.filters_handle;
    task->handles[2] = output_handle;
    task->synchronous = 1;
 
    int ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");


    ret = starpu_task_insert(&cl_relu, STARPU_RW, output_handle);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

    starpu_task_wait_for_all();

    return output_handle;
}

// starpu_data_handle_t backward_pass(convolution conv, const starpu_data_handle_t dl_dout, const double learning_rate, const starpu_data_handle_t input_handle)
// {
//     // derivative loss
//     starpu_data_handle_t dl_dinput = matrix_init(conv.input_shape);
//     starpu_data_handle_t dl_dfilters = block_init(conv.output_shape);
// 
//     // performs cross correlation on input data with each derivative output (dl_dout) and store it in dl_dfilters
//     struct starpu_task* task = starpu_task_create();
//     task->cl = &cl_cross_correlation_2d;
//     task->handles[0] = input_handle;
//     task->handles[1] = dl_dout;
//     task->handles[2] = dl_dfilters;
//     task->synchronous = 1;
//  
//     int ret = starpu_task_submit(task);
// 	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
// 
//     // Here we to do:  dL_dinput += correlate2d(dL_dout[i],self.filters[i], mode="full")
//     // however dl_dout[i] isnt possible in our implem because we take into account that this is a block
//     // performs cross correlation on input data with each derivative output (dl_dout) and store it in dl_dfilters
//     struct starpu_task* task_2 = starpu_task_create();
//     task_2->cl = &cl_cross_correlation_2d;
//     task_2->handles[0] = input_handle;
//     task_2->handles[1] = dl_dout;
//     task_2->handles[2] = dl_dinput;
//     task_2->synchronous = 1;
//  
//     ret = starpu_task_submit(task_2);
// 	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
// }
