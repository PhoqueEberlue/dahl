#include "convolution.h"
#include "codelets.h"
#include "starpu_data.h"
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

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_depth_block,
		.nchildren = conv.num_filters
	};

	starpu_data_partition(output_handle, &f);
	starpu_data_partition(conv.filters_handle, &f);

    for (int i = 0; i < conv.num_filters; i++)
    {
		starpu_data_handle_t sub_output = starpu_data_get_sub_data(output_handle, 1, i);
		starpu_data_handle_t sub_filters = starpu_data_get_sub_data(conv.filters_handle, 1, i);

        struct starpu_task* task = starpu_task_create();
        task->cl = &cl_cross_correlation_2d;
        task->handles[0] = input_handle;
        task->handles[1] = sub_filters;
        task->handles[2] = sub_output;
        task->synchronous = 0;
     
        int ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    }
    
	starpu_data_unpartition(output_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(conv.filters_handle, STARPU_MAIN_RAM);

    int ret = starpu_task_insert(&cl_relu, STARPU_RW, output_handle);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

    // Could be interesting to know if the relu task is really waiting for other tasks because starting?
    // It should be the case because of the data dependency and because it is working but we may verify that
    starpu_task_wait_for_all();

    return output_handle;
}

void gradient_loss_kernels(const starpu_data_handle_t input_data, const starpu_data_handle_t dl_dout, starpu_data_handle_t dl_dfilters)
{
    // performs cross correlation on input data with each derivative output (dl_dout) and store it in dl_dfilters
    struct starpu_task* task = starpu_task_create();
    task->cl = &cl_cross_correlation_2d;
    task->handles[0] = input_data;
    task->handles[1] = dl_dout;
    task->handles[2] = dl_dfilters;
    task->synchronous = 0;

    int ret = starpu_task_submit(task);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

void gradient_loss_inputs(const starpu_data_handle_t dl_dout, const starpu_data_handle_t filters, starpu_data_handle_t dl_dinput)
{
    // Here we to do:  dL_dinput += correlate2d(dL_dout[i],self.filters[i], mode="full")
    struct starpu_task* task = starpu_task_create();
    task->cl = &cl_cross_correlation_2d;
    task->handles[0] = dl_dout;
    task->handles[1] = filters;
    task->handles[2] = dl_dinput;
    task->synchronous = 0;
 
    int ret = starpu_task_submit(task);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

// "Cognitive complexity of 257 (threshold 25)" kekw
starpu_data_handle_t backward_pass(convolution conv, const starpu_data_handle_t dl_dout, const double learning_rate, const starpu_data_handle_t input_handle)
{
    shape3d input_shape3d = { .x = conv.input_shape.x, .y = conv.input_shape.y, .z = conv.num_filters };
    // derivative loss
    starpu_data_handle_t dl_dinput = block_init(input_shape3d);
    starpu_data_handle_t dl_dfilters = block_init(conv.output_shape);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_depth_block,
		.nchildren = conv.num_filters
	};

	starpu_data_partition(dl_dout, &f);
	starpu_data_partition(dl_dfilters, &f);
	starpu_data_partition(conv.filters_handle, &f);
	starpu_data_partition(dl_dinput, &f);

    for (int i = 0; i < conv.num_filters; i++)
    {
		starpu_data_handle_t sub_dl_dout = starpu_data_get_sub_data(dl_dout, 1, i);
		starpu_data_handle_t sub_dl_dfilters = starpu_data_get_sub_data(dl_dfilters, 1, i);
        gradient_loss_kernels(input_handle, sub_dl_dout, sub_dl_dfilters);

        starpu_data_handle_t sub_filters = starpu_data_get_sub_data(conv.filters_handle, 1, i);
        starpu_data_handle_t sub_dl_dinput = starpu_data_get_sub_data(dl_dinput, 1, i);
        
        gradient_loss_inputs()
    }

    starpu_task_wait_for_all();

    starpu_data_unpartition(dl_dout, STARPU_MAIN_RAM);
	starpu_data_unpartition(dl_dfilters, STARPU_MAIN_RAM);
	starpu_data_unpartition(conv.filters_handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(dl_dinput, STARPU_MAIN_RAM); 
    
    // Updating filters and biases
    // self.filters -= lr * dL_dfilters
    // self.biases -= lr * dL_dout

    // task for lr * dL_dfilters
    struct starpu_task *task = starpu_task_create();
    task->cl = &cl_scal;
    task->handles[0] = dl_dfilters;
    task->cl_arg = &learning_rate;
    task->cl_arg_size = sizeof(&learning_rate);
    task->synchronous = 0;

    int ret = starpu_task_submit(task);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

    // task for lr * dL_dout
    struct starpu_task *task_2 = starpu_task_create();
    task_2->cl = &cl_scal;
    task_2->handles[0] = dl_dout;
    task_2->cl_arg = &learning_rate;
    task_2->cl_arg_size = sizeof(&learning_rate);
    task_2->synchronous = 0;

    ret = starpu_task_submit(task_2);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

    // task for self.filters -= (lr * dL_dfilters)
    ret = starpu_task_insert(&cl_sub, STARPU_RW, conv.filters_handle, STARPU_R, dl_dfilters, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

    // task for self.biases -= (lr * dL_dout)
    ret = starpu_task_insert(&cl_sub, STARPU_RW, conv.biases_handle, STARPU_R, dl_dout, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

    starpu_task_wait_for_all();

    // Then don't forget to compute the sum of dl_dfilters
    shape3d res_shape3d = { .x = conv.input_shape.x, .y = conv.input_shape.y, .z = 1 };
    // derivative loss
    starpu_data_handle_t res_dl_dinput = block_init(res_shape3d);

    ret = starpu_task_insert(&cl_sum_z_axis, STARPU_R, dl_dinput, STARPU_W, res_dl_dinput, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

    return res_dl_dinput;
}
