// #include "cuda_usage_example.h"
// #include "starpu_cuda.h"
// #include <starpu.h>
// #include <stdio.h>
// 
// extern void scal_cpu_func(void *buffers[], void *_args);
// extern void scal_cuda_func(void *buffers[], void *_args);
// 
// static struct starpu_perfmodel vector_scal_model =
// {
// 	.type = STARPU_HISTORY_BASED,
// 	.symbol = "vector_scal_model"
// };
// 
// static struct starpu_codelet cl =
// {
// 	// It is also possible to define CPU function in the same codelet
// 	//.cpu_funcs = {scal_cpu_func},
// 	//.cpu_funcs_name = {"scal_cpu_func"},
// 	// CUDA implementation of the codelet
// 	.cuda_funcs = {scal_cuda_func},
// 	.cuda_flags = {STARPU_CUDA_ASYNC},
// 	.nbuffers = 1,
// 	.modes = { STARPU_RW },
// 	.model = &vector_scal_model
// };
// 
// void scal_example(int nx)
// {
//     float *vector = (float*) malloc(nx*sizeof(float));
// 
//     printf("----------------------------------\n");
//     printf("Original vector:\n[");
//     for (int i = 0; i < nx; i++) {
//         vector[i] = 1.0;
//         printf("%f, ", vector[i]);
//     }
//     printf("]\n----------------------------------\n");
// 
//     starpu_cuda_set_device(0);
// 
// 	/* Tell StaPU to associate the "vector" vector with the "vector_handle"
// 	 * identifier. When a task needs to access a piece of data, it should
// 	 * refer to the handle that is associated to it.
// 	 * In the case of the "vector" data interface:
// 	 *  - the first argument of the registration method is a pointer to the
// 	 *    handle that should describe the data
// 	 *  - the second argument is the memory node where the data (ie. "vector")
// 	 *    resides initially: STARPU_MAIN_RAM stands for an address in main memory, as
// 	 *    opposed to an address on a GPU for instance.
// 	 *  - the third argument is the address of the vector in RAM
// 	 *  - the fourth argument is the number of elements in the vector
// 	 *  - the fifth argument is the size of each element.
// 	 */
// 	starpu_data_handle_t vector_handle;
// 	starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, nx, sizeof(vector[0]));
// 
// 	float factor = 3.14;
// 
// 	/* create a synchronous task: any call to starpu_task_block_submit will block
// 	 * until it is terminated */
// 	struct starpu_task *task = starpu_task_create();
// 	task->synchronous = 1;
// 
// 	task->cl = &cl;
// 
// 	/* the codelet manipulates one buffer in RW mode */
// 	task->handles[0] = vector_handle;
// 
// 	/* an argument is passed to the codelet, beware that this is a
// 	 * READ-ONLY buffer and that the codelet may be given a pointer to a
// 	 * COPY of the argument */
// 	task->cl_arg = &factor;
// 	task->cl_arg_size = sizeof(factor);
// 
// 	/* execute the task on any eligible computational resource */
// 	int ret = starpu_task_block_submit(task);
// 	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
// 
//     starpu_task_wait_for_all();
// 
//     starpu_data_acquire(vector_handle, STARPU_R);
// 
//     printf("----------------------------------\n");
//     printf("Scaled up vector with CUDA:\n[");
//     for (int i = 0; i < nx; i++) {
//         printf("%f, ", vector[i]);
//     }
//     printf("]\n----------------------------------\n");
// 
//     starpu_data_release(vector_handle);
// }
