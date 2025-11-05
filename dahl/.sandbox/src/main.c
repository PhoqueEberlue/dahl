#include "starpu_data.h"
#include "starpu_data_interfaces.h"
#include <starpu.h>
#include <stdint.h>
#include <stdio.h>
#include "cpu.h"

extern void cuda_tensor_accumulate(void *buffers[2], void *cl_arg);
                                                                      
static struct starpu_codelet cl_tensor_accumulate = {
    .cpu_funcs = { cpu_tensor_accumulate },
    .cuda_funcs = { cuda_tensor_accumulate },                               
    .cuda_flags = { STARPU_CUDA_ASYNC },                              
    .nbuffers = 2,                                          
    .modes = { STARPU_RW|STARPU_COMMUTE, STARPU_R },                                         
};

extern void cuda_tensor_zero(void *buffers[1], void *cl_arg);
                                                                      
static struct starpu_codelet cl_tensor_zero = {
    .cpu_funcs = { cpu_tensor_zero },
    .cuda_funcs = { cuda_tensor_zero },                               
    .cuda_flags = { STARPU_CUDA_ASYNC },                              
    .nbuffers = 1,                                          
    .modes = { STARPU_W },                                         
};

extern void cuda_any_add(void *buffers[3], void *cl_arg);
                                                                      
static struct starpu_codelet cl_any_add = {
    .cpu_funcs = { cpu_any_add },
    .cuda_funcs = { cuda_any_add },                               
    .cuda_flags = { STARPU_CUDA_ASYNC },                              
    .nbuffers = 3,                                          
    .modes = { STARPU_R, STARPU_R, STARPU_REDUX },                                         
};

// Performs element wise C = A + B, no matter the starpu data type 
void task_add(starpu_data_handle_t a, starpu_data_handle_t b, starpu_data_handle_t c, size_t nb_elem)
{
    int ret = starpu_task_insert(&cl_any_add,
                                 STARPU_VALUE, &nb_elem, sizeof(nb_elem),
                                 STARPU_R, a,
                                 STARPU_R, b,
                                 STARPU_REDUX, c, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

int main()
{
    int ret = starpu_init(nullptr);
    if (ret != 0)
    { 
        return 1;
    }

    starpu_cuda_set_device(0);

    starpu_data_handle_t handle_a = nullptr;
    double tensor_a[4] = { 1, 2, 3, 4 };
    starpu_data_handle_t handle_b = nullptr;
    double tensor_b[4] = { 4, 3, 2, 1 };
    starpu_data_handle_t handle_c = nullptr;
    double tensor_c[4] = { 10, 0, 0, 15 };
    starpu_data_handle_t handle_d = nullptr;
    double tensor_d[4] = { 0, 15, 10, 0 };
    starpu_data_handle_t handle_e = nullptr;
    double tensor_e[4] = { 0, 0, 0, 0 };

    starpu_tensor_data_register(&handle_a, STARPU_MAIN_RAM, (uintptr_t)&tensor_a, 4, 4, 4, 4, 1, 1, 1, sizeof(double));
    starpu_tensor_data_register(&handle_b, STARPU_MAIN_RAM, (uintptr_t)&tensor_b, 4, 4, 4, 4, 1, 1, 1, sizeof(double));
    starpu_tensor_data_register(&handle_c, STARPU_MAIN_RAM, (uintptr_t)&tensor_c, 4, 4, 4, 4, 1, 1, 1, sizeof(double));
    starpu_tensor_data_register(&handle_d, STARPU_MAIN_RAM, (uintptr_t)&tensor_d, 4, 4, 4, 4, 1, 1, 1, sizeof(double));
    starpu_tensor_data_register(&handle_e, STARPU_MAIN_RAM, (uintptr_t)&tensor_e, 4, 4, 4, 4, 1, 1, 1, sizeof(double));
    starpu_data_set_reduction_methods(handle_e, &cl_tensor_accumulate, &cl_tensor_zero);

    // CUDA version only accumulates the result from the last task of the handle
    task_add(handle_a, handle_b, handle_e, 4);
    task_add(handle_c, handle_d, handle_e, 4);

    // Expected:
    // {1,2,3,4}+{4,3,2,1} -> handle_e
    // {10,0,0,15}+{0,15,10,0} -> handle_e
    // Then redux
    // {15, 20, 15, 20}
    //
    // What I get:
    // {10, 15, 10, 15 }
    //
    // If we swap task order:
    // {5,5,5,5}

    starpu_data_acquire(handle_e, STARPU_R);

    for (size_t i = 0; i < 4; i++)
    {
        printf("%f ", tensor_e[i]);
    }
    printf("\n");

    starpu_data_release(handle_e);
}
