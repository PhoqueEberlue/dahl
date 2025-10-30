#include <starpu.h>

// Get the ptr of any StarPU data type. Does not perform any check.
// This works because ptr is always the second field in the struct for vector, matrix, block and tensor,
// so it does not matter what we cast `interface` into. 
// This may be risky though, especially if the field order changes...
#define STARPU_ANY_GET_PTR(interface) (((struct starpu_vector_interface *)(interface))->ptr)

extern "C" void cuda_any_relu(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_any_relu_backward(void* buffers[3], void* cl_arg)
{

}


extern "C" void cuda_any_scal(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_any_power(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_any_sub(void* buffers[3], void* cl_arg)
{

}


extern "C" void cuda_any_add(void* buffers[3], void* cl_arg)
{

}


extern "C" void cuda_any_add_value(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_any_clip(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_any_sum(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_any_mean(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_any_fill(void* buffers[1], void* cl_arg)
{

}


// For debug purposes
extern "C" void cuda_any_wait(void* buffers[1], void* cl_arg)
{

}


extern "C" void cuda_any_copy(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_any_min(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_any_max(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_any_round(void* buffers[2], void* cl_arg)
{

}
