#include <cstdio>
#include <starpu.h>
#include "../../../include/dahl_types.h"
#include "../../macros.h"
#include "common.cuh"

static __global__ void any_relu(size_t nb_elem, dahl_fp const* in, dahl_fp* out)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;

    if (in[index] < 0.0F)
        out[index] = 0.0F;
    else 
        out[index] = in[index];
}

extern "C" void cuda_any_relu(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    auto in = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[0]);
    auto out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);
    
    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_relu<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, in, out);
    dahl_cuda_check_error_and_sync();
}

static __global__ void any_relu_backward(
        size_t nb_elem,
        dahl_fp const* input, dahl_fp const* gradients, dahl_fp* out)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;

    if (input[index] > 0.0F)
        out[index] = gradients[index];
    else 
        out[index] = 0.0F;
}

extern "C" void cuda_any_relu_backward(void* buffers[3], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    auto input = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[0]);
    auto gradients = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[1]);
    auto out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[2]);

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_relu_backward<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(
            nb_elem, input, gradients, out);
    dahl_cuda_check_error_and_sync();
}


extern "C" void cuda_any_scal(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_any_power(void* buffers[2], void* cl_arg)
{

}

static __global__ void any_sub(
        size_t nb_elem,
        dahl_fp const* a, dahl_fp const* b, dahl_fp* c)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;
    c[index] = a[index] - b[index];
}

extern "C" void cuda_any_sub(void* buffers[3], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    auto a = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[0]);
    auto b = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[1]);
    auto c = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[2]);

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_sub<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, a, b, c);
    dahl_cuda_check_error_and_sync();
}

static __global__ void any_add(
        size_t nb_elem,
        dahl_fp const* a, dahl_fp const* b, dahl_fp* c)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;
    c[index] = a[index] + b[index];
}

extern "C" void cuda_any_add(void* buffers[3], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    auto a = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[0]);
    auto b = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[1]);
    auto c = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[2]);

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_add<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, a, b, c);
    dahl_cuda_check_error_and_sync();
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
