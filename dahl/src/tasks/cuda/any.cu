#include <cmath>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
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

static __global__ void any_scal(size_t nb_elem, dahl_fp const* in, dahl_fp* out, dahl_fp factor)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;
    out[index] = in[index] * factor;
}

extern "C" void cuda_any_scal(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp factor;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &factor);

    auto in = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[0]);
    auto out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_scal<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(
            nb_elem, in, out, factor);
    dahl_cuda_check_error_and_sync();

}

static __global__ void any_power(size_t nb_elem, dahl_fp const* in, dahl_fp* out, dahl_fp power)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;
    out[index] = pow(in[index], power);
}

extern "C" void cuda_any_power(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp power;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &power);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_power<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(
            nb_elem, in, out, power);
    dahl_cuda_check_error_and_sync();

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

static __global__ void any_add(size_t nb_elem, dahl_fp const* a, dahl_fp const* b, dahl_fp* c)
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

static __global__ void any_add_value(size_t nb_elem, dahl_fp const* in, dahl_fp* out, dahl_fp value)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;
    out[index] = in[index] + value;
}

extern "C" void cuda_any_add_value(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp value;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &value);

    auto in = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[0]);
    auto out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_add_value<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, in, out, value);
    dahl_cuda_check_error_and_sync();
}

static __global__ void any_clip(size_t nb_elem, dahl_fp const* in, dahl_fp* out, dahl_fp min, dahl_fp max)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;

    if (in[index] > max)
        out[index] = max;
    else if (in[index] < min)
        out[index] = min;
    else
        out[index] = in[index];
}

extern "C" void cuda_any_clip(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp min;
    dahl_fp max;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &min, &max);

    auto in = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[0]);
    auto out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_clip<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, in, out, min, max);
    dahl_cuda_check_error_and_sync();
}


// TODO: Does not make much sense to implement for cuda right?
extern "C" void cuda_any_sum(void* buffers[2], void* cl_arg)
{

}

// TODO: Does not make much sense to implement for cuda right?
extern "C" void cuda_any_mean(void* buffers[2], void* cl_arg)
{

}

static __global__ void any_fill(size_t nb_elem, dahl_fp* buf, dahl_fp value)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;
    buf[index] = value;
}

extern "C" void cuda_any_fill(void* buffers[1], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp value;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &value);

    auto buf = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_fill<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, buf, value);
    dahl_cuda_check_error_and_sync();
}


// For debug purposes
extern "C" void cuda_any_wait(void* buffers[1], void* cl_arg)
{

}


extern "C" void cuda_any_copy(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    cudaMemcpy(out, in, nb_elem * sizeof(dahl_fp), cudaMemcpyKind::cudaMemcpyDefault);
}


// TODO: Does not make much sense to implement for cuda right?
extern "C" void cuda_any_min(void* buffers[2], void* cl_arg)
{

}


// TODO: Does not make much sense to implement for cuda right?
extern "C" void cuda_any_max(void* buffers[2], void* cl_arg)
{

}

static __global__ void any_round(size_t nb_elem, dahl_fp const* in, dahl_fp* out, int8_t precision)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;
    dahl_fp power = pow(10.0F, precision);
    out[index] = round(in[index] * power) / power;
}

extern "C" void cuda_any_round(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    int8_t precision;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &precision);

    auto in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    auto out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_round<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, in, out, precision);
    dahl_cuda_check_error_and_sync();

}
