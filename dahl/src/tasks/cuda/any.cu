#include <cmath>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <starpu.h>
#include <stdio.h>
#include <unistd.h>
#include "../../../include/dahl_types.h"
#include "../../macros.h"
#include "common.cuh"

#define DEFINE_ANY_OPERATION_ABC(func_name, operation)                    \
    static __global__ void func_name(                                     \
            size_t n, const dahl_fp* a, const dahl_fp* b, dahl_fp* c)     \
    {                                                                     \
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;                 \
        if (i < n) operation(a[i], b[i], c[i]);                           \
    }                                                                     \
                                                                          \
    extern "C" void cuda_##func_name(void* buffers[3], void* cl_arg)      \
    {                                                                     \
        size_t nb_elem;                                                   \
        starpu_codelet_unpack_args(cl_arg, &nb_elem);                     \
                                                                          \
        auto a = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[0]);          \
        auto b = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[1]);          \
        auto c = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[2]);                \
                                                                          \
        int threads = 256;                                                \
        int blocks  = (nb_elem + threads - 1) / threads;                  \
                                                                          \
        func_name<<<blocks, threads, 0, starpu_cuda_get_local_stream()>>> \
            (nb_elem, a, b, c);                                           \
        dahl_cuda_check_error_and_sync();                                 \
    }

DEFINE_ANY_OPERATION_ABC(any_add, OPERATION_ADD);
DEFINE_ANY_OPERATION_ABC(any_sub, OPERATION_SUB);
DEFINE_ANY_OPERATION_ABC(any_mul, OPERATION_MUL);
DEFINE_ANY_OPERATION_ABC(any_div, OPERATION_DIV);

// Define execution functions for ANY type for a given operation that takes 2 arguments, in and out.
#define DEFINE_ANY_OPERATION_IN_OUT(func_name, operation)                             \
    static __global__ void func_name(size_t nb_elem, dahl_fp const* in, dahl_fp* out) \
    {                                                                                 \
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;                         \
        if (index < nb_elem) operation(in[index], out[index]);                        \
    }                                                                                 \
                                                                                      \
    extern "C" void cuda_##func_name(void* buffers[2], void* cl_arg)                  \
    {                                                                                 \
        size_t nb_elem;                                                               \
        starpu_codelet_unpack_args(cl_arg, &nb_elem);                                 \
                                                                                      \
        auto in = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[0]);                     \
        auto out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);                          \
                                                                                      \
        int threadsPerBlock = 256;                                                    \
        int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;            \
                                                                                      \
        func_name<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>  \
            (nb_elem, in, out);                                                       \
        dahl_cuda_check_error_and_sync();                                             \
    }

DEFINE_ANY_OPERATION_IN_OUT(any_add_self, OPERATION_ADD_SELF);
DEFINE_ANY_OPERATION_IN_OUT(any_sub_self, OPERATION_SUB_SELF);

static __global__ void any_relu(size_t nb_elem, dahl_fp const* in, dahl_fp* mask, dahl_fp* out)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;

    if (in[index] < 0.0F)
    {
        mask[index] = 0;
        out[index] = 0;
    }
    else
    {
        mask[index] = 1;
        out[index] = in[index];
    }
}

extern "C" void cuda_any_relu(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    auto in = (dahl_fp const*)STARPU_ANY_GET_PTR(buffers[0]);
    auto mask = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);
    auto out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[2]);
    
    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_relu<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, in, mask, out);
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
