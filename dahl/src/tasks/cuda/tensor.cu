#include <starpu.h>
#include <stddef.h>
#include "../../../include/dahl_types.h"
#include "starpu/1.4/starpu_cuda.h"

static __global__ void tensor_sum_t_axis(
        size_t in_nx, size_t in_ny, size_t in_nz, size_t in_nt,
        size_t in_ldy, size_t in_ldz, size_t in_ldt,
        size_t out_ldy, size_t out_ldz,
        dahl_fp const* in, dahl_fp* out)
{
    // Compute 3D thread index
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= in_nx || y >= in_ny || z >= in_nz)
        return;

    dahl_fp sum = 0.0;

    // Sum along t-axis
    for (int t = 0; t < in_nt; t++)
    {
        sum += in[(t * in_ldt) + (z * in_ldz) + (y * in_ldy) + x];
    }

    out[(z * out_ldz) + (y * out_ldy) + x] += sum;
}

extern "C" void cuda_tensor_sum_t_axis(void* buffers[2], void* cl_arg)
{
    size_t const in_nx = STARPU_TENSOR_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_TENSOR_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_TENSOR_GET_NZ(buffers[0]);
    size_t const in_nt = STARPU_TENSOR_GET_NT(buffers[0]);
    size_t const in_ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
    size_t const in_ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
    size_t const in_ldt = STARPU_TENSOR_GET_LDT(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_TENSOR_GET_PTR(buffers[0]);

    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    size_t const out_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const out_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    dim3 block(8, 8, 8);
    dim3 grid((in_nx + block.x - 1) / block.x,
              (in_ny + block.y - 1) / block.y,
              (in_nz + block.z - 1) / block.z);

    tensor_sum_t_axis<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(
            in_nx, in_ny, in_nz, in_nt,
            in_ldy, in_ldz, in_ldt,
            out_ldy, out_ldz,
            in, out);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}


extern "C" void cuda_tensor_sum_xyt_axes(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_tensor_zero(void *buffers[1], void *cl_arg)
{

}


extern "C" void cuda_tensor_accumulate(void *buffers[2], void *cl_arg)
{

}
