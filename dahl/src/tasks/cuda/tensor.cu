#include <starpu.h>
#include <stddef.h>
#include "../macros.h"
#include "../../../include/dahl_types.h"
#include "starpu_cuda.h"
#include "starpu_data_interfaces.h"

static __global__ void tensor_sum_t_axis(
        struct starpu_tensor_interface const in,
        struct starpu_block_interface  const out,
        dahl_fp const* in_p, dahl_fp* out_p)
{
    // Compute 3D thread index
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= in.nx || y >= in.ny || z >= in.nz)
        return;

    dahl_fp sum = 0.0;

    // Sum along t-axis
    for (int t = 0; t < in.nt; t++)
    {
        sum += in_p[(t * in.ldt) + (z * in.ldz) + (y * in.ldy) + x];
    }

    out_p[(z * out.ldz) + (y * out.ldy) + x] += sum;
}

extern "C" void cuda_tensor_sum_t_axis(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_TENSOR_GET(buffers[0]);
    auto out = STARPU_BLOCK_GET(buffers[1]);
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    dim3 block(8, 8, 8);
    dim3 grid((in.nx + block.x - 1) / block.x,
              (in.ny + block.y - 1) / block.y,
              (in.nz + block.z - 1) / block.z);

    tensor_sum_t_axis<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(
            in, out, in_p, out_p);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

static __global__ void tensor_sum_xyt_axes(
        struct starpu_tensor_interface const in,
        dahl_fp const* in_p, dahl_fp* out_p)
{
    size_t z = blockIdx.x * blockDim.x + threadIdx.x;

    if (z >= in.nz) return;

    dahl_fp sum = 0.0;

    for (int t = 0; t < in.nt; t++)
    {
        for (int y = 0; y < in.ny; y++)
        {
            for (int x = 0; x < in.nx; x++)
            {
                sum += in_p[(t * in.ldt) + (z * in.ldz) + (y * in.ldy) + x];
            }
        }
    }

    out_p[z] += sum;
}

extern "C" void cuda_tensor_sum_xyt_axes(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_TENSOR_GET(buffers[0]);
    auto out = STARPU_VECTOR_GET(buffers[1]);
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    size_t threadsPerBlock = 256;
    size_t blocks = (in.nz + threadsPerBlock - 1) / threadsPerBlock;

    tensor_sum_xyt_axes<<<blocks, threadsPerBlock>>>(in, in_p, out_p);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}


extern "C" void cuda_tensor_zero(void *buffers[1], void *cl_arg)
{

}


extern "C" void cuda_tensor_accumulate(void *buffers[2], void *cl_arg)
{

}
