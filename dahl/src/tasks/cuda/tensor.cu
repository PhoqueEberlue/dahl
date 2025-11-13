#include <starpu.h>
#include <stddef.h>
#include <stdio.h>
#include "../../macros.h"
#include "starpu_cuda.h"
#include "starpu_data_interfaces.h"
#include "./common.cuh"
#include "../../../include/dahl_data.h"

static __global__ void tensor_sum_t_axis(
        struct starpu_tensor_interface const in,
        struct starpu_block_interface  const out)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;

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

    dim3 block(8, 8, 8);
    dim3 grid((in.nx + block.x - 1) / block.x,
              (in.ny + block.y - 1) / block.y,
              (in.nz + block.z - 1) / block.z);

    tensor_sum_t_axis<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(in, out);
    dahl_cuda_check_error_and_sync();
}

static __global__ void tensor_sum_xyt_axes(
        struct starpu_tensor_interface const in,
        struct starpu_vector_interface out)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;

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

    size_t threadsPerBlock = 256;
    size_t blocks = (in.nz + threadsPerBlock - 1) / threadsPerBlock;

    tensor_sum_xyt_axes<<<blocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(in, out);
    dahl_cuda_check_error_and_sync();
}


extern "C" void cuda_tensor_zero(void *buffers[1], void *cl_arg)
{
    auto in = STARPU_TENSOR_GET(buffers[0]);
	cudaMemsetAsync((dahl_fp*)in.ptr, 0, in.ldt * in.nt * in.elemsize, 
            starpu_cuda_get_local_stream());
}

extern "C" void cuda_tensor_accumulate(void *buffers[2], void *cl_arg)
{
    auto dst = STARPU_TENSOR_GET(buffers[0]);
    auto src = STARPU_TENSOR_GET(buffers[1]);
    auto dst_p = (dahl_fp*)dst.ptr;
    auto src_p = (dahl_fp const*)src.ptr;

    size_t nb_elem = dst.nx * dst.ny * dst.nz * dst.nt;

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    cuda_accumulate<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, dst_p, src_p);
    dahl_cuda_check_error_and_sync();
}

static __global__ void tensor_print(struct starpu_tensor_interface const ts)
{
    auto tensor_p = (dahl_fp const*)ts.ptr;

    printf("tensor=%p nx=%llu ny=%llu nz=%llu nt=%llu ldy=%llu ldz=%llu ldt=%llu\n{\n", 
            (void*)ts.ptr, ts.nx, ts.ny, ts.nz, ts.nt, ts.ldy, ts.ldz, ts.ldt);
    for(size_t t = 0; t < ts.nt; t++)
    {
        printf("\t{\n");
        for(size_t z = 0; z < ts.nz; z++)
        {
            printf("\t\t{\n");
            for(size_t y = 0; y < ts.ny; y++)
            {
                printf("\t\t\t{ ");
                for(size_t x = 0; x < ts.nx; x++)
                {
                    dahl_fp value = 
                        tensor_p[(t * ts.ldt) + (z * ts.ldz) + (y * ts.ldy) + x];
                    printf("%f, ", value);
                }
                printf("},\n");
            }
            printf("\t\t},\n");
        }
        printf("\t},\n");
    }
	printf("}\n");
}

extern "C" void cuda_tensor_print(void *buffers[1], void *cl_arg)
{
    auto tensor = STARPU_TENSOR_GET(buffers[0]);
    tensor_print<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(tensor);
    dahl_cuda_check_error_and_sync();
}
