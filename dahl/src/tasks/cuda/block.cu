#include <starpu.h>
#include <stdio.h>
#include "./common.cuh"
#include "../../macros.h"

static __global__ void block_sum_z_axis(
        struct starpu_block_interface const in,
        struct starpu_matrix_interface  const out)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.nx || y >= in.ny) return;

    dahl_fp sum = 0.0;

    for (size_t z = 0; z < in.nz; z++)
    {
        sum += in_p[(z * in.ldz) + (y * in.ldy) + x];
    }

    out_p[(y * out.ld) + x] += sum;
}

extern "C" void cuda_block_sum_z_axis(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_BLOCK_GET(buffers[0]);
    auto out = STARPU_MATRIX_GET(buffers[1]);

    dim3 block(16, 16);  // 256 threads per block
    dim3 grid((in.nx + block.x - 1) / block.x,
              (in.ny + block.y - 1) / block.y);

    block_sum_z_axis<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(in, out);
    dahl_cuda_check_error_and_sync();
}

static __global__ void block_sum_y_axis(
        struct starpu_block_interface const in,
        struct starpu_matrix_interface const out)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.nx || z >= in.nz) return;

    dahl_fp sum = 0.0;

    for (size_t y = 0; y < in.ny; y++)
    {
        sum += in_p[(z * in.ldz) + (y * in.ldy) + x];
    }

    out_p[(z * out.ld) + x] += sum;
}

extern "C" void cuda_block_sum_y_axis(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_BLOCK_GET(buffers[0]);
    auto out = STARPU_MATRIX_GET(buffers[1]);

    dim3 block(16, 16);  // 256 threads per block
    dim3 grid((in.nx + block.x - 1) / block.x,
              (in.nz + block.z - 1) / block.z);

    block_sum_y_axis<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(in, out);
    dahl_cuda_check_error_and_sync();
}

static __global__ void block_sum_xy_axes(
        struct starpu_block_interface const in,
        struct starpu_vector_interface const out)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z;  // One grid layer per z

    if (x >= in.nx || y >= in.ny || z >= in.nz) return;

    // Compute index in input tensor
    size_t index = (z * in.ldz) + (y * in.ldy) + x;

    // Each thread contributes one value to out[z]
    atomicAdd(&out_p[z], in_p[index]);
}

extern "C" void cuda_block_sum_xy_axes(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_BLOCK_GET(buffers[0]);
    auto out = STARPU_VECTOR_GET(buffers[1]);

    dim3 block(16, 16);
    dim3 grid((in.nx + block.x - 1) / block.x,
              (in.ny + block.y - 1) / block.y,
               in.nz);

    block_sum_xy_axes<<<grid, block>>>(in, out);
    dahl_cuda_check_error_and_sync();
}

// TODO: Does not make much sense to implement for cuda right?
extern "C" void cuda_block_add_padding(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_block_zero(void *buffers[1], void *cl_arg)
{
    auto in = STARPU_BLOCK_GET(buffers[0]);
	cudaMemsetAsync((dahl_fp*)in.ptr, 0, in.ldz * in.nz * in.elemsize, 
            starpu_cuda_get_local_stream());
}

extern "C" void cuda_block_accumulate(void *buffers[2], void *cl_arg)
{
    auto dst = STARPU_BLOCK_GET(buffers[0]);
    auto src = STARPU_BLOCK_GET(buffers[1]);
    auto dst_p = (dahl_fp*)dst.ptr;
    auto src_p = (dahl_fp const*)src.ptr;

    size_t nb_elem = dst.nx * dst.ny * dst.nz;

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    cuda_accumulate<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, dst_p, src_p);
    dahl_cuda_check_error_and_sync();
}

static __global__ void block_print(struct starpu_block_interface const bl)
{
    auto block_p = (dahl_fp const*)bl.ptr;

    printf("block=%p nx=%llu ny=%llu nz=%llu ldy=%llu ldz=%llu\n{\n", 
            (void*)bl.ptr, bl.nx, bl.ny, bl.nz, bl.ldy, bl.ldz);
    for(size_t z = 0; z < bl.nz; z++)
    {
        printf("\t{\n");
        for(size_t y = 0; y < bl.ny; y++)
        {
            printf("\t\t{ ");
            for(size_t x = 0; x < bl.nx; x++)
            {
                dahl_fp value = block_p[(z * bl.ldz) + (y * bl.ldy) + x];
                printf("%f, ", value);
            }
            printf("},\n");
        }
        printf("\t},\n");
    }
    printf("}\n");
}

extern "C" void cuda_block_print(void *buffers[1], void *cl_arg)
{
    auto block = STARPU_BLOCK_GET(buffers[0]);
    block_print<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(block);
    dahl_cuda_check_error_and_sync();
}
