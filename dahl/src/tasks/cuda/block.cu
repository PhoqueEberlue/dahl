#include <starpu.h>
#include "./common.cuh"
#include "../../macros.h"

extern "C" void cuda_block_sum_z_axis(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_block_sum_y_axis(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_block_sum_xy_axes(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_block_add_padding(void* buffers[2], void* cl_arg)
{

}


extern "C" void cuda_block_zero(void *buffers[1], void *cl_arg)
{
    auto in = STARPU_BLOCK_GET(buffers[0]);
	cudaMemsetAsync((dahl_fp*)in.ptr, 0, in.ldy * in.nz * in.elemsize, 
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

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
