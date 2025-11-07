#include <starpu.h>
#include "common.cuh"
#include "../../macros.h"
#include "../../../include/dahl_types.h"
#include "starpu_cuda.h"

extern "C" void cuda_check_predictions_batch(void* buffers[3], void* cl_arg)
{

}


extern "C" void cuda_cross_entropy_loss_batch(void* buffers[3], void* cl_arg)
{

}


extern "C" void cuda_cross_entropy_loss_gradient(void* buffers[3], void* cl_arg)
{

}

static __global__ void convolution_2d(
        struct starpu_block_interface const in,
        struct starpu_block_interface const ker,
        struct starpu_matrix_interface  const out)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto ker_p = (dahl_fp const*)ker.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    // Compute (i, j) coordinates for this thread
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= out.nx || j >= out.ny)
        return;

    dahl_fp cell_res = 0.0F;

    // Convolution accumulation
    for (size_t m = 0; m < ker.nz; m++)
    {
        for (size_t l = 0; l < ker.ny; l++)
        {
            for (size_t k = 0; k < ker.nx; k++)
            {
                dahl_fp kernel_value = ker_p[(m * ker.ldz) + (l * ker.ldy) + k];
                dahl_fp in_value     = in_p[(m * in.ldz) + ((l + j) * in.ldy) + (k + i)];
                cell_res += in_value * kernel_value;
            }
        }
    }

    // Write output
    out_p[(j * out.ld) + i] = cell_res;
}

extern "C" void cuda_convolution_2d(void* buffers[3], void* cl_arg)
{
    auto in = STARPU_BLOCK_GET(buffers[0]);
    auto ker = STARPU_BLOCK_GET(buffers[1]);
    auto out = STARPU_MATRIX_GET(buffers[2]);

    dim3 block(16, 16);
    dim3 grid((out.nx + block.x - 1) / block.x,
              (out.ny + block.y - 1) / block.y);

    convolution_2d<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(in, ker, out);
    dahl_cuda_check_error_and_sync();
}

static __global__ void convolution_2d_backward_filters(
        struct starpu_block_interface const in,
        struct starpu_matrix_interface const ker,
        struct starpu_block_interface  const out)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto ker_p = (dahl_fp const*)ker.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    // 3D indexing: (i, j, k)
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;  // X
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;  // Y
    size_t k = blockIdx.z;                             // Z (channel)

    if (i >= out.nx || j >= out.ny || k >= out.nz)
        return;

    dahl_fp cell_res = 0.0F;

    // loop through l,m on axes x,y of the kernel
    for (size_t m = 0; m < ker.ny; m++)
    {
        for (size_t l = 0; l < ker.nx; l++)
        {
            dahl_fp kernel_value = ker_p[(m * ker.ld) + l];
            // Here we use k, the index on the z axis of the output, as input owns as many channels.
            // The kernel doesn't own a channel dimension in this function, so we ignore it.
            // Then we add the offset of the slidding window (i,j) to (l,m)
            // as they both correspond to (x,y).
            dahl_fp in_value = in_p[(k * in.ldz) + ((m + j) * in.ldy) + l + i];

            cell_res += in_value * kernel_value;
        }
    }

    // Set the corresponding value for index i,j,k
    out_p[(k * out.ldz) + (j * out.ldy) + i] = cell_res;
}

extern "C" void cuda_convolution_2d_backward_filters(void* buffers[3], void* cl_arg)
{
    auto in = STARPU_BLOCK_GET(buffers[0]);
    auto ker = STARPU_MATRIX_GET(buffers[1]);
    auto out = STARPU_BLOCK_GET(buffers[2]);

    dim3 block(16, 16);
    dim3 grid((out.nx + block.x - 1) / block.x,
              (out.ny + block.y - 1) / block.y,
               out.nz);

    convolution_2d_backward_filters<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(in, ker, out);
    dahl_cuda_check_error_and_sync();
}

// TODO: Skipping for now, padding free is just better
extern "C" void cuda_convolution_2d_backward_input(void* buffers[3], void* cl_arg)
{

}

// __device__ __forceinline__ size_t sub_sat(size_t a, size_t b)
// {
//     // Equivalent to: if (a >= b) return a - b; else return 0;
//     // Implemented with predication, no branching.
//     size_t diff = a - b;
//     return (a >= b) ? diff : 0;
// }

// Without predicates
__device__ __forceinline__ size_t sub_sat(size_t a, size_t b)
{
    size_t diff = a - b;
    // Generate mask = all ones if a >= b, else 0
    size_t mask = -(size_t)(a >= b);
    return diff & mask;
}

// See the CPU version for more informations
static __global__ void convolution_2d_backward_input_padding_free(
        struct starpu_matrix_interface const in,
        struct starpu_block_interface const ker,
        struct starpu_block_interface  const out)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto ker_p = (dahl_fp const*)ker.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    // 3D indexing: (i, j, k)
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;  // X
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;  // Y
    size_t k = blockIdx.z;                             // Z (channel)

    if (i >= out.nx || j >= out.ny || k >= out.nz)
        return;

    size_t const pad_nx = ker.nx - 1;
    size_t const pad_ny = ker.ny - 1;

    dahl_fp cell_res = 0.0F;

    // --- Compute valid region of overlap between input and kernel
    size_t y_start = sub_sat(j, pad_ny);
    size_t y_end = sub_sat(in.ny - 1, j);
    y_end = in.ny - y_end;
    size_t y_ker = sub_sat(ker.ny, y_end);

    for (size_t m = y_start; m < y_end; m++)
    {
        size_t x_start = sub_sat(i, pad_nx);
        size_t x_end = sub_sat(in.nx - 1, i);
        x_end = in.nx - x_end;
        size_t x_ker = sub_sat(ker.nx, x_end);

        for (size_t l = x_start; l < x_end; l++)
        {
            dahl_fp kernel_value =
                ker_p[(k * ker.ldz) + ((ker.ny - 1 - y_ker) * ker.ldy) + (ker.nx - 1 - x_ker)];
            dahl_fp in_value = in_p[(m * in.ld) + l];
            cell_res += in_value * kernel_value;
            x_ker++;
        }
        y_ker++;
    }

    out_p[(k * out.ldz) + (j * out.ldy) + i] = cell_res;
}

extern "C" void cuda_convolution_2d_backward_input_padding_free(void* buffers[3], void* cl_arg)
{
    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto ker = STARPU_BLOCK_GET(buffers[1]);
    auto out = STARPU_BLOCK_GET(buffers[2]);

    dim3 block(16, 16);
    dim3 grid((out.nx + block.x - 1) / block.x,
              (out.ny + block.y - 1) / block.y,
               out.nz);

    convolution_2d_backward_input_padding_free<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(in, ker, out);
    dahl_cuda_check_error_and_sync();
}
