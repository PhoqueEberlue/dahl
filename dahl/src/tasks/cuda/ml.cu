#include <starpu.h>
#include "common.cuh"
#include "../../macros.h"
#include "../../../include/dahl_types.h"
#include "starpu_cuda.h"

static __global__ void check_predictions_batch(
    struct starpu_matrix_interface const pred,
    struct starpu_matrix_interface const targ,
    struct starpu_variable_interface const correct_predictions)
{
    auto const pred_p = (const dahl_fp*)pred.ptr;
    auto const targ_p = (const dahl_fp*)targ.ptr;
    auto correct_p = (dahl_fp*)correct_predictions.ptr;

    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= pred.ny)
        return;

    // Find argmax across x dimension
    dahl_fp max_val = pred_p[y * pred.ld];
    size_t max_index = 0;

    for (size_t x = 1; x < pred.nx; x++)
    {
        dahl_fp val = pred_p[(y * pred.ld) + x];
        if (val > max_val)
        {
            max_val = val;
            max_index = x;
        }
    }

    // Compare to target
    if (targ_p[(y * targ.ld) + max_index] == 1)
        atomicAdd(correct_p, 1.0F);
}

extern "C" void cuda_check_predictions_batch(void* buffers[3], void* cl_arg)
{
    auto pred = STARPU_MATRIX_GET(buffers[0]);
    auto targ = STARPU_MATRIX_GET(buffers[1]);
    auto correct_predictions = STARPU_VARIABLE_GET(buffers[2]);

    // Zero initialize result on device
    dahl_fp zero = 0;
    cudaMemcpyAsync((void*)correct_predictions.ptr, &zero, sizeof(dahl_fp),
                    cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());

    // Configure 1 thread per batch row
    unsigned int threads = 256;
    unsigned int blocks = (pred.ny + threads - 1) / threads;

    check_predictions_batch<<<blocks, threads, 0, starpu_cuda_get_local_stream()>>>(
        pred, targ, correct_predictions);

    dahl_cuda_check_error_and_sync();
}

static __global__ void cross_entropy_loss_batch(
    struct starpu_matrix_interface const pred,
    struct starpu_matrix_interface const targ,
    struct starpu_variable_interface const out)
{
    auto const pred_p = (const dahl_fp*)pred.ptr;
    auto const targ_p = (const dahl_fp*)targ.ptr;
    auto out_p  = (dahl_fp*)out.ptr;

    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= pred.ny)
        return;

    // Find max value in the prediction batch
    dahl_fp max_pred = pred_p[y * pred.ld];
    for (size_t x = 1; x < pred.nx; x++)
    {
        dahl_fp val = pred_p[y * pred.ld + x];
        if (val > max_pred)
            max_pred = val;
    }

    // Compute log-sum-exp
    dahl_fp sum_exp = 0.0;
    for (size_t x = 0; x < pred.nx; x++)
        sum_exp += exp(pred_p[(y * pred.ld) + x] - max_pred);

    dahl_fp log_sum_exp = log(sum_exp);

    // Finding the index of the true class because targ is in one-hot format
    size_t true_idx = 0;
    for (size_t x = 0; x < targ.nx; x++)
    {
        if (targ_p[(y * targ.ld) + x] == 1.0)
        {
            true_idx = x;
            break;
        }
    }

    // Log probability of the true class
    dahl_fp log_prob = pred_p[(y * pred.ld) + true_idx] - max_pred - log_sum_exp;
    dahl_fp loss = -log_prob / (dahl_fp)pred.ny;

    // Accumulate loss (negative log likelihood)
    atomicAdd(out_p, loss);
}

extern "C" void cuda_cross_entropy_loss_batch(void* buffers[3], void* cl_arg)
{
    auto pred = STARPU_MATRIX_GET(buffers[0]);
    auto targ = STARPU_MATRIX_GET(buffers[1]);
    auto out  = STARPU_VARIABLE_GET(buffers[2]);

    // Zero the output loss accumulator on device
    dahl_fp zero = 0.0;
    cudaMemcpyAsync((void*)out.ptr, &zero, sizeof(dahl_fp),
                    cudaMemcpyHostToDevice, starpu_cuda_get_local_stream());

    // Configure launch: 1 thread per sample
    unsigned int threads = 256;
    unsigned int blocks = (pred.ny + threads - 1) / threads;

    cross_entropy_loss_batch<<<blocks, threads, 0, starpu_cuda_get_local_stream()>>>(
        pred, targ, out);

    dahl_cuda_check_error_and_sync();
}

static __global__ void cross_entropy_loss_gradient_batch(
    struct starpu_matrix_interface const pred,
    struct starpu_matrix_interface const targ,
    struct starpu_matrix_interface const out)
{
    auto const pred_p = (const dahl_fp*)pred.ptr;
    auto const targ_p = (const dahl_fp*)targ.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= pred.ny)
        return;

    const size_t num_classes = pred.nx;
    const size_t ld = pred.ld;
    const dahl_fp inv_batch = 1.0 / (dahl_fp)pred.ny;

    // Find max value in the prediction batch
    dahl_fp max_pred = pred_p[y * ld];
    for (size_t x = 1; x < num_classes; x++)
    {
        dahl_fp val = pred_p[y * ld + x];
        if (val > max_pred)
            max_pred = val;
    }

    // Compute denominator of softmax
    dahl_fp sum_exp = 0.0;
    for (size_t x = 0; x < num_classes; x++)
        sum_exp += exp(pred_p[(y * ld) + x] - max_pred);

    // Softmax probabilities and gradient
    for (size_t x = 0; x < num_classes; x++)
    {
        dahl_fp p = exp(pred_p[(y * ld) + x] - max_pred) / sum_exp;
        out_p[(y * ld) + x] = p * inv_batch;
    }

    // Finding the index of the true class because targ is in one-hot format
    size_t true_idx = 0;
    for (size_t x = 0; x < num_classes; x++)
    {
        if (targ_p[(y * ld) + x] == 1.0)
        {
            true_idx = x;
            break;
        }
    }

    out_p[(y * ld) + true_idx] -= inv_batch;
}

extern "C" void cuda_cross_entropy_loss_gradient_batch(void* buffers[3], void* cl_arg)
{
    auto pred = STARPU_MATRIX_GET(buffers[0]);
    auto targ = STARPU_MATRIX_GET(buffers[1]);
    auto out  = STARPU_MATRIX_GET(buffers[2]);

    dim3 block(256);
    dim3 grid((pred.ny + block.x - 1) / block.x);

    cross_entropy_loss_gradient_batch<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(
        pred, targ, out);

    dahl_cuda_check_error_and_sync();
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

    // TODO: remove += when starpu redux cuda is fixed
    // Set the corresponding value for index i,j,k
    out_p[(k * out.ldz) + (j * out.ldy) + i] += cell_res;
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
