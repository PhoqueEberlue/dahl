#include "../codelets.h"
#include "starpu_data_interfaces.h"
#include "starpu_task_util.h"
#include "../../../include/dahl_types.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>

void check_predictions_batch(void* buffers[3], void* cl_arg)
{
    size_t const pred_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const pred_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const pred_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* pred = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Targets vector
    size_t const targ_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const targ_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const targ_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* targ = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    dahl_fp* correct_predictions = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[2]);

    assert(pred_nx == targ_nx);
    assert(pred_ny == targ_ny);

    int count = 0;
    // Loop through each batch
    for (size_t y = 0; y < pred_ny; y++)
    {
        // Take the first prediction of this batch
        dahl_fp max_val = pred[(y * pred_ld)];
        size_t max_index = 0;

        for (size_t x = 0; x < pred_nx; x++)
        {
            dahl_fp current_value = pred[(y * pred_ld) + x];

            if (current_value > max_val)
            {
                max_val = current_value;
                max_index = x;
            }
        }

        if (targ[(y * targ_ld) + max_index] == 1)
        {
            count++;
        }
    }

    *correct_predictions = count;
}

void cross_entropy_loss_batch(void* buffers[3], void* cl_arg)
{
    // Predictions batch
    size_t const pred_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const pred_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const pred_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* pred = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Targets batch
    size_t const targ_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const targ_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const targ_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* targ = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    // Output scalar
    dahl_fp* out = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[2]);

    assert(pred_nx == targ_nx);
    assert(pred_ny == targ_ny);
    assert(pred_ld == targ_ld);

    dahl_fp batch_loss = 0.0F;

    // Loop through batch
    for (size_t y = 0; y < pred_ny; y++) {

        dahl_fp max_pred = pred[0];
        // Find max value in the prediction batch
        for (size_t x = 0; x < pred_nx; x++) {
            if (pred[(y * pred_ld) + x] > max_pred)
            {
                max_pred = pred[(y * pred_ld) + x];
            }
        }

        // Compute log-sum-exp
        dahl_fp sum_exp = 0.0F;
        for (size_t x = 0; x < pred_nx; x++) {
            sum_exp += exp(pred[(y * pred_ld) + x] - max_pred);
        }

        dahl_fp log_sum_exp = log(sum_exp);

        size_t index = 0;
        // Finding the index of the true class because targ is in one-hot format
        for (size_t x = 0; x < targ_nx; x++)
        {
            if (targ[(y * targ_ld) + x] == 1.0F)
            {
                index = x;
                continue;
            }
        }

        // Log probability of the true class
        dahl_fp log_prob = pred[(y * pred_ld) + index] - max_pred - log_sum_exp;

        // Accumulate loss (negative log likelihood)
        batch_loss -= log_prob;
    }

    // Average over batch
    *out = batch_loss / (dahl_fp)pred_ny;
}

void cross_entropy_loss_gradient_batch(void* buffers[3], void* cl_arg)
{
    // Predictions batch
    size_t const pred_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const pred_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const pred_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* pred = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Targets batch
    size_t const targ_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const targ_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const targ_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* targ = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    // Output by batch
    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[2]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[2]);

    assert(pred_nx == targ_nx);
    assert(pred_ny == targ_ny);
    assert(pred_nx == out_nx);
    assert(pred_ny == out_ny);
    assert(pred_ld == targ_ld);
    assert(pred_ld == out_ld);

    // Batch values are on y dimension, x contains the predictions per class
    size_t const batch_size = pred_ny;
    size_t const num_classes = pred_nx;

    // Loop through batch
    for (int y = 0; y < batch_size; y++) {

        dahl_fp max_pred = pred[(y * pred_ld)];
        // Find max value in the prediction batch
        for (size_t x = 0; x < num_classes; x++) {
            if (pred[(y * pred_ld) + x] > max_pred)
            {
                max_pred = pred[(y * pred_ld) + x];
            }
        }

        // Compute denominator of softmax
        dahl_fp sum_exp = 0.0F;
        for (size_t x = 0; x < num_classes; x++) {
            sum_exp += exp(pred[(y * pred_ld) + x] - max_pred);
        }

        // Softmax probabilities and gradient
        for (size_t x = 0; x < num_classes; x++) {
            dahl_fp p = exp(pred[(y * pred_ld) + x] - max_pred) / sum_exp;
            out[(y * out_ld) + x] = p / (float)batch_size;
        }

        size_t index = 0;
        // Finding the index of the true class because targ is in one-hot format
        for (size_t x = 0; x < num_classes; x++)
        {
            if (targ[(y * targ_ld) + x] == 1.0F)
            {
                index = x;
                continue;
            }
        }

        // Subtract 1 for the true class
        out[(y * out_ld) + index] -= 1.0F / (dahl_fp)batch_size;
    }
}

void convolution_2d(void* buffers[3], void* cl_arg)
{
    // Input block (because the image can have multiple channels)
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const in_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const in_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Kernel block
    size_t const k_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const k_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const k_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    size_t const k_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const k_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp const* kernel = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output matrix
    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[2]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[2]);

    assert(out_nx == in_nx - k_nx + 1);
    assert(out_ny == in_ny - k_ny + 1);
    assert(in_nz == k_nz);

    // loop through i,j on axes x,y of the output matrix
    for (size_t j = 0; j < out_ny; j++)
    {
        for (size_t i = 0; i < out_nx; i++)
        {
            dahl_fp cell_res = 0.0F;

            // loop through k,l,m on axes x,y,z of the kernel
            for (size_t m = 0; m < k_nz; m++)
            {
                for (size_t l = 0; l < k_ny; l++)
                {
                    for (size_t k = 0; k < k_nx; k++)
                    {
                        dahl_fp kernel_value = kernel[(m * k_ldz) + (l * k_ldy) + k];
                        // Here we add the offset of the slidding window (i,j) to (k,l)
                        // as they both correspond to (x,y).
                        dahl_fp in_value = in[(m * in_ldz) + ((l + j) * in_ldy) + k + i];
                        
                        cell_res += in_value * kernel_value;
                    }
                }
            }

            out[(j * out_ld) + i] = cell_res;
        }
    }
}

void convolution_2d_backward_filters(void* buffers[3], void* cl_arg)
{
    // Input block, here the orginal input of the forward pass
    size_t const in_nx = STARPU_BLOCK_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_BLOCK_GET_NY(buffers[0]);
    size_t const in_nz = STARPU_BLOCK_GET_NZ(buffers[0]);
    size_t const in_ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
    size_t const in_ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    // Kernel matrix, here the gradients output of the layer just after the convolution
    size_t const k_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const k_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const k_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* kernel = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    // Output block, here the loss derivative of the convolution filters
    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[2]);
    size_t const out_ldy = STARPU_BLOCK_GET_LDY(buffers[2]);
    size_t const out_ldz = STARPU_BLOCK_GET_LDZ(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

    assert(out_nx == in_nx - k_nx + 1);
    assert(out_ny == in_ny - k_ny + 1);
    assert(out_nz == in_nz);

    // loop through i,j,k on axes x,y,z of the output block
    for (size_t k = 0; k < out_nz; k++)
    {
        for (size_t j = 0; j < out_ny; j++)
        {
            for (size_t i = 0; i < out_nx; i++)
            {
                dahl_fp cell_res = 0.0F;

                // loop through l,m on axes x,y of the kernel
                for (size_t m = 0; m < k_ny; m++)
                {
                    for (size_t l = 0; l < k_nx; l++)
                    {
                        dahl_fp kernel_value = kernel[(m * k_ld) + l];
                        // Here we use k, the index on the z axis of the output, as input owns as many channels.
                        // The kernel doesn't own a channel dimension in this function, so we ignore it.
                        // Then we add the offset of the slidding window (i,j) to (l,m)
                        // as they both correspond to (x,y).
                        dahl_fp in_value = in[(k * in_ldz) + ((m + j) * in_ldy) + l + i];

                        cell_res += in_value * kernel_value;
                    }
                }

                // Set the corresponding value for index i,j,k
                out[(k * out_ldz) + (j * out_ldy) + i] += cell_res;
            }
        }
    }
}

// void convolution_2d_backward_input(void* buffers[3], void* cl_arg)
// {
//     // Input matrix, here the gradients output of the layer just after the convolution
//     size_t const in_nx = STARPU_MATRIX_GET_NX(buffers[0]);
//     size_t const in_ny = STARPU_MATRIX_GET_NY(buffers[0]);
//     size_t const in_ld = STARPU_MATRIX_GET_LD(buffers[0]);
//     dahl_fp const* in = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);
// 
//     // Kernel block, here the filters (weights) associated to the convolution
//     size_t const k_nx = STARPU_BLOCK_GET_NX(buffers[1]);
//     size_t const k_ny = STARPU_BLOCK_GET_NY(buffers[1]);
//     size_t const k_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
//     size_t const k_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
//     size_t const k_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
//     dahl_fp const* kernel = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);
// 
//     // Output block, here the loss derivative of the input
//     size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[2]);
//     size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[2]);
//     size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[2]);
//     size_t const out_ldy = STARPU_BLOCK_GET_LDY(buffers[2]);
//     size_t const out_ldz = STARPU_BLOCK_GET_LDZ(buffers[2]);
//     dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);
// 
//     assert(out_nx == in_nx - k_nx + 1);
//     assert(out_ny == in_ny - k_ny + 1);
//     assert(out_nz == k_nz);
// 
//     // loop through i,j,k on axes x,y,z of the output block
//     for (size_t k = 0; k < out_nz; k++)
//     {
//         for (size_t j = 0; j < out_ny; j++)
//         {
//             for (size_t i = 0; i < out_nx; i++)
//             {
//                 dahl_fp cell_res = 0.0F;
// 
//                 // loop through l,m on axes x,y of the kernel
//                 for (size_t m = 0; m < k_ny; m++)
//                 {
//                     for (size_t l = 0; l < k_nx; l++)
//                     {
//                         // Reverse indexes l and m so we don't actually have to rotate(180) the kernel matrix.
//                         // However we still use k for axis z because we write each result for the current channel into an output channel.
//                         dahl_fp kernel_value = kernel[(k * k_ldz) + ((k_ny - 1 - m) * k_ldy) + (k_nx - 1 - l)];
//                         // Here we use k, the index on the z axis of the output, as input owns as many channels.
//                         // The kernel doesn't own a channel dimension in this function, so we ignore it.
//                         // Then we add the offset of the slidding window (i,j) to (l,m)
//                         // as they both correspond to (x,y).
//                         dahl_fp in_value = in[((m + j) * in_ld) + l + i];
// 
//                         cell_res += in_value * kernel_value;
//                     }
//                 }
// 
//                 // Set the corresponding value for index i,j,k
//                 out[(k * out_ldz) + (j * out_ldy) + i] = cell_res;
//             }
//         }
//     }
// }

/*
 * This function implements a "full" convolution with padding free input. This means that the output
 * is larger than the input, but we don't need to use zero padding and compute useless operations on
 * the padding.
 * We do that by computing start/end indexes of each kernel window so that we ignore out-of-bound
 * kernel values.
 * It uses saturating arithmetic trick to prevent conditionnal branches to appear in for loops.
 *
 *      kernel size
 *     ┌───────────┐
 *     ▼           ▼
 *        Actual range we want
 *             ┌───┐
 *             ▼   ▼
 *      -2  -1   0   1   2   3     
 *     ┌ ─ ┬ ─ ┬ ─ ┬ ─ ┬ ─ ┬ ─ ┐
 *  -2   0   0   0   0   0   0                            0   1   2   3
 *     ├ ─ ┼ ─ ┼ ─ ┼ ─ ┼ ─ ┼ ─ ┤        0   1   2       ┌───┬───┬───┬───┐
 *  -1   0   0   0   0   0   0        ┌───┬───┬───┐   0 │   │   │   │   │
 *     ├ ─ ┼ ─ ┼───┼───┼ ─ ┼ ─ ┤    0 │   │   │   │     ├───┼───┼───┼───┤
 *   0   0   0 │   │   │ 0   0        ├───┼───┼───┤   1 │   │   │   │   │
 *     ├ ─ ┼ ─ ┼───┼───┼ ─ ┼ ─ ┤    1 │   │   │   │     ├───┼───┼───┼───┤
 *   1   0   0 │   │   │ 0   0        ├───┼───┼───┤   2 │   │   │   │   │
 *     ├ ─ ┼ ─ ┼───┼───┼ ─ ┼ ─ ┤    2 │   │   │   │     ├───┼───┼───┼───┤
 *   2   0    Input buffer   0        └───┴───┴───┘   3 │   │   │   │   │
 *     ├ ─ ┼ ─ ┼ ─ ┼ ─ ┼ ─ ┼ ─ ┤         Kernel         └───┴───┴───┴───┘
 *   3   0   0   0   0   0   0       (omitted z dim)         Output
 *     └ ─ ┴ ─ ┴ ─ ┴ ─ ┴ ─ ┴ ─ ┘                         (omitted z dim)
 *            Fake Padding
 */
void convolution_2d_backward_input_padding_free(void* buffers[3], void* cl_arg)
{
    // Input matrix, here the gradients output of the layer just after the convolution
    size_t const in_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    size_t const in_ld = STARPU_MATRIX_GET_LD(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[0]);

    // Kernel block, here the filters (weights) associated to the convolution
    size_t const k_nx = STARPU_BLOCK_GET_NX(buffers[1]);
    size_t const k_ny = STARPU_BLOCK_GET_NY(buffers[1]);
    size_t const k_nz = STARPU_BLOCK_GET_NZ(buffers[1]);
    size_t const k_ldy = STARPU_BLOCK_GET_LDY(buffers[1]);
    size_t const k_ldz = STARPU_BLOCK_GET_LDZ(buffers[1]);
    dahl_fp const* kernel = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    // Output block, here the loss derivative of the input
    size_t const out_nx = STARPU_BLOCK_GET_NX(buffers[2]);
    size_t const out_ny = STARPU_BLOCK_GET_NY(buffers[2]);
    size_t const out_nz = STARPU_BLOCK_GET_NZ(buffers[2]);
    size_t const out_ldy = STARPU_BLOCK_GET_LDY(buffers[2]);
    size_t const out_ldz = STARPU_BLOCK_GET_LDZ(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[2]);

    assert(out_nx == in_nx + k_nx - 1);
    assert(out_ny == in_ny + k_ny - 1);
    assert(out_nz == k_nz);

    // Compute padding size required to produce the correct output shape.
    size_t pad_nx = k_nx - 1;
    size_t pad_ny = k_ny - 1;

    // loop through i,j,k on axes x,y,z of the output block
    for (size_t k = 0; k < out_nz; k++)
    {
        for (size_t j = 0; j < out_ny; j++)
        {
            for (size_t i = 0; i < out_nx; i++)
            {
                dahl_fp cell_res = 0.0F;

                // [`y_start`, `y_end`[ corresponds to the window where we pull values from the                 
                // input matrix, `y_ker` simulates how shifted the kernel is, its values gets    
                // incremented each loop to simulate shifting. Same principle for x dimension.   
                
                // Here, we will always shift our kernel up from `pad_ny` values relative to j.
                // Using sub_sat let us handle cases where the kernel has values outside the matrix,
                // underflow will saturate to value 0.
                size_t y_start = sub_sat(j, pad_ny);

                // We can use the same trick for the end index, but here we "reverse" indexes, so we
                // can use the underflow mechanic. Minus 1 because this is the remaining part of the
                // kernel (pad_ny being included in y_start).
                size_t y_end = sub_sat(in_ny - 1, j); 

                // Reverse indexes once again to retrieve the actual index of the end. When we reach
                // the bottom of the matrix, the kernel will be outside bounds so `y_end` values
                // will be saturated to 0, thus `y_end` will equal `in_ny`.
                y_end = in_ny - y_end;

                // Here we compute where the kernel should start by substracting the end index.
                // If it overflows, it means there's no more need to shift the kernel, all start at
                // 0
                size_t y_ker = sub_sat(k_ny, y_end);

                // Loop through l,m on axes x,y of computed window
                for (size_t m = y_start; m < y_end; m++)
                {
                    size_t x_start = sub_sat(i, pad_nx);
                    size_t x_end = sub_sat(in_nx - 1, i); 
                    x_end = in_nx - x_end;
                    size_t x_ker = sub_sat(k_nx, x_end);

                    for (size_t l = x_start; l < x_end; l++)
                    {
                        // Reverse indexes x_ker and y_ker so we don't actually have to rotate(180)
                        // the kernel matrix. However we still use k for axis z because we write
                        // each result for the current `kernel` channel into the same `out` channel.
                        dahl_fp kernel_value = kernel[
                            (k * k_ldz) + ((k_ny - 1 - y_ker) * k_ldy) + (k_nx - 1 - x_ker)];
                        dahl_fp in_value = in[(m * in_ld) + l];

                        cell_res += in_value * kernel_value;
                        x_ker++;
                    }

                    y_ker++;
                }

                // Set the corresponding value for index i,j,k
                out[(k * out_ldz) + (j * out_ldy) + i] = cell_res;
            }
        }
    }
}
