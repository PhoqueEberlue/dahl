#include <starpu.h>

extern "C" void cuda_check_predictions_batch(void* buffers[3], void* cl_arg)
{

}


extern "C" void cuda_cross_entropy_loss_batch(void* buffers[3], void* cl_arg)
{

}


extern "C" void cuda_cross_entropy_loss_gradient(void* buffers[3], void* cl_arg)
{

}


extern "C" void cuda_convolution_2d(void* buffers[3], void* cl_arg)
{

}


extern "C" void cuda_convolution_2d_backward_filters(void* buffers[3], void* cl_arg)
{

}


extern "C" void cuda_convolution_2d_backward_input(void* buffers[3], void* cl_arg)
{

}


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
extern "C" void cuda_convolution_2d_backward_input_padding_free(void* buffers[3], void* cl_arg)
{

}

