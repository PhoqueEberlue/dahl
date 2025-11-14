#include <cstdio>
#include <starpu.h>
#include "common.cuh"
#include "../../macros.h"
#include "../../../include/dahl_types.h"
#include "starpu_cuda.h"
#include "starpu_data_interfaces.h"

extern "C" void cuda_matrix_cross_correlation(void* buffers[3], void* cl_arg)
{

}

static __global__ void matrix_max_pooling(
        struct starpu_matrix_interface const in,
        struct starpu_matrix_interface const mask,
        struct starpu_matrix_interface const out,
        size_t pool_size)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto mask_p = (dahl_fp*)mask.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= out.nx || j >= out.ny)
        return;

    // Compute pooling window bounds
    size_t start_y = j * pool_size;
    size_t start_x = i * pool_size;
    size_t end_y   = start_y + pool_size;
    size_t end_x   = start_x + pool_size;

    // Set max value and index to first element to handle cases with negative numbers
    dahl_fp current_max = in_p[(start_y * in.ld) + start_x];
    size_t current_max_y = start_y;
    size_t current_max_x = start_x;

    for (size_t y = start_y; y < end_y; y++)
    {
        for (size_t x = start_x; x < end_x; x++)
        {
            dahl_fp value = in_p[(y * in.ld) + x];
            if (value > current_max)
            {
                current_max = value;
                current_max_y = y;
                current_max_x = x;
            }
        }
    }

    // Write max value to output
    out_p[(j * out.ld) + i] = current_max;

    // Mark position in mask
    mask_p[(current_max_y * mask.ld) + current_max_x] = 1.0F;
}

extern "C" void cuda_matrix_max_pooling(void* buffers[3], void* cl_arg)
{
    size_t pool_size;
    starpu_codelet_unpack_args(cl_arg, &pool_size);

    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto mask = STARPU_MATRIX_GET(buffers[1]);
    auto out = STARPU_MATRIX_GET(buffers[2]);

    dim3 block(16, 16);
    dim3 grid((out.nx + block.x - 1) / block.x,
              (out.ny + block.y - 1) / block.y);

    matrix_max_pooling<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(in, mask, out, pool_size);
    dahl_cuda_check_error_and_sync();
}

static __global__ void matrix_backward_max_pooling(
        struct starpu_matrix_interface const in,
        struct starpu_matrix_interface const mask,
        struct starpu_matrix_interface const out,
        size_t pool_size)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto mask_p = (dahl_fp*)mask.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= out.nx || j >= out.ny)
        return;

    size_t start_l = j * pool_size;
    size_t start_k = i * pool_size;
    size_t end_l = start_l + pool_size;
    size_t end_k = start_k + pool_size;

    // Loop through k,l on axes x,y of matrix `out` and `mask`
    for (size_t l = start_l; l < end_l; l++)
    {
        for (size_t k = start_k; k < end_k; k++)
        {
            dahl_fp in_value = in_p[(j * in.ld) + i];
            dahl_fp mask_value = mask_p[(l * mask.ld) + k];

            // If the current value of the mask mask is 0, the result is ignored.
            // I'm not sure if its better to check that conditionnaly, or to always write a result.
            out_p[(l * out.ld) + k] = in_value * mask_value;
        }
    }
}

extern "C" void cuda_matrix_backward_max_pooling(void *buffers[3], void *cl_arg)
{
    size_t pool_size;
    starpu_codelet_unpack_args(cl_arg, &pool_size);

    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto mask = STARPU_MATRIX_GET(buffers[1]);
    auto out = STARPU_MATRIX_GET(buffers[2]);

    dim3 block(16, 16);
    dim3 grid((in.nx + block.x - 1) / block.x,
              (in.ny + block.y - 1) / block.y);

    matrix_backward_max_pooling<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(in, mask, out, pool_size);
    dahl_cuda_check_error_and_sync();
}

static __global__ void matrix_matrix_product(
        struct starpu_matrix_interface const a,
        struct starpu_matrix_interface const b,
        struct starpu_matrix_interface const c)
{
    auto a_p = (dahl_fp const*)a.ptr;
    auto b_p = (dahl_fp const*)b.ptr;
    auto c_p = (dahl_fp*)c.ptr;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x; // column of c
    size_t y = blockIdx.y * blockDim.y + threadIdx.y; // row of c

    if (x >= c.nx || y >= c.ny)
        return;

    dahl_fp sum = 0.0;

    // Dot product of row a[y,*] and col b[*,x]
    for (size_t i = 0; i < a.nx; i++)
    {
        dahl_fp val_a = a_p[(y * a.ld) + i];
        dahl_fp val_b = b_p[(i * b.ld) + x];
        sum += val_a * val_b;
    }

    c_p[(y * c.ld) + x] = sum;
}

extern "C" void cuda_matrix_matrix_product(void* buffers[3], void* cl_arg)
{
    auto a = STARPU_MATRIX_GET(buffers[0]);
    auto b = STARPU_MATRIX_GET(buffers[1]);
    auto c = STARPU_MATRIX_GET(buffers[2]);

    dim3 block(16, 16);
    dim3 grid((c.nx + block.x - 1) / block.x,
              (c.ny + block.y - 1) / block.y);

    matrix_matrix_product<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(a, b ,c);
    dahl_cuda_check_error_and_sync();
}

static __global__ void matrix_sum_y_axis(
        struct starpu_matrix_interface const in,
        struct starpu_vector_interface  const out)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;

    // Column index handled by this thread
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= in.nx)
        return;

    dahl_fp sum = 0.0;

    // Sum along Y axis
    for (size_t y = 0; y < in.ny; y++)
    {
        sum += in_p[(y * in.ld) + x];
    }

    out_p[x] = sum;
}

extern "C" void cuda_matrix_sum_y_axis(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto out = STARPU_VECTOR_GET(buffers[1]);

    int threadsPerBlock = 256;
    int blocksPerGrid = (in.nx + threadsPerBlock - 1) / threadsPerBlock;

    matrix_sum_y_axis<<<blocksPerGrid, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(in, out);
    dahl_cuda_check_error_and_sync();
}

static __global__ void matrix_vector_product(
        struct starpu_matrix_interface const mat,
        struct starpu_vector_interface const vec,
        struct starpu_vector_interface  const out)
{
    auto mat_p = (dahl_fp const*)mat.ptr;
    auto vec_p = (dahl_fp const*)vec.ptr;
    auto out_p = (dahl_fp*)out.ptr;   

    // Thread index: one thread per row of the matrix
    size_t y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= mat.ny)
        return;

    dahl_fp sum = 0.0;

    // Compute dot product of row `y` of mat with vec
    for (size_t x = 0; x < mat.nx; x++)
    {
        sum += mat_p[(y * mat.ld) + x] * vec_p[x];
    }

    out_p[y] = sum;
}

extern "C" void cuda_matrix_vector_product(void* buffers[3], void* cl_arg)
{
    auto mat = STARPU_MATRIX_GET(buffers[0]);
    auto vec = STARPU_VECTOR_GET(buffers[1]);
    auto out = STARPU_VECTOR_GET(buffers[2]);

    int threadsPerBlock = 256;
    int blocksPerGrid = (mat.ny + threadsPerBlock - 1) / threadsPerBlock;

    matrix_vector_product<<<blocksPerGrid, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(mat, vec, out);
    dahl_cuda_check_error_and_sync();
}

static __global__ void matrix_transpose(
        struct starpu_matrix_interface const in,
        struct starpu_matrix_interface  const out)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;   

    // Thread coordinates (x = column, y = row)
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.nx || y >= in.ny)
        return;

    // Perform the transpose
    out_p[(x * out.ld) + y] = in_p[(y * in.ld) + x];
}

extern "C" void cuda_matrix_transpose(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto out = STARPU_MATRIX_GET(buffers[1]);

    dim3 block(16, 16);
    dim3 grid((in.nx + block.x - 1) / block.x,
              (in.ny + block.y - 1) / block.y);

    matrix_transpose<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(in, out);
    dahl_cuda_check_error_and_sync();
}

extern "C" void cuda_matrix_resize(void* buffers[1], void* cl_arg)
{
    size_t new_nx;
    size_t new_ny;
    size_t new_ld;
    starpu_codelet_unpack_args(cl_arg, &new_nx, &new_ny, &new_ld);

    STARPU_MATRIX_SET_NX(buffers[0], new_nx);
    STARPU_MATRIX_SET_NY(buffers[0], new_ny);
    STARPU_MATRIX_SET_LD(buffers[0], new_ld);
}

static __global__ void matrix_rotate_180(
        struct starpu_matrix_interface const in,
        struct starpu_matrix_interface  const out)
{
    auto in_p = (dahl_fp const*)in.ptr;
    auto out_p = (dahl_fp*)out.ptr;   

    // Thread coordinates (x = column, y = row)
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.nx || y >= in.ny)
        return;

    // Perform the transpose
    size_t y_rot = (out.ny - 1 - y);
    size_t x_rot = (out.nx - 1 - x);
    out_p[(y_rot * out.ld) + x_rot] = in_p[(y * in.ld) + x];
}

extern "C" void cuda_matrix_rotate_180(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_MATRIX_GET(buffers[0]);
    auto out = STARPU_MATRIX_GET(buffers[1]);

    dim3 block(16, 16);
    dim3 grid((in.nx + block.x - 1) / block.x,
              (in.ny + block.y - 1) / block.y);

    matrix_rotate_180<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(in, out);
    dahl_cuda_check_error_and_sync();
}

extern "C" void cuda_matrix_zero(void *buffers[1], void *cl_arg)
{
    auto in = STARPU_MATRIX_GET(buffers[0]);
	cudaMemsetAsync((dahl_fp*)in.ptr, 0, in.ld * in.ny * in.elemsize, 
            starpu_cuda_get_local_stream());
}

extern "C" void cuda_matrix_accumulate(void *buffers[2], void *cl_arg)
{
    auto dst = STARPU_MATRIX_GET(buffers[0]);
    auto src = STARPU_MATRIX_GET(buffers[1]);
    auto dst_p = (dahl_fp*)dst.ptr;
    auto src_p = (dahl_fp const*)src.ptr;

    size_t nb_elem = dst.nx * dst.ny;

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    cuda_accumulate<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, dst_p, src_p);
    dahl_cuda_check_error_and_sync();
}

static __global__ void matrix_print(struct starpu_matrix_interface const mat)
{
    auto matrix_p = (dahl_fp const*)mat.ptr;

    printf("matrix=%p nx=%llu ny=%llu ld=%llu\n{ ", (void*)mat.ptr, mat.nx, mat.ny, mat.ld);
    for(size_t y = 0; y < mat.ny; y++)
    {
        printf("\t{ ");
        for(size_t x = 0; x < mat.nx; x++)
        {
            dahl_fp value = matrix_p[(y * mat.ld) + x];
            printf("%f, ", value);
        }
        printf("},\n");
    }
    printf("}\n");
}

extern "C" void cuda_matrix_print(void *buffers[1], void *cl_arg)
{
    auto matrix = STARPU_MATRIX_GET(buffers[0]);
    matrix_print<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(matrix);
    dahl_cuda_check_error_and_sync();
}
