#include <starpu.h>
#include <stdio.h>
#include "common.cuh"
#include "../../macros.h"

extern "C" void cuda_vector_softmax(void* buffers[2], void* cl_arg)
{

}

static __global__ void vector_dot_product(
    struct starpu_vector_interface const a,
    struct starpu_vector_interface const b,
    struct starpu_variable_interface const c)
{
    const dahl_fp* a_p = (const dahl_fp*)a.ptr;
    const dahl_fp* b_p = (const dahl_fp*)b.ptr;
    dahl_fp* c_p = (dahl_fp*)c.ptr;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    dahl_fp local_sum = 0.0f;
    for (; i < a.nx; i += stride)
        local_sum += a_p[i] * b_p[i];

    atomicAdd(c_p, local_sum);
}

extern "C" void cuda_vector_dot_product(void* buffers[3], void* cl_arg)
{
    auto a = STARPU_VECTOR_GET(buffers[0]);
    auto b = STARPU_VECTOR_GET(buffers[1]);
    auto c = STARPU_VARIABLE_GET(buffers[2]);

    // Reset output
    cudaMemsetAsync((void*)c.ptr, 0, sizeof(dahl_fp), starpu_cuda_get_local_stream());

    int threads = 256;
    int blocks = (a.nx + threads - 1) / threads;

    vector_dot_product<<<blocks, threads, 0, starpu_cuda_get_local_stream()>>>(a, b, c);
    dahl_cuda_check_error_and_sync();
}

static __global__ void vector_diag(
    struct starpu_vector_interface const in,
    struct starpu_matrix_interface const out)
{
    const dahl_fp* in_p = (const dahl_fp*)in.ptr;
    dahl_fp* out_p = (dahl_fp*)out.ptr;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < in.nx)
        out_p[(i * out.ld) + i] = in_p[i];
}

extern "C" void cuda_vector_diag(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_VECTOR_GET(buffers[0]);
    auto out = STARPU_MATRIX_GET(buffers[1]);

    int threads = 256;
    int blocks = (in.nx + threads - 1) / threads;
    vector_diag<<<blocks, threads, 0, starpu_cuda_get_local_stream()>>>(in, out);
    dahl_cuda_check_error_and_sync();
}

static __global__ void vector_outer_product(
    struct starpu_vector_interface const a,
    struct starpu_vector_interface const b,
    struct starpu_matrix_interface const c)
{
    const dahl_fp* a_p = (const dahl_fp*)a.ptr;
    const dahl_fp* b_p = (const dahl_fp*)b.ptr;
    dahl_fp* c_p = (dahl_fp*)c.ptr;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: remove += when starpu redux for cuda is fixed
    if (x < a.nx && y < b.nx)
        c_p[(y * c.ld) + x] += b_p[y] * a_p[x];
}

extern "C" void cuda_vector_outer_product(void* buffers[3], void* cl_arg)
{
    auto a = STARPU_VECTOR_GET(buffers[0]);
    auto b = STARPU_VECTOR_GET(buffers[1]);
    auto c = STARPU_MATRIX_GET(buffers[2]);

    dim3 block(16, 16);
    dim3 grid((a.nx + 15) / 16, (b.nx + 15) / 16);
    vector_outer_product<<<grid, block, 0, starpu_cuda_get_local_stream()>>>(a, b, c);
    dahl_cuda_check_error_and_sync();
}

extern "C" void cuda_vector_shuffle(void* buffers[1], void* cl_arg)
{

}

static __global__ void vector_matrix_product(
        struct starpu_vector_interface const vec,
        struct starpu_matrix_interface const mat,
        struct starpu_vector_interface  const out)
{
    auto vec_p = (dahl_fp const*)vec.ptr;
    auto mat_p = (dahl_fp const*)mat.ptr;
    auto out_p = (dahl_fp*)out.ptr;   

    // Thread index: one thread per row of the matrix
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= mat.nx)
        return;

    dahl_fp sum = 0.0;

    // Compute dot product of row `y` of mat with vec
    for (size_t y = 0; y < mat.ny; y++)
    {
        sum += mat_p[(y * mat.ld) + x] * vec_p[y];
    }

    out_p[x] = sum;
}

extern "C" void cuda_vector_matrix_product(void* buffers[3], void* cl_arg)
{
    auto vec = STARPU_VECTOR_GET(buffers[0]);
    auto mat = STARPU_MATRIX_GET(buffers[1]);
    auto out = STARPU_VECTOR_GET(buffers[2]);

    int threadsPerBlock = 256;
    int blocksPerGrid = (mat.nx + threadsPerBlock - 1) / threadsPerBlock;

    vector_matrix_product<<<blocksPerGrid, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(vec, mat, out);
    dahl_cuda_check_error_and_sync();
}

extern "C" void cuda_vector_zero(void *buffers[1], void *cl_arg)
{
    auto in = STARPU_VECTOR_GET(buffers[0]);
	cudaMemsetAsync((dahl_fp*)in.ptr, 0, in.nx * in.elemsize, 
            starpu_cuda_get_local_stream());
}

extern "C" void cuda_vector_accumulate(void *buffers[2], void *cl_arg)
{
    auto dst = STARPU_VECTOR_GET(buffers[0]);
    auto src = STARPU_VECTOR_GET(buffers[1]);
    auto dst_p = (dahl_fp*)dst.ptr;
    auto src_p = (dahl_fp const*)src.ptr;

    int threadsPerBlock = 256;
    int numBlocks = (dst.nx + threadsPerBlock - 1) / threadsPerBlock;

    cuda_accumulate<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(dst.nx, dst_p, src_p);
    dahl_cuda_check_error_and_sync();
}

static __global__ void vector_print(struct starpu_vector_interface const vec)
{
    auto vec_p = (dahl_fp const*)vec.ptr;

    printf("vector=%p nx=%llu\n{ ", (void*)vec.ptr, vec.nx);
    for (size_t x = 0; x < vec.nx; x++)
    {
        printf("%.14f, ", vec_p[x]);
    }
    printf("}\n");
}

extern "C" void cuda_vector_print(void *buffers[1], void *cl_arg)
{
    auto vec = STARPU_VECTOR_GET(buffers[0]);
    vector_print<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(vec);
    dahl_cuda_check_error_and_sync();
}
