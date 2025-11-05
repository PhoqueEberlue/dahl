#include <cstdio>
#include <starpu.h>

#define STARPU_ANY_GET_PTR(interface) (((struct starpu_vector_interface *)(interface))->ptr)
#define STARPU_TENSOR_GET(interface) (*(struct starpu_tensor_interface const*)(interface))

extern "C" void cuda_tensor_zero(void *buffers[1], void *cl_arg)
{
    auto in = STARPU_TENSOR_GET(buffers[0]);
	cudaMemsetAsync((double*)in.ptr, 0, in.ldt * in.nt * in.elemsize, 
            starpu_cuda_get_local_stream());
}

static __global__ void cuda_accumulate(
        size_t nb_elem, double* dst_p, double const* src_p)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;
    dst_p[index] += src_p[index];
}

extern "C" void cuda_tensor_accumulate(void *buffers[2], void *cl_arg)
{
    printf("called cuda accumulate\n");
    auto dst = STARPU_TENSOR_GET(buffers[0]);
    auto src = STARPU_TENSOR_GET(buffers[1]);
    auto dst_p = (double*)dst.ptr;
    auto src_p = (double const*)src.ptr;

    size_t nb_elem = dst.nx * dst.ny * dst.nz * dst.nt;

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    cuda_accumulate<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, dst_p, src_p);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}


static __global__ void any_add(
        size_t nb_elem,
        double const* a, double const* b, double* c)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;
    c[index] = a[index] + b[index];
}

extern "C" void cuda_any_add(void* buffers[3], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    auto a = (double const*)STARPU_ANY_GET_PTR(buffers[0]);
    auto b = (double const*)STARPU_ANY_GET_PTR(buffers[1]);
    auto c = (double*)STARPU_ANY_GET_PTR(buffers[2]);

    int threadsPerBlock = 256;
    int numBlocks = (nb_elem + threadsPerBlock - 1) / threadsPerBlock;

    any_add<<<numBlocks, threadsPerBlock, 0, starpu_cuda_get_local_stream()>>>(nb_elem, a, b, c);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
