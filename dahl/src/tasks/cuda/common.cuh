#ifndef DAHL_COMMON_CUH
#define DAHL_COMMON_CUH

#include <starpu.h>
#include "../../../include/dahl_types.h"

#define dahl_cuda_check_error_and_sync() ({                      \
    cudaError_t status = cudaGetLastError();                     \
    if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status); \
    cudaStreamSynchronize(starpu_cuda_get_local_stream());       \
})

static __global__ void cuda_accumulate(
        size_t nb_elem, dahl_fp* dst_p, dahl_fp const* src_p)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;
    // dst_p[index] += src_p[index];
    // TODO: tmp test, remove after
    atomicAdd(&dst_p[index], src_p[index]);
}

#endif //!DAHL_COMMON_CUH
