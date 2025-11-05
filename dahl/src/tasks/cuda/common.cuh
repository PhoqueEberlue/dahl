#ifndef DAHL_COMMON_CUH
#define DAHL_COMMON_CUH

#include <starpu.h>
#include "../../../include/dahl_types.h"

static __global__ void cuda_accumulate(
        size_t nb_elem, dahl_fp* dst_p, dahl_fp const* src_p)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nb_elem) return;
    dst_p[index] += src_p[index];
}

#endif //!DAHL_COMMON_CUH
