#include <starpu.h>

#define STARPU_ANY_GET_PTR(interface) (((struct starpu_vector_interface *)(interface))->ptr)

void cpu_tensor_zero(void *buffers[1], void *cl_arg)
{
	// Tensor
    size_t const nx = STARPU_TENSOR_GET_NX(buffers[0]);
    size_t const ny = STARPU_TENSOR_GET_NY(buffers[0]);
    size_t const nz = STARPU_TENSOR_GET_NZ(buffers[0]);
    size_t const nt = STARPU_TENSOR_GET_NT(buffers[0]);
    size_t const ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
    size_t const ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
    size_t const ldt = STARPU_TENSOR_GET_LDT(buffers[0]);
    double* data = (double*)STARPU_TENSOR_GET_PTR(buffers[0]);

    for (int t = 0; t < nt; t++)
    {
        for (int z = 0; z < nz; z++)
        {
            for (int y = 0; y < ny; y++)
            {
                for (int x = 0; x < nx; x++)
                {
                    data[(t * ldt) + (z * ldz) + (y * ldy) + x] = 0;
                }
            }
        }
    }
}

void cpu_tensor_accumulate(void *buffers[2], void *cl_arg)
{
    printf("called cuda accumulate\n");
	// dst tensor accumulator
    size_t const dst_nx = STARPU_TENSOR_GET_NX(buffers[0]);
    size_t const dst_ny = STARPU_TENSOR_GET_NY(buffers[0]);
    size_t const dst_nz = STARPU_TENSOR_GET_NZ(buffers[0]);
    size_t const dst_nt = STARPU_TENSOR_GET_NT(buffers[0]);
    size_t const dst_ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
    size_t const dst_ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
    size_t const dst_ldt = STARPU_TENSOR_GET_LDT(buffers[0]);
    double* dst = (double*)STARPU_TENSOR_GET_PTR(buffers[0]);

    // source tensor
    size_t const src_nx = STARPU_TENSOR_GET_NX(buffers[1]);
    size_t const src_ny = STARPU_TENSOR_GET_NY(buffers[1]);
    size_t const src_nz = STARPU_TENSOR_GET_NZ(buffers[1]);
    size_t const src_nt = STARPU_TENSOR_GET_NT(buffers[1]);
    size_t const src_ldy = STARPU_TENSOR_GET_LDY(buffers[1]);
    size_t const src_ldz = STARPU_TENSOR_GET_LDZ(buffers[1]);
    size_t const src_ldt = STARPU_TENSOR_GET_LDT(buffers[1]);
    double const* src = (double*)STARPU_TENSOR_GET_PTR(buffers[1]);

    assert(dst_nx == src_nx);
    assert(dst_ny == src_ny);
    assert(dst_nz == src_nz);
    assert(dst_nt == src_nt);
    assert(dst_ldy == src_ldy);
    assert(dst_ldz == src_ldz);
    assert(dst_ldt == src_ldt);

    for (int t = 0; t < dst_nt; t++)
    {
        for (int z = 0; z < dst_nz; z++)
        {
            for (int y = 0; y < dst_ny; y++)
            {
                for (int x = 0; x < dst_nx; x++)
                {
                    dst[(t * dst_ldt) + (z * dst_ldz) + (y * dst_ldy) + x] += 
                        src[(t * src_ldt) + (z * src_ldz) + (y * src_ldy) + x];
                }
            }
        }
    }
}

void cpu_any_add(void* buffers[3], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    double const* a = (double*)STARPU_ANY_GET_PTR(buffers[0]);
    double const* b = (double*)STARPU_ANY_GET_PTR(buffers[1]);
    double* c = (double*)STARPU_ANY_GET_PTR(buffers[2]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        c[i] = a[i] + b[i];
    }
}
