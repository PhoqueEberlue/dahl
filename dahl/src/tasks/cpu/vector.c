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

void vector_softmax(void* buffers[2], void* cl_arg)
{
    auto in = STARPU_VECTOR_GET(buffers[0]);
    auto in_p = (dahl_fp const*)in.ptr;

    auto out = STARPU_VECTOR_GET(buffers[1]);
    auto out_p = (dahl_fp*)in.ptr;

    assert(in.nx == out.nx);

    dahl_fp max_value = 0.0F;

    // Getting max value
    for (size_t i = 0; i < in.nx; i++)
    {
        if (in_p[i] > max_value)
        {
            max_value = in_p[i];
        }
    }

    dahl_fp sum_values = 0.0F;

    // Shifting by the max value, computing exponent for each element, and summing
    for (size_t i = 0; i < in.nx; i++)
    {
        out_p[i] = exp(in_p[i] - max_value);
        sum_values += out_p[i];
    }

    // Computing the probabilities
    for (size_t i = 0; i < in.nx; i++)
    {
        out_p[i] = out_p[i] / sum_values;
    }
}

void vector_dot_product(void* buffers[3], void* cl_arg)
{
    auto a = STARPU_VECTOR_GET(buffers[0]);
    auto a_p = (dahl_fp const*)a.ptr;

    auto b = STARPU_VECTOR_GET(buffers[1]);
    auto b_p = (dahl_fp const*)b.ptr;

    dahl_fp* c = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[2]);

    assert(a.nx == b.nx);

    for (size_t i = 0; i < a.nx; i++)
    {
        *c += a_p[i] * b_p[i];
    }
}

void vector_diag(void* buffers[2], void* cl_arg)
{
    // Input vector
    auto in = STARPU_VECTOR_GET(buffers[0]);
    auto in_p = (dahl_fp const*)in.ptr;

    // Output matrix
    auto out = STARPU_MATRIX_GET(buffers[1]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(in.nx == out.nx);
    assert(in.nx == out.ny);

    for (size_t i = 0; i < in.nx; i++)
    {
        // Copy the vector's elements in a diagonal manner into the matrix
        out_p[(i * out.ld) + i] = in_p[i];
    }
}

void vector_outer_product(void* buffers[3], void* cl_arg)
{
    auto a = STARPU_VECTOR_GET(buffers[0]);
    auto a_p = (dahl_fp const*)a.ptr;

    auto b = STARPU_VECTOR_GET(buffers[1]);
    auto b_p = (dahl_fp const*)b.ptr;

    auto c = STARPU_MATRIX_GET(buffers[2]);
    auto c_p = (dahl_fp*)c.ptr;

    assert(a.nx == c.nx);
    assert(b.nx == c.ny);

    for (size_t y = 0; y < b.nx; y++)
    {
        for (size_t x = 0; x < a.nx; x++)
        {
            c_p[(y * c.ld) + x] += b_p[y] * a_p[x];
        }
    }
}

void vector_shuffle(void* buffers[1], void* cl_arg)
{
    auto vec = STARPU_VECTOR_GET(buffers[0]);
    auto vec_p = (dahl_fp*)vec.ptr;

    for (size_t i = vec.nx - 1; i > 0; i--)
    {
        // Generate a random index respecting 0 <= j <= i
        size_t j = ( rand() / (RAND_MAX / i));
        assert(0 <= j && j <= i);

        // Swap the two values
        dahl_fp tmp = vec_p[i];
        vec_p[i] = vec_p[j];
        vec_p[j] = tmp;
    }
}

void vector_matrix_product(void* buffers[3], void* cl_arg)
{
    // Input vector
    auto vec = STARPU_VECTOR_GET(buffers[0]);
    auto vec_p = (dahl_fp const*)vec.ptr;

    // Input matrix
    auto mat = STARPU_MATRIX_GET(buffers[1]);
    auto mat_p = (dahl_fp const*)mat.ptr;

    // Output vector
    auto out = STARPU_VECTOR_GET(buffers[2]);
    auto out_p = (dahl_fp*)out.ptr;

    assert(vec.nx == mat.ny);
    assert(out.nx == mat.nx);

    // Loop through x,y of the matrix
    for (size_t y = 0; y < mat.ny; y++)
    {
        for (size_t x = 0; x < mat.nx; x++)
        {
            out_p[x] += vec_p[y] * mat_p[(y * mat.ld) + x];
        }
    }
}

void vector_zero(void *buffers[1], void *cl_arg)
{
	// Vector
    auto vec = STARPU_VECTOR_GET(buffers[0]);
    auto vec_p = (dahl_fp*)vec.ptr;

    for (int x = 0; x < vec.nx; x++)
    {
        vec_p[x] = 0;
    }
}

void vector_accumulate(void *buffers[2], void *cl_arg)
{
    // dst vector accumulator
    auto dst = STARPU_VECTOR_GET(buffers[0]);
    auto dst_p = (dahl_fp*)dst.ptr;

    // src vector
    auto src = STARPU_VECTOR_GET(buffers[1]);
    auto src_p = (dahl_fp const*)src.ptr;

    assert(dst.nx == src.nx);

    for (size_t x = 0; x < dst.nx; x++)
    {
        dst_p[x] += src_p[x];
    }
}
