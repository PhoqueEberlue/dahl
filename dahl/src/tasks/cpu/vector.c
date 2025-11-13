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
    size_t const in_len = STARPU_VECTOR_GET_NX(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    size_t const out_len = STARPU_VECTOR_GET_NX(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[1]);

    assert(in_len == out_len);

    dahl_fp max_value = 0.0F;

    // Getting max value
    for (size_t i = 0; i < in_len; i++)
    {
        if (in[i] > max_value)
        {
            max_value = in[i];
        }
    }

    dahl_fp sum_values = 0.0F;

    // Shifting by the max value, computing exponent for each element, and summing
    for (size_t i = 0; i < in_len; i++)
    {
        out[i] = exp(in[i] - max_value);
        sum_values += out[i];
    }

    // Computing the probabilities
    for (size_t i = 0; i < in_len; i++)
    {
        out[i] = out[i] / sum_values;
    }
}

void vector_dot_product(void* buffers[3], void* cl_arg)
{
    size_t const a_len = STARPU_VECTOR_GET_NX(buffers[0]);
    dahl_fp const* a = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    size_t const b_len = STARPU_VECTOR_GET_NX(buffers[1]);
    dahl_fp const* b = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[1]);

    dahl_fp* c = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[2]);

    assert(a_len == b_len);

    for (size_t i = 0; i < a_len; i++)
    {
        *c += a[i] * b[i];
    }
}

void vector_diag(void* buffers[2], void* cl_arg)
{
    // Input vector
    size_t const in_len = STARPU_VECTOR_GET_NX(buffers[0]);
    dahl_fp const* in = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    // Output matrix
    size_t const out_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const out_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const out_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    assert(in_len == out_nx);
    assert(in_len == out_ny);

    for (size_t i = 0; i < in_len; i++)
    {
        // Copy the vector's elements in a diagonal manner into the matrix
        out[(i * out_ld) + i] = in[i];
    }
}

void vector_outer_product(void* buffers[3], void* cl_arg)
{
    size_t const a_len = STARPU_VECTOR_GET_NX(buffers[0]);
    dahl_fp const* a = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    size_t const b_len = STARPU_VECTOR_GET_NX(buffers[1]);
    dahl_fp const* b = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[1]);

    size_t const c_nx = STARPU_MATRIX_GET_NX(buffers[2]);
    size_t const c_ny = STARPU_MATRIX_GET_NY(buffers[2]);
    size_t const c_ld = STARPU_MATRIX_GET_LD(buffers[2]);
    dahl_fp* c = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[2]);

    assert(a_len == c_nx);
    assert(b_len == c_ny);

    for (size_t y = 0; y < b_len; y++)
    {
        for (size_t x = 0; x < a_len; x++)
        {
            c[(y * c_ld) + x] += b[y] * a[x];
        }
    }
}

void vector_shuffle(void* buffers[1], void* cl_arg)
{
    size_t const vec_len = STARPU_VECTOR_GET_NX(buffers[0]);
    dahl_fp* vec = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    for (size_t i = vec_len - 1; i > 0; i--)
    {
        // Generate a random index respecting 0 <= j <= i
        size_t j = ( rand() / (RAND_MAX / i));
        assert(0 <= j && j <= i);

        // Swap the two values
        dahl_fp tmp = vec[i];
        vec[i] = vec[j];
        vec[j] = tmp;
    }
}

void vector_matrix_product(void* buffers[3], void* cl_arg)
{
    // Input vector
    size_t const vec_len = STARPU_VECTOR_GET_NX(buffers[0]);
    dahl_fp const* vec = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    // Input matrix
    size_t const mat_nx = STARPU_MATRIX_GET_NX(buffers[1]);
    size_t const mat_ny = STARPU_MATRIX_GET_NY(buffers[1]);
    size_t const mat_ld = STARPU_MATRIX_GET_LD(buffers[1]);
    dahl_fp const* mat = (dahl_fp*)STARPU_MATRIX_GET_PTR(buffers[1]);

    // Output vector
    size_t const out_len = STARPU_VECTOR_GET_NX(buffers[2]);
    dahl_fp* out = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[2]);

    assert(vec_len == mat_ny);
    assert(out_len == mat_nx);

    // Loop through x,y of the matrix
    for (size_t y = 0; y < mat_ny; y++)
    {
        for (size_t x = 0; x < mat_nx; x++)
        {
            out[x] += vec[y] * mat[(y * mat_ld) + x];
        }
    }
}

void vector_zero(void *buffers[1], void *cl_arg)
{
	// Vector
    size_t const nx = STARPU_VECTOR_GET_NX(buffers[0]);
    dahl_fp* data = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    for (int x = 0; x < nx; x++)
    {
        data[x] = 0;
    }
}

void vector_accumulate(void *buffers[2], void *cl_arg)
{
    // dst vector accumulator
	size_t const dst_len = STARPU_VECTOR_GET_NX(buffers[0]);
	dahl_fp* dst = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[0]);

    // src vector
	size_t const src_len = STARPU_VECTOR_GET_NX(buffers[1]);
	dahl_fp* const src = (dahl_fp*)STARPU_VECTOR_GET_PTR(buffers[1]);

    assert(dst_len == src_len);

    for (size_t x = 0; x < dst_len; x++)
    {
        dst[x] += src[x];
    }
}
