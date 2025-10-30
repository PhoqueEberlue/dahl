#include "../codelets.h"
#include "starpu_data_interfaces.h"
#include "starpu_task_util.h"
#include "../../../include/dahl_types.h"
#include "unistd.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>

// Get the ptr of any StarPU data type. Does not perform any check.
// This works because ptr is always the second field in the struct for vector, matrix, block and tensor,
// so it does not matter what we cast `interface` into. 
// This may be risky though, especially if the field order changes...
#define STARPU_ANY_GET_PTR(interface) (((struct starpu_vector_interface *)(interface))->ptr)

void any_relu(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        if (in[i] < 0.0F)
        {
            out[i] = 0.0F;
        }
        else 
        {
            out[i] = in[i];
        }
    }
}

void any_relu_backward(void* buffers[3], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* input = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp const* gradients = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[2]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        if (input[i] > 0.0F)
        {
            out[i] = gradients[i];
        }
        else 
        {
            out[i] = 0.0F;
        }
    }
}

void any_scal(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp factor;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &factor);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        out[i] = in[i] * factor;
    }
}

void any_power(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp power;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &power);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        out[i] = pow(in[i], power);
    }
}

void any_sub(void* buffers[3], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* a = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp const* b = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);
    dahl_fp* c = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[2]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        c[i] = a[i] - b[i];
    }
}

void any_add(void* buffers[3], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* a = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp const* b = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);
    dahl_fp* c = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[2]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void any_add_value(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp value;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &value);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        out[i] = in[i] + value;
    }
}

void any_clip(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp min;
    dahl_fp max;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &min, &max);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        if (in[i] > max)
        {
            out[i] = max;
        }
        else if (in[i] < min)
        {
            out[i] = min;
        }
        else
        {
            out[i] = in[i];
        }
    }
}

void any_sum(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        *out += in[i];
    }
}

void any_mean(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[1]);

    dahl_fp sum = 0.0F;

    for (size_t i = 0; i < nb_elem; i++)
    {
        sum += in[i];
    }

    *out = sum / (dahl_fp)nb_elem;
}

void any_fill(void* buffers[1], void* cl_arg)
{
    size_t nb_elem;
    dahl_fp value;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &value);

    dahl_fp* buf = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        buf[i] = value;
    }
}

// For debug purposes
void any_wait(void* buffers[1], void* cl_arg)
{
    unsigned int duration;
    starpu_codelet_unpack_args(cl_arg, &duration);
    usleep(duration);
}

void any_copy(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        out[i] = in[i];
    }
}

void any_min(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[1]);

    dahl_fp min = in[0];

    for (size_t i = 0; i < nb_elem; i++)
    {
        if (min > in[i]) { min = in[i]; }
    }

    *out = min;
}

void any_max(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_VARIABLE_GET_PTR(buffers[1]);

    dahl_fp max = in[0];

    for (size_t i = 0; i < nb_elem; i++)
    {
        if (max < in[i]) { max = in[i]; }
    }

    *out = max;
}

void any_round(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    int8_t precision;
    starpu_codelet_unpack_args(cl_arg, &nb_elem, &precision);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        out[i] = fp_round(in[i], precision);
    }
}
