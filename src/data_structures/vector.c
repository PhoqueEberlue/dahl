#include "data_structures.h"
#include "starpu_data.h"
#include <math.h>

dahl_vector* vector_init(size_t const len)
{
    // Arena always returns 0 initialized data, no need to fill it
    dahl_fp* data = dahl_arena_alloc(len * sizeof(dahl_fp));

    starpu_data_handle_t handle = nullptr;
    starpu_vector_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        len,
        sizeof(dahl_fp)
    );

    dahl_arena_attach_handle(handle);

    dahl_vector* vector = dahl_arena_alloc(sizeof(dahl_vector));
    vector->handle = handle;
    vector->data = data;
    vector->is_sub_data = false;

    return vector;
}

dahl_vector* vector_init_from(size_t const len, dahl_fp const* data)
{
    dahl_vector* vector = vector_init(len);
    
    for (int i = 0; i < len; i++)
    {
        vector->data[i] = data[i];
    }

    return vector;
}

dahl_vector* vector_init_random(size_t const len)
{
    dahl_vector* vector = vector_init(len);

    for (int i = 0; i < len; i += 1)
    {
        vector->data[i] = (dahl_fp)( ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) );
    }

    return vector;
}

dahl_vector* vector_clone(dahl_vector const* vector)
{
    size_t shape = vector_get_len(vector);

    starpu_data_acquire(vector->handle, STARPU_R);
    dahl_vector* res = vector_init_from(shape, vector->data);
    starpu_data_release(vector->handle);

    return res;
}

size_t vector_get_len(dahl_vector const *const vector)
{
    starpu_data_acquire(vector->handle, STARPU_R);
    size_t nx = starpu_vector_get_nx(vector->handle);
    starpu_data_release(vector->handle);
    return nx;
}

dahl_fp* vector_data_acquire(dahl_vector const* vector)
{
    starpu_data_acquire(vector->handle, STARPU_RW);
    return vector->data;
}

void vector_data_release(dahl_vector const* vector)
{
    starpu_data_release(vector->handle);
}

bool vector_equals(dahl_vector const* a, dahl_vector const* b, bool const rounding, u_int8_t const precision)
{
    size_t const len_a = vector_get_len(a);
    size_t const len_b = vector_get_len(b);

    assert(len_a == len_b);

    starpu_data_acquire(a->handle, STARPU_R);
    starpu_data_acquire(b->handle, STARPU_R);

    bool res = true;

    for (int i = 0; i < len_a; i++)
    {
        if (rounding)
        {
            if (fp_round(a->data[i], precision) != fp_round(b->data[i], precision))
            {
                res = false;
                break;
            }
        }
        else 
        {
            if (a->data[i] != b->data[i])
            {
                res = false;
                break;
            }
        }
    }

    starpu_data_release(a->handle);
    starpu_data_release(b->handle);

    return res;
}

void vector_print(dahl_vector const* vector)
{
    const size_t len = vector_get_len(vector);

	starpu_data_acquire(vector->handle, STARPU_R);

    printf("vector=%p nx=%zu\n", vector->data, len);

    for(size_t x = 0; x < len; x++)
    {
        printf("%f ", vector->data[x]);
    }
    printf("\n");

	starpu_data_release(vector->handle);
}

// FIXME 
dahl_matrix* vector_to_matrix(dahl_vector* vector, dahl_shape2d shape)
{
    size_t len = vector_get_len(vector);

    starpu_data_acquire(vector->handle, STARPU_R);

    assert(shape.x * shape.y == len);

    dahl_matrix* res = matrix_init_from(shape, vector->data);

    starpu_data_release(vector->handle);

    return res;
}

// FIXME 
dahl_matrix* vector_to_column_matrix(dahl_vector* vector)
{
    size_t len = vector_get_len(vector);
    dahl_shape2d new_shape = { .x = 1, .y = len };
    return vector_to_matrix(vector, new_shape);
}

// FIXME 
dahl_matrix* vector_to_row_matrix(dahl_vector* vector)
{
    size_t len = vector_get_len(vector);
    dahl_shape2d new_shape = { .x = len, .y = 1 };
    return vector_to_matrix(vector, new_shape);
}

// FIXME 
dahl_block* vector_to_block(dahl_vector* vector, dahl_shape3d shape)
{
    size_t len = vector_get_len(vector);
    starpu_data_acquire(vector->handle, STARPU_R);

    assert(shape.x * shape.y * shape.z == len);

    dahl_block* res = block_init_from(shape, vector->data);

    starpu_data_release(vector->handle);

    return res;
}

// FIXME 
// TODO: why wouldn't it be a codelet?
dahl_matrix* vector_as_categorical(dahl_vector const* vector, size_t const num_classes)
{
    size_t len = vector_get_len(vector);

    starpu_data_acquire(vector->handle, STARPU_R);

    dahl_shape2d shape = { .x = num_classes, .y = len };
    dahl_matrix* matrix = matrix_init(shape);

    starpu_data_acquire(matrix->handle, STARPU_W);

    for (size_t i = 0; i < len; i++)
    {
        size_t class = (size_t)vector->data[i];
        matrix->data[(i * num_classes) + class] = 1;
    }

    starpu_data_release(vector->handle);
    starpu_data_release(matrix->handle);

    return matrix;
}
