#include "data_structures.h"

// See `block_init_from_ptr` for more information.
dahl_vector* vector_init_from_ptr(size_t const len, dahl_fp* data)
{
    starpu_data_handle_t handle = nullptr;

    // Under the hood, dahl_vector is in fact a starpu_block with only 1 y and z dimensions
    starpu_block_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        len,   // TODO: is it the len?
        len,   // TODO: is it the len?
        len, // nx
        1,   // ny
        1,   // nz
        sizeof(dahl_fp)
    );

    dahl_vector* vector = malloc(sizeof(dahl_vector));
    vector->handle = handle;
    vector->data = data;
    vector->is_sub_matrix_data = false;

    return vector;
}

dahl_vector* vector_init_from(size_t const len, dahl_fp const* data)
{
    dahl_fp* data_copy = malloc(len * sizeof(dahl_fp));
    
    for (int i = 0; i < len; i++)
    {
        data_copy[i] = data[i];
    }

    return vector_init_from_ptr(len, data_copy);
}

dahl_vector* vector_init_random(size_t const len)
{
    dahl_fp* data = malloc(len * sizeof(dahl_fp));

    for (int i = 0; i < len; i += 1)
    {
        data[i] = (dahl_fp)( ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) );
    }

    return vector_init_from(len, data);
}

// Initialize a starpu block at 0 and return its handle
dahl_vector* vector_init(size_t const len)
{
    dahl_fp* data = malloc(len * sizeof(dahl_fp));

    for (int i = 0; i < len; i += 1)
    {
        data[i] = 0;
    }

    return vector_init_from_ptr(len, data);
}

dahl_vector* vector_clone(dahl_vector const* vector)
{
    dahl_fp* data = vector_data_acquire(vector);
    size_t shape = vector_get_len(vector);

    dahl_vector* res = vector_init_from(shape, data);
    vector_data_release((dahl_vector*)vector);

    return res;
}

size_t vector_get_len(dahl_vector const *const vector)
{
    return starpu_block_get_nx(vector->handle);
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

bool vector_equals(dahl_vector const* a, dahl_vector const* b, bool const rounding)
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
            if (round(a->data[i]) != round(b->data[i]))
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
        printf("%e ", vector->data[x]);
    }
    printf("\n");

	starpu_data_release(vector->handle);
}

void vector_finalize_without_data(dahl_vector* vector)
{
    if (vector->is_sub_matrix_data)
    {
        printf("ERROR: vector_finalize_without_data() shouldn't be used on sub matrix data because it will be freed by matrix_unpartition().");
        abort();
    }

    starpu_data_unregister(vector->handle);
    free(vector);
}

// We don't have to free matrix->data because it should be managed by the user
void vector_finalize(dahl_vector* vector)
{
    if (vector->is_sub_matrix_data)
    {
        printf("ERROR: vector_finalize() shouldn't be used on sub matrix data because it will be freed by matrix_unpartition().");
        abort();
    }

    starpu_data_unregister(vector->handle);
    free(vector->data);
    free(vector);
}

dahl_matrix* vector_to_matrix(dahl_vector* vector, dahl_shape2d shape)
{
    size_t len = vector_get_len(vector);
    dahl_fp* data = vector_data_acquire(vector);

    assert(shape.x * shape.y == len);

    dahl_matrix* res = matrix_init_from_ptr(shape, data);

    vector_data_release(vector);
    vector_finalize_without_data(vector);

    return res;
}

dahl_matrix* vector_to_column_matrix(dahl_vector* vector)
{
    size_t len = vector_get_len(vector);
    dahl_shape2d new_shape = { .x = 1, .y = len };
    return vector_to_matrix(vector, new_shape);
}

dahl_matrix* vector_to_row_matrix(dahl_vector* vector)
{
    size_t len = vector_get_len(vector);
    dahl_shape2d new_shape = { .x = len, .y = 1 };
    return vector_to_matrix(vector, new_shape);
}

dahl_block* vector_to_block(dahl_vector* vector, dahl_shape3d shape)
{
    size_t len = vector_get_len(vector);
    dahl_fp* data = vector_data_acquire(vector);

    assert(shape.x * shape.y * shape.z == len);

    dahl_block* res = block_init_from_ptr(shape, data);

    vector_data_release(vector);
    vector_finalize_without_data(vector);

    return res;
}

// TODO: why wouldn't it be a codelet?
dahl_matrix* vector_as_categorical(dahl_vector const* vector, size_t const num_classes)
{
    dahl_fp* vec = vector_data_acquire(vector);
    size_t len = vector_get_len(vector);

    dahl_shape2d shape = { .x = num_classes, .y = len };
    dahl_matrix* res = matrix_init(shape);

    dahl_fp* mat = matrix_data_acquire(res);

    for (size_t i = 0; i < len; i++)
    {
        size_t class = (size_t)vec[i];
        mat[(i * num_classes) + class] = 1;
    }

    vector_data_release(vector);
    matrix_data_release(res);

    return res;
}
