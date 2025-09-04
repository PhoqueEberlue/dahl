#include "data_structures.h"
#include "starpu_data.h"
#include "starpu_data_interfaces.h"

void* _vector_init_from_ptr(dahl_arena* arena, starpu_data_handle_t handle, dahl_fp* data)
{
    // The vector cannot be partitioned so we don't allocate space for the partition list
    metadata* md = dahl_arena_alloc(arena, sizeof(metadata));
    md->current_partition = -1;
    md->origin_arena = arena;

    dahl_vector* vector = dahl_arena_alloc(arena, sizeof(dahl_vector));
    vector->handle = handle;
    vector->data = data;

    return vector;
}

dahl_vector* vector_init(dahl_arena* arena, size_t const len)
{
    dahl_fp* data = dahl_arena_alloc(arena, len * sizeof(dahl_fp));

    for (size_t i = 0; i < len; i++)
        data[i] = 0.0F;

    starpu_data_handle_t handle = nullptr;
    starpu_vector_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        len,
        sizeof(dahl_fp)
    );

    dahl_arena_attach_handle(arena, handle);

    return _vector_init_from_ptr(arena, handle, data);
}

dahl_vector* vector_init_from(dahl_arena* arena, size_t const len, dahl_fp const* data)
{
    dahl_vector* vector = vector_init(arena, len);
    
    for (int i = 0; i < len; i++)
    {
        vector->data[i] = data[i];
    }

    return vector;
}

dahl_vector* vector_init_random(dahl_arena* arena, size_t const len)
{
    dahl_vector* vector = vector_init(arena, len);

    for (int i = 0; i < len; i += 1)
    {
        vector->data[i] = (dahl_fp)( ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) );
    }

    return vector;
}

size_t vector_get_len(dahl_vector const *const vector)
{
    return starpu_vector_get_nx(vector->handle);
}

starpu_data_handle_t _vector_get_handle(void const* vector)
{
    return ((dahl_vector*)vector)->handle;
}

size_t _vector_get_nb_elem(void const* vector)
{
    return vector_get_len((dahl_vector*)vector);
}

dahl_fp vector_get_value(dahl_vector const* vector, size_t index)
{
    assert(index < starpu_vector_get_nx(vector->handle));
    vector_data_acquire(vector);
    dahl_fp res = vector->data[index];
    vector_data_release(vector);
    return res;
}

void vector_set_value(dahl_vector* vector, size_t index, dahl_fp value)
{
    assert(index < starpu_vector_get_nx(vector->handle));
    vector_data_acquire(vector);
    vector->data[index] = value;
    vector_data_release(vector);
}

void vector_set_from(dahl_vector* vector, dahl_fp const* data)
{
    size_t len = starpu_vector_get_nx(vector->handle);

    vector_data_acquire(vector);

    for (int i = 0; i < len; i += 1)
    {
        vector->data[i] = data[i];
    }

    vector_data_release(vector);
}

dahl_fp const* vector_data_acquire(dahl_vector const* vector)
{
    starpu_data_acquire(vector->handle, STARPU_R);
    return vector->data;
}

dahl_fp* vector_data_acquire_mut(dahl_vector* vector)
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

void _vector_print_file(void const* vvector, FILE* fp)
{
    auto vector = (dahl_vector const*)vvector;
    const size_t len = vector_get_len(vector);

	starpu_data_acquire(vector->handle, STARPU_R);

    fprintf(fp, "vector=%p nx=%zu\n{ ", vector->data, len);
    for(size_t x = 0; x < len; x++)
    {
        fprintf(fp, "%+.15f", vector->data[x]);

        // Omit last comma
        if (x != len - 1)
            fprintf(fp, ", ");
    }
    fprintf(fp, " }\n");

	starpu_data_release(vector->handle);
}

void vector_print(dahl_vector const* vector)
{
    _vector_print_file(vector, stdout);
}

dahl_matrix* vector_to_categorical(dahl_arena* arena, dahl_vector const* vector, size_t const num_classes)
{
    size_t len = vector_get_len(vector);

    starpu_data_acquire(vector->handle, STARPU_R);

    dahl_shape2d shape = { .x = num_classes, .y = len };
    dahl_matrix* matrix = matrix_init(arena, shape);

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
