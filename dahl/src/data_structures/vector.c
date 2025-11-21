#include "data_structures.h"
#include "starpu.h"
#include "starpu_data.h"
#include "starpu_data_interfaces.h"
#include "../tasks/codelets.h"
#include <stdio.h>

void* _vector_init_from_ptr(dahl_arena* arena, starpu_data_handle_t handle, dahl_fp* data)
{
    // The vector cannot be partitioned so we don't allocate space for the partition list
    metadata* md = dahl_arena_alloc(arena, sizeof(metadata));
    md->current_partition = -1;
    md->origin_arena = arena;

    dahl_vector* vector = dahl_arena_alloc(arena, sizeof(dahl_vector));
    vector->handle = handle;
    vector->data = data;
    vector->is_redux = false;
    vector->meta = md;

    return vector;
}

dahl_fp* _vector_data_alloc(dahl_arena* arena, size_t const len)
{
    return dahl_arena_alloc(arena, len * sizeof(dahl_fp));
}

starpu_data_handle_t _vector_data_register(dahl_arena* arena, size_t const len, dahl_fp* data)
{
    starpu_data_handle_t handle = nullptr;
    starpu_vector_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        len,
        sizeof(dahl_fp)
    );

    dahl_arena_attach_handle(arena, handle);
    return handle;
}

dahl_vector* vector_init(dahl_arena* arena, size_t const len)
{
    dahl_fp* data = _vector_data_alloc(arena, len);
    memset(data, 0, len * sizeof(dahl_fp));

    starpu_data_handle_t handle = _vector_data_register(arena, len, data);
    dahl_vector* vector = _vector_init_from_ptr(arena, handle, data);
    return vector;
}

dahl_vector* vector_init_redux(dahl_arena* arena, size_t const len)
{
    dahl_vector* vector = vector_init(arena, len);
    // Enable redux mode
    _vector_enable_redux(vector);
    return vector;
}

dahl_vector* vector_init_from(dahl_arena* arena, size_t const len, dahl_fp const* data)
{
    dahl_fp* vector_data = _vector_data_alloc(arena, len);

    for (size_t i = 0; i < len; i++)
         vector_data[i] = data[i];

    starpu_data_handle_t handle = _vector_data_register(arena, len, vector_data);
    return _vector_init_from_ptr(arena, handle, vector_data);
}

dahl_vector* vector_init_random(dahl_arena* arena, size_t const len, dahl_fp min, dahl_fp max)
{
    dahl_fp* vector_data = _vector_data_alloc(arena, len);

    for (size_t x = 0; x < len; x++)
        vector_data[x] = fp_rand(min, max);

    starpu_data_handle_t handle = _vector_data_register(arena, len, vector_data);
    return _vector_init_from_ptr(arena, handle, vector_data);
}

size_t vector_get_len(dahl_vector const *const vector)
{
    return starpu_vector_get_nx(vector->handle);
}

starpu_data_handle_t _vector_get_handle(void const* vector)
{
    return ((dahl_vector*)vector)->handle;
}

void _vector_enable_redux(void* vector)
{
    ((dahl_vector*)vector)->is_redux = true;
    starpu_data_set_reduction_methods(
            ((dahl_vector*)vector)->handle, &cl_vector_accumulate, &cl_vector_zero);
}

bool _vector_get_is_redux(void const* vector)
{
    return ((dahl_vector const*)vector)->is_redux;
}

size_t _vector_get_nb_elem(void const* vector)
{
    return vector_get_len((dahl_vector*)vector);
}

inline dahl_fp vector_get_value(dahl_vector const* vector, size_t index)
{
    return vector->data[index];
}

inline void vector_set_value(dahl_vector* vector, size_t index, dahl_fp value)
{
    vector->data[index] = value;
}

void vector_set_from(dahl_vector* vector, dahl_fp const* data)
{
    size_t len = starpu_vector_get_nx(vector->handle);

    vector_acquire(vector);

    size_t i = 0;
    for (size_t x = 0; x < len; x++)
    {
        vector_set_value(vector, x, data[i]);
        i++;
    }

    vector_release(vector);
}

void vector_acquire(dahl_vector const* vector)
{
    starpu_data_acquire(vector->handle, STARPU_R);
}

void vector_acquire_mut(dahl_vector* vector)
{
    starpu_data_acquire(vector->handle, STARPU_RW);
}

void vector_release(dahl_vector const* vector)
{
    starpu_data_release(vector->handle);
}

bool vector_equals(dahl_vector const* a, dahl_vector const* b, bool const rounding, int8_t const precision)
{
    size_t const len_a = vector_get_len(a);
    size_t const len_b = vector_get_len(b);

    assert(len_a == len_b);

    vector_acquire(a);
    vector_acquire(b);

    bool res = true;

    for (size_t x = 0; x < len_a; x++)
    {
        dahl_fp a_val = vector_get_value(a, x);
        dahl_fp b_val = vector_get_value(b, x);

        if (rounding) { res = fp_equals_round(a_val, b_val, precision); }
        else          { res = fp_equals(a_val, b_val);                  }

        if (!res)     { break; }
    }

    vector_release(a);
    vector_release(b);

    return res;
}

dahl_block* vector_to_block_no_copy(dahl_vector const* vector, dahl_shape3d const new_shape)
{
    size_t len = vector_get_len(vector);
    assert(len == new_shape.x * new_shape.y * new_shape.z);

    // Registers our vector data as a block (with new shape), handle will be attached to the
    // vector's origin arena.
    starpu_data_handle_t handle = _block_data_register(
            vector->meta->origin_arena, new_shape, vector->data);
    
    dahl_block* res = _block_init_from_ptr(vector->meta->origin_arena, handle, vector->data);

    // Here we use the same trick when doing manual partitioning:
    // Use cl_switch to force data refresh in our new handle from the vector handle
	int ret = starpu_task_insert(&cl_switch, STARPU_RW, vector->handle, STARPU_W, handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "vector_flatten_along_t_no_copy");

    // Then deactivate the vector handle
    starpu_data_invalidate_submit(vector->handle);

    return res;
}

void _vector_print_file(void const* vvector, FILE* fp, int8_t const precision)
{
    auto vector = (dahl_vector const*)vvector;
    const size_t len = vector_get_len(vector);

	vector_acquire(vector);

    fprintf(fp, "vector=%p nx=%zu\n{ ", vector->data, len);
    for(size_t x = 0; x < len; x++)
    {
        fprintf(fp, "%+.*f, ", precision, vector_get_value(vector, x));
    }
    fprintf(fp, "}\n");

	vector_release(vector);
}

void vector_print(dahl_vector const* vector)
{
    _vector_print_file(vector, stdout, DAHL_DEFAULT_PRINT_PRECISION);
}

dahl_matrix* vector_to_categorical(dahl_arena* arena, dahl_vector const* vector, size_t const num_classes)
{
    size_t len = vector_get_len(vector);
    dahl_shape2d shape = { .x = num_classes, .y = len };
    dahl_matrix* matrix = matrix_init(arena, shape);

    vector_acquire(vector);
    matrix_acquire_mut(matrix);

    for (size_t y = 0; y < len; y++)
    {
        size_t class_index = (size_t)vector_get_value(vector, y);
        // Activate value at position `class_index`
        matrix_set_value(matrix, class_index, y, 1);
    }

    vector_release(vector);
    matrix_release(matrix);
    return matrix;
}
