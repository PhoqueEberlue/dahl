#include "data_structures.h"

// See `block_init_from_ptr` for more information.
dahl_matrix* matrix_init_from_ptr(dahl_shape2d const shape, dahl_fp* data)
{
    starpu_data_handle_t handle = nullptr;

    // Under the hood, dahl_matrix is in fact a starpu_block with only 1 z dimension
    starpu_block_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        shape.x,
        shape.x*shape.y,
        shape.x,
        shape.y,
        1,
        sizeof(dahl_fp)
    );

    dahl_matrix* matrix = malloc(sizeof(dahl_matrix));
    matrix->handle = handle;
    matrix->data = data;
    matrix->is_sub_block_data = false;
    matrix->is_partitioned = false;
    matrix->sub_vectors = nullptr;

    return matrix;
}

dahl_matrix* matrix_init_from(dahl_shape2d const shape, dahl_fp const* data)
{
    size_t n_elems = shape.x * shape.y;
    dahl_fp* data_copy = malloc(n_elems * sizeof(dahl_fp));
    
    // TODO: memcpy doesn't work, it's not a big deal but it would be nice to understand why
    // memcpy(data_copy, data, n_elems);

    for (int i = 0; i < n_elems; i++)
    {
        data_copy[i] = data[i];
    }

    return matrix_init_from_ptr(shape, data_copy);
}

dahl_matrix* matrix_init_random(dahl_shape2d const shape)
{
    size_t n_elems = shape.x * shape.y;
    dahl_fp* data = malloc(n_elems * sizeof(dahl_fp));

    for (int i = 0; i < n_elems; i += 1)
    {
        data[i] = (dahl_fp)( ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) );
    }

    return matrix_init_from(shape, data);
}

// Initialize a starpu block at 0 and return its handle
dahl_matrix* matrix_init(dahl_shape2d const shape)
{
    size_t n_elems = shape.x * shape.y;
    dahl_fp* data = malloc(n_elems * sizeof(dahl_fp));

    for (int i = 0; i < n_elems; i += 1)
    {
        data[i] = 0;
    }

    return matrix_init_from_ptr(shape, data);
}

dahl_matrix* matrix_clone(dahl_matrix const* matrix)
{
    dahl_fp* data = matrix_data_acquire(matrix);
    dahl_shape2d shape = matrix_get_shape(matrix);

    dahl_matrix* res = matrix_init_from(shape, data);
    matrix_data_release((dahl_matrix*)matrix);

    return res;
}

dahl_shape2d matrix_get_shape(dahl_matrix const *const matrix)
{
    size_t nx = starpu_block_get_nx(matrix->handle);
    size_t ny = starpu_block_get_ny(matrix->handle);
    
    dahl_shape2d res = { .x = nx, .y = ny };
    return res;
}

dahl_fp* matrix_data_acquire(dahl_matrix const* matrix)
{
    starpu_data_acquire(matrix->handle, STARPU_RW);
    return matrix->data;
}

void matrix_data_release(dahl_matrix const* matrix)
{
    starpu_data_release(matrix->handle);
}

bool matrix_equals(dahl_matrix const* a, dahl_matrix const* b, bool const rounding)
{
    dahl_shape2d const shape_a = matrix_get_shape(a);
    dahl_shape2d const shape_b = matrix_get_shape(b);

    assert(shape_a.x == shape_b.x 
        && shape_a.y == shape_b.y);

    starpu_data_acquire(a->handle, STARPU_R);
    starpu_data_acquire(b->handle, STARPU_R);

    bool res = true;

    for (int i = 0; i < (shape_a.x * shape_a.y); i++)
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

void matrix_partition_along_y(dahl_matrix* const matrix)
{
    dahl_shape2d const shape = matrix_get_shape(matrix);

    struct starpu_data_filter f =
	{
        // Reminder that our matrix are implemented using starpu_block with only one Z dimension
		.filter_func = starpu_block_filter_vertical_block,
		.nchildren = shape.y,
	};

	starpu_data_partition(matrix->handle, &f);
    
    matrix->is_partitioned = true;
    matrix->sub_vectors = malloc(shape.y * sizeof(dahl_matrix));

    for (int i = 0; i < starpu_data_get_nb_children(matrix->handle); i++)
    {
		starpu_data_handle_t sub_vector_handle = starpu_data_get_sub_data(matrix->handle, 1, i);

        dahl_fp* data = (dahl_fp*)starpu_block_get_local_ptr(sub_vector_handle);

        matrix->sub_vectors[i].handle = sub_vector_handle;
        matrix->sub_vectors[i].data = data;
        matrix->sub_vectors[i].is_sub_matrix_data = false;
    }
}

void matrix_unpartition(dahl_matrix* const matrix)
{
    starpu_data_unpartition(matrix->handle, STARPU_MAIN_RAM);
    free(matrix->sub_vectors);
    matrix->sub_vectors = nullptr;
    matrix->is_partitioned = false;
}

size_t matrix_get_sub_vector_nb(dahl_matrix const* matrix)
{
    return starpu_data_get_nb_children(matrix->handle);
}

dahl_vector* matrix_get_sub_vector(dahl_matrix const* matrix, const size_t index)
{
    assert(matrix->is_partitioned 
        && matrix->sub_vectors != nullptr 
        && index < starpu_data_get_nb_children(matrix->handle));

    return &matrix->sub_vectors[index];
}

void matrix_print(dahl_matrix const* matrix)
{
    const dahl_shape2d shape = matrix_get_shape(matrix);

    // block ldy is equal to matrix ld
	size_t ld = starpu_block_get_local_ldy(matrix->handle);

	starpu_data_acquire(matrix->handle, STARPU_R);

    printf("matrix=%p nx=%zu ny=%zu ld=%zu\n", matrix->data, shape.x, shape.y, ld);

    for(size_t y = 0; y < shape.y; y++)
    {
        // printf("%s", space_offset(shape.y - y - 1));

        for(size_t x = 0; x < shape.x; x++)
        {
            printf("%f ", matrix->data[(y*ld)+x]);
        }
        printf("\n");
    }
    printf("\n");

	starpu_data_release(matrix->handle);
}

void matrix_print_ascii(dahl_matrix const* matrix, dahl_fp const threshold)
{
    const dahl_shape2d shape = matrix_get_shape(matrix);

    // block ldy is equal to matrix ld
	size_t ld = starpu_block_get_local_ldy(matrix->handle);

	starpu_data_acquire(matrix->handle, STARPU_R);

    printf("matrix=%p nx=%zu ny=%zu ld=%zu\n", matrix->data, shape.x, shape.y, ld);

    for(size_t y = 0; y < shape.y; y++)
    {
        for(size_t x = 0; x < shape.x; x++)
        {
            dahl_fp value = matrix->data[(y*ld)+x];

            value < threshold ? printf(". ") : printf("# ");
        }
        printf("\n");
    }
    printf("\n");

	starpu_data_release(matrix->handle);
}

// We don't have to free matrix->data because it should be managed by the user
void matrix_finalize(dahl_matrix* matrix)
{
    if (matrix->is_partitioned)
    {
        // Case where user forgot to unpartition data
        starpu_data_unpartition(matrix->handle, STARPU_MAIN_RAM);
    }

    if (matrix->is_sub_block_data)
    {
        printf("ERROR: matrix_finalize() shouldn't be used on sub block data because it will be freed by block_unpartition().");
        abort();
    }

    starpu_data_unregister(matrix->handle);
    free(matrix->data);
    free(matrix);
}
