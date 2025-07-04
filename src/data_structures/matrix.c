#include "data_structures.h"
#include "starpu_data_filters.h"
#include "sys/types.h"
#include <math.h>

dahl_matrix* matrix_init(dahl_shape2d const shape)
{
    size_t n_elems = shape.x * shape.y;
    dahl_fp* data = dahl_arena_alloc(n_elems * sizeof(dahl_fp));

    for (size_t i = 0; i < n_elems; i++)
        data[i] = 0.0F;

    starpu_data_handle_t handle = nullptr;
    starpu_matrix_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        shape.x,
        shape.x,
        shape.y,
        sizeof(dahl_fp)
    );

    dahl_arena_attach_handle(handle);

    dahl_matrix* matrix = dahl_arena_alloc(sizeof(dahl_matrix));
    matrix->handle = handle;
    matrix->data = data;
    matrix->is_sub_data = false;
    matrix->is_partitioned = false;
    matrix->sub_vectors = nullptr;

    return matrix;
}

dahl_matrix* matrix_init_from(dahl_shape2d const shape, dahl_fp const* data)
{
    dahl_matrix* matrix = matrix_init(shape);
    size_t n_elems = shape.x * shape.y;
    
    for (int i = 0; i < n_elems; i++)
    {
        matrix->data[i] = data[i];
    }

    return matrix;
}

dahl_matrix* matrix_init_random(dahl_shape2d const shape)
{
    dahl_matrix* matrix = matrix_init(shape);
    size_t n_elems = shape.x * shape.y;

    for (int i = 0; i < n_elems; i += 1)
    {
        matrix->data[i] = (dahl_fp)( ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) );
    }

    return matrix;
}

dahl_matrix* matrix_clone(dahl_matrix const* matrix)
{
    dahl_shape2d shape = matrix_get_shape(matrix);

    dahl_fp* data = matrix_data_acquire(matrix);
    dahl_matrix* res = matrix_init_from(shape, data);
    matrix_data_release(matrix);

    return res;
}

dahl_shape2d matrix_get_shape(dahl_matrix const *const matrix)
{
    starpu_data_acquire(matrix->handle, STARPU_R);
    size_t nx = starpu_matrix_get_nx(matrix->handle);
    size_t ny = starpu_matrix_get_ny(matrix->handle);
    starpu_data_release(matrix->handle);
    
    dahl_shape2d res = { .x = nx, .y = ny };
    return res;
}

starpu_data_handle_t _matrix_get_handle(void const* matrix)
{
    return ((dahl_matrix*)matrix)->handle;
}

size_t _matrix_get_nb_elem(void const* matrix)
{
    dahl_shape2d shape = matrix_get_shape((dahl_matrix*)matrix);
    return shape.x * shape.y;
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

bool matrix_equals(dahl_matrix const* a, dahl_matrix const* b, bool const rounding, u_int8_t const precision)
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

void matrix_partition_along_y(dahl_matrix* const matrix)
{
    dahl_shape2d const shape = matrix_get_shape(matrix);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_pick_vector_y,
		.nchildren = shape.y,
		.get_child_ops = starpu_matrix_filter_pick_vector_child_ops
	};

	starpu_data_partition(matrix->handle, &f);
    
    matrix->is_partitioned = true;
    matrix->sub_vectors = dahl_arena_alloc(shape.y * sizeof(dahl_vector));

    for (int i = 0; i < starpu_data_get_nb_children(matrix->handle); i++)
    {
		starpu_data_handle_t sub_vector_handle = starpu_data_get_sub_data(matrix->handle, 1, i);

        dahl_fp* data = (dahl_fp*)starpu_vector_get_local_ptr(sub_vector_handle);
        assert(data);

        matrix->sub_vectors[i].handle = sub_vector_handle;
        matrix->sub_vectors[i].data = data;
        matrix->sub_vectors[i].is_sub_data = true;
    }
}

void matrix_unpartition(dahl_matrix* const matrix)
{
    starpu_data_unpartition(matrix->handle, STARPU_MAIN_RAM);
    // No need to free the subvectors, they are managed by the arena, however we can reset the pointer
    matrix->sub_vectors = nullptr;
    matrix->is_partitioned = false;
}

size_t matrix_get_nb_children(dahl_matrix const* matrix)
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

	size_t ld = starpu_matrix_get_local_ld(matrix->handle);

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

	size_t ld = starpu_matrix_get_local_ld(matrix->handle);

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
