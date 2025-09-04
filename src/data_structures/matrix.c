#include "data_structures.h"
#include "custom_filters.h"
#include "starpu_data.h"
#include "starpu_data_interfaces.h"
#include "sys/types.h"
#include <stdio.h>

void* _matrix_init_from_ptr(dahl_arena* arena, starpu_data_handle_t handle, dahl_fp* data)
{
    metadata* md = dahl_arena_alloc(
        arena,
        // Metadata struct itself
        sizeof(metadata) + 
        // + a flexible array big enough partition pointers to 
        // store all kinds of matrix partioning
        (MATRIX_NB_PARTITION_TYPE * sizeof(dahl_partition*)
    ));

    for (size_t i = 0; i < MATRIX_NB_PARTITION_TYPE; i++)
        md->partitions[i] = nullptr;

    md->current_partition = -1;
    md->origin_arena = arena;

    dahl_matrix* matrix = dahl_arena_alloc(arena, sizeof(dahl_matrix));
    matrix->handle = handle;
    matrix->data = data;
    matrix->meta = md;

    return matrix;
}

dahl_matrix* matrix_init(dahl_arena* arena, dahl_shape2d const shape)
{
    size_t n_elems = shape.x * shape.y;
    dahl_fp* data = dahl_arena_alloc(arena, n_elems * sizeof(dahl_fp));

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

    dahl_arena_attach_handle(arena, handle);

    return _matrix_init_from_ptr(arena, handle, data);
}

dahl_matrix* matrix_init_from(dahl_arena* arena, dahl_shape2d const shape, dahl_fp const* data)
{
    dahl_matrix* matrix = matrix_init(arena, shape);
    size_t n_elems = shape.x * shape.y;
    
    for (int i = 0; i < n_elems; i++)
    {
        matrix->data[i] = data[i];
    }

    return matrix;
}

dahl_matrix* matrix_init_random(dahl_arena* arena, dahl_shape2d const shape)
{
    dahl_matrix* matrix = matrix_init(arena, shape);
    size_t n_elems = shape.x * shape.y;

    for (int i = 0; i < n_elems; i += 1)
    {
        matrix->data[i] = (dahl_fp)( ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) );
    }

    return matrix;
}

dahl_tensor* matrix_to_tensor_no_copy(dahl_matrix const* matrix, dahl_shape4d const new_shape)
{
    dahl_shape2d shape = matrix_get_shape(matrix);
    assert(shape.x * shape.y == new_shape.x * new_shape.y * new_shape.z * new_shape.t);

    starpu_data_handle_t handle = nullptr;
    starpu_tensor_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)matrix->data,
        new_shape.x,
        new_shape.x*new_shape.y,
        new_shape.x*new_shape.y*new_shape.z,
        new_shape.x,
        new_shape.y,
        new_shape.z,
        new_shape.t,
        sizeof(dahl_fp)
    );

    dahl_arena_attach_handle(matrix->meta->origin_arena, handle);

    return _tensor_init_from_ptr(matrix->meta->origin_arena, handle, matrix->data);
}

void matrix_set_from(dahl_matrix* matrix, dahl_fp const* data)
{
    dahl_shape2d shape = matrix_get_shape(matrix);
    size_t nb_elems = shape.x * shape.y;

    matrix_data_acquire(matrix);

    for (int i = 0; i < nb_elems; i += 1)
    {
        matrix->data[i] = data[i];
    }

    matrix_data_release(matrix);
}

dahl_shape2d matrix_get_shape(dahl_matrix const* matrix)
{
    size_t nx = starpu_matrix_get_nx(matrix->handle);
    size_t ny = starpu_matrix_get_ny(matrix->handle);
    
    dahl_shape2d res = { .x = nx, .y = ny };
    return res;
}

starpu_data_handle_t _matrix_get_handle(void const* matrix)
{
    return ((dahl_matrix*)matrix)->handle;
}

dahl_partition* _matrix_get_current_partition(void const* matrix)
{
    metadata* m = ((dahl_matrix const*)matrix)->meta;
    assert(m-> current_partition >= 0 && 
           m->current_partition < MATRIX_NB_PARTITION_TYPE);

    assert(m->partitions[m->current_partition] != nullptr);
    return m->partitions[m->current_partition];
}

size_t _matrix_get_nb_elem(void const* matrix)
{
    dahl_shape2d shape = matrix_get_shape((dahl_matrix*)matrix);
    return shape.x * shape.y;
}

dahl_fp const* matrix_data_acquire(dahl_matrix const* matrix)
{
    starpu_data_acquire(matrix->handle, STARPU_R);
    return matrix->data;
}

dahl_fp* matrix_data_acquire_mut(dahl_matrix* matrix)
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

void _matrix_partition_along_y(dahl_matrix const* matrix, bool is_mut)
{
    assert(matrix->meta->current_partition == -1);
    matrix_partition_type t = MATRIX_PARTITION_ALONG_Y;

    // If the partition already exists, no need to create it.
    if (matrix->meta->partitions[t] != nullptr)
        goto submit;

    size_t const nparts = matrix_get_shape(matrix).y;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_pick_vector_y,
		.nchildren = nparts,
		.get_child_ops = starpu_matrix_filter_pick_vector_child_ops
	};

    // Create and set the partition
    matrix->meta->partitions[t] = _partition_init(nparts, is_mut, &dahl_traits_vector,
                                        &f, matrix->handle, matrix->meta->origin_arena);

submit:
    _partition_submit_if_needed(matrix->meta, t, is_mut, matrix->handle);
}

void matrix_partition_along_y_mut(dahl_matrix* matrix)
{
    _matrix_partition_along_y(matrix, true);
}

void matrix_partition_along_y(dahl_matrix const* matrix)
{
    _matrix_partition_along_y(matrix, false);
}

void _matrix_partition_along_y_batch(dahl_matrix const* matrix, size_t batch_size, bool is_mut)
{
    assert(matrix->meta->current_partition == -1);
    matrix_partition_type t = MATRIX_PARTITION_ALONG_Y_BATCH;
    size_t const nparts = matrix_get_shape(matrix).y / batch_size;

    dahl_partition* p = matrix->meta->partitions[MATRIX_PARTITION_ALONG_Y_BATCH];
    // If the partition already exists AND had the same batch size, no need to create it. 
    // FIX Warning, here the memory is lost if we create many partitions with different batch size
    if (p != nullptr && p->nb_children == nparts)
        goto submit;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_vertical_matrix,
		.nchildren = nparts,
		.get_child_ops = starpu_matrix_filter_pick_matrix_child_ops
	};

    // Create and set the partition
    matrix->meta->partitions[t] = _partition_init(nparts, is_mut, &dahl_traits_matrix,
                                        &f, matrix->handle, matrix->meta->origin_arena);

submit:
    _partition_submit_if_needed(matrix->meta, t, is_mut, matrix->handle);
}

void matrix_partition_along_y_batch_mut(dahl_matrix* matrix, size_t batch_size)
{
    _matrix_partition_along_y_batch(matrix, batch_size, true);
}

void matrix_partition_along_y_batch(dahl_matrix const* matrix, size_t batch_size)
{
    _matrix_partition_along_y_batch(matrix, batch_size, false);
}


void matrix_unpartition(dahl_matrix const* matrix)
{
    dahl_partition* p = matrix->meta->partitions[matrix->meta->current_partition];
    assert(p); // Shouldn't crash, an non-active partition is identified by the if bellow
    assert(matrix->meta->current_partition >= 0 && 
           matrix->meta->current_partition < MATRIX_NB_PARTITION_TYPE);

    matrix->meta->current_partition = -1;
    starpu_data_unpartition_submit(matrix->handle, p->nb_children,
                                   p->handles, STARPU_MAIN_RAM);

    // Note that the starpu handles for the children will be cleaned up at arena reseting.
    // This means that when submitting a new identical partition, the same handles will be reused.
}

void _matrix_print_file(void const* vmatrix, FILE* fp)
{
    auto matrix = (dahl_matrix const*)vmatrix;
    const dahl_shape2d shape = matrix_get_shape(matrix);

	size_t ld = starpu_matrix_get_local_ld(matrix->handle);

	starpu_data_acquire(matrix->handle, STARPU_R);

    fprintf(fp, "matrix=%p nx=%zu ny=%zu ld=%zu\n{ ", matrix->data, shape.x, shape.y, ld);
    for(size_t y = 0; y < shape.y; y++)
    {
        fprintf(fp, "\n\t{ ");
        for(size_t x = 0; x < shape.x; x++)
        {
            fprintf(fp, "%+.15f, ", matrix->data[(y*ld)+x]);
        }
        fprintf(fp, "},");
    }
    fprintf(fp, "\n}\n");

	starpu_data_release(matrix->handle);
}

void matrix_print(dahl_matrix const* matrix)
{
    _matrix_print_file(matrix, stdout);
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
