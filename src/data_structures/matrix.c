#include "data_structures.h"
#include "starpu_data.h"
#include "starpu_data_filters.h"
#include "starpu_data_interfaces.h"
#include "sys/types.h"
#include <math.h>
#include <stdio.h>

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
    matrix->partition_type = DAHL_NONE;
    matrix->sub_handles = nullptr;
    matrix->nb_sub_handles = 0;

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

    starpu_data_acquire(matrix->handle, STARPU_R);
    dahl_matrix* res = matrix_init_from(shape, matrix->data);
    starpu_data_release(matrix->handle);

    return res;
}

dahl_shape2d matrix_get_shape(dahl_matrix const *const matrix)
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

dahl_fp* matrix_data_acquire_mutable(dahl_matrix* matrix)
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

void matrix_partition_along_y(dahl_matrix* matrix)
{
    // The data shouldn't be already partionned
    assert(matrix->partition_type == DAHL_NONE);

    size_t const nparts = matrix_get_shape(matrix).y;

    matrix->partition_type = DAHL_VECTOR;
    // Alloc space for the sub vectors
    matrix->sub_data.vectors = dahl_arena_alloc(nparts * sizeof(dahl_vector));
    // Pre allocate the data handles because they will be partitionned asynchronously
    matrix->sub_handles = (starpu_data_handle_t*)dahl_arena_alloc(nparts * sizeof(starpu_data_handle_t));
    matrix->nb_sub_handles = nparts; 

    // Now create the partition plan
    struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_pick_vector_y,
		.nchildren = nparts,
		.get_child_ops = starpu_matrix_filter_pick_vector_child_ops
	};

    // This call will update the sub_handles
	starpu_data_partition_plan(matrix->handle, &f, matrix->sub_handles);

    // Now we can copy the sub_handles in the vector wrappers
    for (int i = 0; i < nparts; i++)
    {
        matrix->sub_data.vectors[i].handle = matrix->sub_handles[i];
        matrix->sub_data.vectors[i].data = (dahl_fp*)starpu_vector_get_local_ptr(matrix->sub_handles[i]);
    }

    // And asynchronously submit
	starpu_data_partition_submit(matrix->handle, nparts, matrix->sub_handles);
}

// Custom filter
static void starpu_matrix_filter_vertical_matrix(
    void* parent_interface, void* child_interface, 
    STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter* f,
    unsigned id, unsigned nparts)
{
	struct starpu_matrix_interface* matrix_parent = (struct starpu_matrix_interface *) parent_interface;
	struct starpu_matrix_interface* matrix_child = (struct starpu_matrix_interface *) child_interface;

	unsigned blocksize;

	size_t nx = matrix_parent->nx;
	size_t ny = matrix_parent->ny;
	
    blocksize = matrix_parent->ld;

	size_t elemsize = matrix_parent->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	size_t new_ny = matrix_parent->ny / nparts;

	STARPU_ASSERT_MSG(nparts <= ny, "cannot get %u vectors", nparts);
	STARPU_ASSERT_MSG((chunk_pos + id) < ny, "the chosen sub matrix should be in the matrix");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(matrix_parent->id == STARPU_MATRIX_INTERFACE_ID, "%s can only be applied on a block data", __func__);
	matrix_child->id = STARPU_MATRIX_INTERFACE_ID;

    matrix_child->nx = nx;
    matrix_child->ny = new_ny;

	matrix_child->elemsize = elemsize;
	matrix_child->allocsize = matrix_child->nx * matrix_child->ny * elemsize;

	if (matrix_parent->dev_handle)
	{
		if (matrix_parent->ptr)
			matrix_child->ptr = matrix_parent->ptr + offset;
		
		matrix_child->dev_handle = matrix_parent->dev_handle;
		matrix_child->offset = matrix_parent->offset + offset;
	}
}

struct starpu_data_interface_ops *starpu_matrix_filter_pick_matrix_child_ops(
    STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_matrix_ops;
}

// FIXME: make it asynchronous
void matrix_partition_along_y_batch(dahl_matrix* matrix, size_t batch_size)
{
    // The data shouldn't be already partionned
    assert(matrix->partition_type == DAHL_NONE);

    dahl_shape2d const shape = matrix_get_shape(matrix);
    size_t const nparts = shape.y / batch_size;

    matrix->partition_type = DAHL_MATRIX;
    matrix->sub_data.matrices = dahl_arena_alloc(nparts * sizeof(dahl_matrix));
    matrix->sub_handles = (starpu_data_handle_t*)dahl_arena_alloc(nparts * sizeof(starpu_data_handle_t));
    matrix->nb_sub_handles = nparts;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_vertical_matrix,
		.nchildren = nparts,
		.get_child_ops = starpu_matrix_filter_pick_matrix_child_ops
	};

	starpu_data_partition_plan(matrix->handle, &f, matrix->sub_handles);
    
    for (int i = 0; i < nparts; i++)
    {
        matrix->sub_data.matrices[i].handle = matrix->sub_handles[i];
        matrix->sub_data.matrices[i].data = (dahl_fp*)starpu_matrix_get_local_ptr(matrix->sub_handles[i]);
        matrix->sub_data.matrices[i].partition_type = DAHL_NONE;
    }

    starpu_data_partition_submit(matrix->handle, nparts, matrix->sub_handles);
}

void matrix_unpartition(dahl_matrix* matrix)
{
    switch (matrix->partition_type)
    {
        case DAHL_NONE:
            printf("ERROR: Tried calling %s but the object is not partitioned", __func__);
            abort();
            break;
        case DAHL_TENSOR:
        case DAHL_BLOCK:
            printf("ERROR: got value %i in function %s, however matrix should only be partioned into vector or matrix", 
                   matrix->partition_type, __func__);
            abort();
            break;
        case DAHL_VECTOR:
            matrix->sub_data.vectors = nullptr;
            break;
        case DAHL_MATRIX:
            matrix->sub_data.matrices = nullptr;
            break;
    }

    matrix->partition_type = DAHL_NONE;
    starpu_data_unpartition_submit(matrix->handle, matrix->nb_sub_handles,
                                   matrix->sub_handles, STARPU_MAIN_RAM);

    // Clean the sub handles.
    // One improvement would be to reuse those for multiple partitionning call and only clean
    // when the application gets stopped. However, if we need a different number of handles it
    // makes it harder.
    starpu_data_partition_clean(matrix->handle, matrix->nb_sub_handles, matrix->sub_handles);
}

size_t matrix_get_nb_children(dahl_matrix const* matrix)
{
    assert(matrix->partition_type != DAHL_NONE);
    return matrix->nb_sub_handles;
}

dahl_vector* matrix_get_sub_vector(dahl_matrix const* matrix, const size_t index)
{
    assert(matrix->partition_type == DAHL_VECTOR 
        && matrix->sub_data.vectors != nullptr 
        && index < matrix->nb_sub_handles);

    return &matrix->sub_data.vectors[index];
}

dahl_matrix* matrix_get_sub_matrix(dahl_matrix const* matrix, size_t index)
{
    assert(matrix->partition_type == DAHL_MATRIX 
        && matrix->sub_data.matrices != nullptr 
        && index < matrix->nb_sub_handles);

    return &matrix->sub_data.matrices[index];
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
