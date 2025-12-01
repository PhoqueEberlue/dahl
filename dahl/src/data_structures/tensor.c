#include "data_structures.h"
#include "../tasks/codelets.h"
#include "../misc.h"
#include "starpu_data.h"
#include "starpu_data_filters.h"
#include "starpu_data_interfaces.h"
#include "custom_filters.h"
#include "starpu_util.h"
#include "sys/types.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void* _tensor_init_from_ptr(dahl_arena* arena, starpu_data_handle_t handle, dahl_fp* data)
{
    dahl_tensor* tensor = dahl_arena_alloc(arena, sizeof(dahl_tensor));
    tensor->handle = handle;
    tensor->data = data;
    tensor->origin_arena = arena;
    tensor->is_redux = false;
    tensor->partition = (dahl_partition**)dahl_arena_alloc(arena, sizeof(dahl_partition**));

    return tensor;
}

dahl_fp* _tensor_data_alloc(dahl_arena* arena, dahl_shape4d const shape)
{
    size_t n_elems = shape.x * shape.y * shape.z * shape.t;
    return dahl_arena_alloc(arena, n_elems * sizeof(dahl_fp));
}

starpu_data_handle_t _tensor_data_register(
        dahl_arena* arena, dahl_shape4d const shape, dahl_fp* data)
{
    starpu_data_handle_t handle = nullptr;
    starpu_tensor_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        shape.x,
        shape.x*shape.y,
        shape.x*shape.y*shape.z,
        shape.x,
        shape.y,
        shape.z,
        shape.t,
        sizeof(dahl_fp)
    );

    dahl_arena_attach_handle(arena, handle);
    return handle;
}

dahl_tensor* tensor_init(dahl_arena* arena, dahl_shape4d const shape)
{
    dahl_fp* data = _tensor_data_alloc(arena, shape);
    memset(data, 0, shape.x*shape.y*shape.z*shape.t * sizeof(dahl_fp));
    starpu_data_handle_t handle = _tensor_data_register(arena, shape, data);
    dahl_tensor* tensor = _tensor_init_from_ptr(arena, handle, data);
    return tensor;
}

dahl_tensor* tensor_init_redux(dahl_arena* arena, dahl_shape4d const shape)
{
    dahl_tensor* tensor = tensor_init(arena, shape);
    // Enable redux mode
    _tensor_enable_redux(tensor);
    return tensor;
}

dahl_tensor* tensor_init_from(dahl_arena* arena, dahl_shape4d const shape, dahl_fp const* data)
{
    dahl_fp* tensor_data = _tensor_data_alloc(arena, shape);

    for (size_t i = 0; i < shape.x*shape.y*shape.z*shape.t; i++)
         tensor_data[i] = data[i];

    starpu_data_handle_t handle = _tensor_data_register(arena, shape, tensor_data);
    return _tensor_init_from_ptr(arena, handle, tensor_data);
}

dahl_tensor* tensor_init_random(
        dahl_arena* arena, dahl_shape4d const shape, dahl_fp min, dahl_fp max)
{
    dahl_fp* tensor_data = _tensor_data_alloc(arena, shape);

    for (size_t t = 0; t < shape.t; t++)
    {
        for (size_t z = 0; z < shape.z; z++)
        {
            for (size_t y = 0; y < shape.y; y++)
            {
                for (size_t x = 0; x < shape.x; x++)
                {
                    size_t index = (t * shape.x * shape.y * shape.z) +
                                   (z * shape.x * shape.y) +
                                   (y * shape.x) + x;
                    tensor_data[index] = fp_rand(min, max);
                }
            }
        }
    }

    starpu_data_handle_t handle = _tensor_data_register(arena, shape, tensor_data);
    return _tensor_init_from_ptr(arena, handle, tensor_data);
}

void tensor_set_from(dahl_tensor* tensor, dahl_fp const* data)
{
    dahl_shape4d shape = tensor_get_shape(tensor);
    tensor_acquire(tensor);

    size_t i = 0;
    for (size_t t = 0; t < shape.t; t++)
    {
        for (size_t z = 0; z < shape.z; z++)
        {
            for (size_t y = 0; y < shape.y; y++)
            {
                for (size_t x = 0; x < shape.x; x++)
                {
                    tensor_set_value(tensor, x, y, z, t, data[i]);
                    i++;
                }
            }
        }
    }

    tensor_release(tensor);
}

void _tensor_enable_redux(void* tensor)
{
    ((dahl_tensor*)tensor)->is_redux = true;
    starpu_data_set_reduction_methods(
            ((dahl_tensor*)tensor)->handle, &cl_tensor_accumulate, &cl_tensor_zero);
}

bool _tensor_get_is_redux(void const* tensor)
{
    return ((dahl_tensor const*)tensor)->is_redux;
}

inline dahl_fp tensor_get_value(dahl_tensor const* tensor, size_t x, size_t y, size_t z, size_t t)
{
    size_t ldy = starpu_tensor_get_local_ldy(tensor->handle);
    size_t ldz = starpu_tensor_get_local_ldz(tensor->handle);
    size_t ldt = starpu_tensor_get_local_ldt(tensor->handle);
    return tensor->data[(t * ldt) + (z * ldz) + (y * ldy) + x];
}

inline void tensor_set_value(dahl_tensor* tensor, size_t x, size_t y, size_t z, size_t t, dahl_fp value)
{
    size_t ldy = starpu_tensor_get_local_ldy(tensor->handle);
    size_t ldz = starpu_tensor_get_local_ldz(tensor->handle);
    size_t ldt = starpu_tensor_get_local_ldt(tensor->handle);
    tensor->data[(t * ldt) + (z * ldz) + (y * ldy) + x] = value;
}

dahl_matrix* tensor_flatten_along_t_no_copy(dahl_tensor const* tensor)
{
    dahl_shape4d shape = tensor_get_shape(tensor);
    dahl_shape2d new_shape = { .x = shape.x * shape.y * shape.z, .y = shape.t };

    // Registers our tensor data as a matrix (with new shape), handle will be attached to the
    // tensor's origin arena.
    starpu_data_handle_t handle = _matrix_data_register(
            tensor->origin_arena, new_shape, tensor->data);
    
    dahl_matrix* res = _matrix_init_from_ptr(tensor->origin_arena, handle, tensor->data);

    // Here we use the same trick when doing manual partitioning:
    // Use cl_switch to force data refresh in our new handle from the tensor handle
	int ret = starpu_task_insert(&cl_switch, STARPU_RW, tensor->handle, STARPU_W, handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "tensor_flatten_along_t_no_copy");

    // Then deactivate the tensor handle
    starpu_data_invalidate_submit(tensor->handle);

    return res;
}

dahl_matrix_part* tensor_flatten_along_t_no_copy_partition(dahl_tensor_part* tensor)
{
    // Initialize a matrix, just to hold the childrens
    dahl_shape4d shape = tensor_get_shape(tensor);
    dahl_shape2d new_shape = { .x = shape.x * shape.y * shape.z, .y = shape.t };

    // Registers our tensor data as a matrix (with new shape), handle will be attached to the
    // tensor's origin arena.
    starpu_data_handle_t handle = _matrix_data_register(
            tensor->origin_arena, new_shape, tensor->data);
    
    dahl_matrix* matrix = _matrix_init_from_ptr(tensor->origin_arena, handle, tensor->data);

    size_t const batch_size = GET_NB_CHILDREN(tensor);

    // Create a fake partition to hold our flat children
    dahl_partition* p = dahl_arena_alloc(
        tensor->origin_arena,
        // The partition object itself
        sizeof(dahl_partition) + 
        // + the children array with enough space to store their pointers
        (batch_size * sizeof(void*))
    );

    p->handles = (starpu_data_handle_t*)dahl_arena_alloc(
        tensor->origin_arena,
        batch_size * sizeof(starpu_data_handle_t));
    p->access = DAHL_READ;
    p->nb_children = batch_size;
    p->trait = &dahl_traits_vector;
    p->main_handle = tensor->handle;
    p->type = FAKE_PARTITION;

    for (size_t i = 0; i < batch_size; i++)
    {
        dahl_block const* block = GET_SUB_BLOCK(tensor, i);
        dahl_vector* flat_child = block_flatten_no_copy(block);
        p->children[i] = flat_child;
        p->handles[i] = flat_child->handle;

        // Invalidate each children of the origin tensor
        starpu_data_invalidate_submit(block->handle);
    }

    // Pretend the matrix is partitioned
    *matrix->partition = p;
    return matrix;
}

dahl_shape4d tensor_get_shape(dahl_tensor const* tensor)
{
    size_t nx = starpu_tensor_get_nx(tensor->handle);
	size_t ny = starpu_tensor_get_ny(tensor->handle);
	size_t nz = starpu_tensor_get_nz(tensor->handle);
	size_t nt = starpu_tensor_get_nt(tensor->handle);
    dahl_shape4d res = { .x = nx, .y = ny, .z = nz, .t = nt };

    return res;
}

starpu_data_handle_t _tensor_get_handle(void const* tensor)
{
    return ((dahl_tensor*)tensor)->handle;
}

size_t _tensor_get_nb_elem(void const* tensor)
{
    dahl_shape4d shape = tensor_get_shape((dahl_tensor*)tensor);
    return shape.x * shape.y * shape.z * shape.t;
}

void tensor_acquire(dahl_tensor const* tensor)
{
    starpu_data_acquire(tensor->handle, STARPU_R);
}

void tensor_acquire_mut(dahl_tensor* tensor)
{
    starpu_data_acquire(tensor->handle, STARPU_RW);
}

void tensor_release(dahl_tensor const* tensor)
{
    starpu_data_release(tensor->handle);
}

bool tensor_equals(dahl_tensor const* a, dahl_tensor const* b, bool const rounding, int8_t const precision)
{
    dahl_shape4d shape_a = tensor_get_shape(a);
    dahl_shape4d shape_b = tensor_get_shape(b);

    assert(shape_a.x == shape_b.x 
        && shape_a.y == shape_b.y 
        && shape_a.z == shape_b.z 
        && shape_a.t == shape_b.t);

    tensor_acquire(a);
    tensor_acquire(b);

    bool res = true;

    for (size_t t = 0; t < shape_a.t; t++)
    {
        for (size_t z = 0; z < shape_a.z; z++)
        {
            for (size_t y = 0; y < shape_a.y; y++)
            {
                for (size_t x = 0; x < shape_a.x; x++)
                {
                    dahl_fp a_val = tensor_get_value(a, x, y, z, t);
                    dahl_fp b_val = tensor_get_value(b, x, y, z, t);

                    if (rounding) { res = fp_equals_round(a_val, b_val, precision); }
                    else          { res = fp_equals(a_val, b_val);                  }

                    if (!res)     { goto RELEASE; }
                }
            }
        }
    }

RELEASE:
    tensor_release(a);
    tensor_release(b);

    return res;
}

dahl_partition* _tensor_get_partition(void const* tensor)
{
    return *((dahl_tensor*)tensor)->partition;
}

void tensor_partition_along_t(dahl_tensor const* tensor, dahl_access access)
{
    size_t const nparts = tensor_get_shape(tensor).t;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_tensor_filter_pick_block_t,
		.nchildren = nparts,
		.get_child_ops = starpu_tensor_filter_pick_block_child_ops
	};

    // Create the partition
    dahl_partition* p = _partition_init(nparts, access, &dahl_traits_block, &f, tensor->handle,
                                        tensor->origin_arena, TENSOR_PARTITION_ALONG_T);
    _partition_submit(p);
    *tensor->partition = p;
}

void tensor_partition_along_t_batch(
        dahl_tensor const* tensor, dahl_access access, size_t batch_size)
{
    size_t const nparts = tensor_get_shape(tensor).t / batch_size;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_tensor_filter_t_tensor,
		.nchildren = nparts,
		.get_child_ops = starpu_tensor_filter_pick_tensor_child_ops
	};

    dahl_partition* p = _partition_init(
            nparts, access, &dahl_traits_tensor, &f, tensor->handle, 
            tensor->origin_arena, TENSOR_PARTITION_ALONG_T_BATCH);

    _partition_submit(p);
    *tensor->partition = p;
}

void tensor_unpartition(dahl_tensor_part const* tensor)
{
    dahl_partition* p = *tensor->partition;
    assert(p && p->is_active);
    _unpartition_submit(p);
}

void _tensor_print_file(void const* vtensor, FILE* fp, int8_t const precision)
{
    dahl_tensor const* tensor = (dahl_tensor const*)vtensor;
    const dahl_shape4d shape = tensor_get_shape(tensor);
	const size_t ldy = starpu_tensor_get_local_ldy(tensor->handle);
	const size_t ldz = starpu_tensor_get_local_ldz(tensor->handle);
	const size_t ldt = starpu_tensor_get_local_ldt(tensor->handle);

	tensor_acquire(tensor);

    fprintf(fp, "tensor=%p nx=%zu ny=%zu nz=%zu nt=%zu ldy=%zu ldz=%zu ldt=%zu\n{\n", tensor->data, shape.x, shape.y, shape.z, shape.t, ldy, ldz, ldt);
	for(size_t t = 0; t < shape.t; t++)
    {
        fprintf(fp, "\t{\n");
        for(size_t z = 0; z < shape.z; z++)
        {
            fprintf(fp, "\t\t{\n");
            for(size_t y = 0; y < shape.y; y++)
            {
                fprintf(fp, "\t\t\t{ ");
                for(size_t x = 0; x < shape.x; x++)
                {
                    fprintf(fp, "%+.*f, ", precision, tensor_get_value(tensor, x, y, z, t));
                }
                fprintf(fp, "},\n");
            }
            fprintf(fp, "\t\t},\n");
        }
        fprintf(fp, "\t},\n");
    }
	fprintf(fp, "}\n");

	tensor_release(tensor);
}

void tensor_print(dahl_tensor const* tensor)
{
    _tensor_print_file(tensor, stdout, DAHL_DEFAULT_PRINT_PRECISION);
}
