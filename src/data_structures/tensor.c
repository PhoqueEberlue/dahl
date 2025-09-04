#include "data_structures.h"
#include "../utils.h"
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
    metadata* md = dahl_arena_alloc(
        arena,
        // Metadata struct itself
        sizeof(metadata) + 
        // + a flexible array big enough partition pointers to 
        // store all kinds of tensor partioning
        (TENSOR_NB_PARTITION_TYPE * sizeof(dahl_partition*)
    ));

    for (size_t i = 0; i < TENSOR_NB_PARTITION_TYPE; i++)
        md->partitions[i] = nullptr;

    md->current_partition = -1;
    md->origin_arena = arena;

    dahl_tensor* tensor = dahl_arena_alloc(arena, sizeof(dahl_tensor));
    tensor->handle = handle;
    tensor->data = data;
    tensor->meta = md;

    return tensor;
}

dahl_tensor* tensor_init(dahl_arena* arena, dahl_shape4d const shape)
{
    size_t n_elems = shape.x * shape.y * shape.z * shape.t;
    dahl_fp* data = dahl_arena_alloc(arena, n_elems * sizeof(dahl_fp));

    for (size_t i = 0; i < n_elems; i++)
        data[i] = 0.0F;

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

    return _tensor_init_from_ptr(arena, handle, data);
}

dahl_tensor* tensor_init_from(dahl_arena* arena, dahl_shape4d const shape, dahl_fp const* data)
{
    dahl_tensor* tensor = tensor_init(arena, shape);
    size_t const n_elems = shape.x * shape.y * shape.z * shape.t;

    for (int i = 0; i < n_elems; i++)
    {
        tensor->data[i] = data[i];
    }

    return tensor;
}

dahl_tensor* tensor_init_random(dahl_arena* arena, dahl_shape4d const shape)
{
    dahl_tensor* tensor = tensor_init(arena, shape);
    size_t const n_elems = shape.x * shape.y * shape.z * shape.t;

    for (int i = 0; i < n_elems; i += 1)
    {
        tensor->data[i] = (dahl_fp)( 
            ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) 
        );
    }

    return tensor;
}

void tensor_set_from(dahl_tensor* tensor, dahl_fp const* data)
{
    dahl_shape4d shape = tensor_get_shape(tensor);
    size_t nb_elems = shape.x * shape.y * shape.z * shape.t;

    tensor_data_acquire(tensor);

    for (int i = 0; i < nb_elems; i += 1)
    {
        tensor->data[i] = data[i];
    }

    tensor_data_release(tensor);
}

dahl_matrix* tensor_flatten_along_t_no_copy(dahl_tensor const* tensor)
{
    dahl_shape4d shape = tensor_get_shape(tensor);
    size_t new_nx = shape.x * shape.y * shape.z;
    size_t new_ny = shape.t;

    starpu_data_handle_t handle = nullptr;
    starpu_matrix_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)tensor->data,
        new_nx,
        new_nx,
        new_ny,
        sizeof(dahl_fp)
    );

    dahl_arena_attach_handle(tensor->meta->origin_arena, handle);

    return _matrix_init_from_ptr(tensor->meta->origin_arena, handle, tensor->data);
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

dahl_partition* _tensor_get_current_partition(void const* tensor)
{
    metadata* m = ((dahl_tensor const*)tensor)->meta;
    assert(m-> current_partition >= 0 && 
           m->current_partition < TENSOR_NB_PARTITION_TYPE);

    assert(m->partitions[m->current_partition] != nullptr);
    return m->partitions[m->current_partition];
}

size_t _tensor_get_nb_elem(void const* tensor)
{
    dahl_shape4d shape = tensor_get_shape((dahl_tensor*)tensor);
    return shape.x * shape.y * shape.z * shape.t;
}

dahl_fp const* tensor_data_acquire(dahl_tensor const* tensor)
{
    starpu_data_acquire(tensor->handle, STARPU_R);
    return tensor->data;
}

dahl_fp* tensor_data_acquire_mut(dahl_tensor* tensor)
{
    starpu_data_acquire(tensor->handle, STARPU_RW);
    return tensor->data;
}

void tensor_data_release(dahl_tensor const* tensor)
{
    starpu_data_release(tensor->handle);
}

bool tensor_equals(dahl_tensor const* a, dahl_tensor const* b, bool const rounding, u_int8_t const precision)
{
    dahl_shape4d shape_a = tensor_get_shape(a);
    dahl_shape4d shape_b = tensor_get_shape(b);

    assert(shape_a.x == shape_b.x 
        && shape_a.y == shape_b.y 
        && shape_a.z == shape_b.z 
        && shape_a.t == shape_b.t);

    starpu_data_acquire(a->handle, STARPU_R);
    starpu_data_acquire(b->handle, STARPU_R);

    bool res = true;

    for (int i = 0; i < (shape_a.x * shape_a.y * shape_a.z * shape_a.t); i++)
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

void _tensor_partition_along_t(dahl_tensor const* tensor, bool is_mut)
{
    assert(tensor->meta->current_partition == -1);
    tensor_partition_type t = TENSOR_PARTITION_ALONG_T;

    // If the partition already exists, no need to create it.
    if (tensor->meta->partitions[t] != nullptr)
        goto submit;

    size_t const nparts = tensor_get_shape(tensor).t;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_tensor_filter_pick_block_t,
		.nchildren = nparts,
		.get_child_ops = starpu_tensor_filter_pick_block_child_ops
	};

    // Create and set the partition
    tensor->meta->partitions[t] = _partition_init(nparts, is_mut, &dahl_traits_block,
                                        &f, tensor->handle, tensor->meta->origin_arena);

submit:
    _partition_submit_if_needed(tensor->meta, t, is_mut, tensor->handle);
}

void tensor_partition_along_t_mut(dahl_tensor* tensor)
{
    _tensor_partition_along_t(tensor, true);
}

void tensor_partition_along_t(dahl_tensor const* tensor)
{
    _tensor_partition_along_t(tensor, false);
}

void _tensor_partition_along_t_batch(dahl_tensor const* tensor, size_t batch_size, bool is_mut)
{
    assert(tensor->meta->current_partition == -1);
    tensor_partition_type t = TENSOR_PARTITION_ALONG_T_BATCH;

    size_t const nparts = tensor_get_shape(tensor).t / batch_size;

    dahl_partition* p = tensor->meta->partitions[t];
    // If the partition already exists AND had the same batch size, no need to create it. 
    // FIX Warning, here the memory is lost if we create many partitions with different batch size
    if (p != nullptr && p->nb_children == nparts)
        goto submit;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_tensor_filter_t_tensor,
		.nchildren = nparts,
		.get_child_ops = starpu_tensor_filter_pick_tensor_child_ops
	};

    // Create and set the partition
    tensor->meta->partitions[t] = _partition_init(nparts, is_mut, &dahl_traits_tensor,
                                        &f, tensor->handle, tensor->meta->origin_arena);

submit:
    _partition_submit_if_needed(tensor->meta, t, is_mut, tensor->handle);
}

void tensor_partition_along_t_batch_mut(dahl_tensor* tensor, size_t batch_size)
{
    _tensor_partition_along_t_batch(tensor, batch_size, true);
}

void tensor_partition_along_t_batch(dahl_tensor const* tensor, size_t batch_size)
{
    _tensor_partition_along_t_batch(tensor, batch_size, false);
}

void tensor_unpartition(dahl_tensor const* tensor)
{
    dahl_partition* p = tensor->meta->partitions[tensor->meta->current_partition];
    assert(p); // Shouldn't crash, an non-active partition is identified by the if bellow
    assert(tensor->meta->current_partition >= 0 && 
           tensor->meta->current_partition < TENSOR_NB_PARTITION_TYPE);

    tensor->meta->current_partition = -1;
    starpu_data_unpartition_submit(tensor->handle, p->nb_children,
                                   p->handles, STARPU_MAIN_RAM);
}

void _tensor_print_file(void const* vtensor, FILE* fp)
{
    dahl_tensor const* tensor = (dahl_tensor const*)vtensor;
    const dahl_shape4d shape = tensor_get_shape(tensor);
	const size_t ldy = starpu_tensor_get_local_ldy(tensor->handle);
	const size_t ldz = starpu_tensor_get_local_ldz(tensor->handle);
	const size_t ldt = starpu_tensor_get_local_ldt(tensor->handle);

	starpu_data_acquire(tensor->handle, STARPU_R);

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
                    fprintf(fp, "%+.15f, ", tensor->data[(t*ldt)+(z*ldz)+(y*ldy)+x]);
                }
                fprintf(fp, "},\n");
            }
            fprintf(fp, "\t\t},\n");
        }
        fprintf(fp, "\t},\n");
    }
	fprintf(fp, "}\n");

	starpu_data_release(tensor->handle);
}

void tensor_print(dahl_tensor const* tensor)
{
    _tensor_print_file(tensor, stdout);
}
