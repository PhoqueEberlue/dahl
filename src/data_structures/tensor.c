#include "data_structures.h"
#include "../utils.h"
#include "starpu_data.h"
#include "starpu_data_filters.h"
#include "starpu_data_interfaces.h"
#include "starpu_util.h"
#include "sys/types.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

dahl_tensor* tensor_init(dahl_shape4d const shape)
{
    size_t n_elems = shape.x * shape.y * shape.z * shape.t;
    dahl_fp* data = dahl_arena_alloc(n_elems * sizeof(dahl_fp));

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

    dahl_arena_attach_handle(handle);

    dahl_tensor* tensor = dahl_arena_alloc(sizeof(dahl_tensor));
    tensor->handle = handle;
    tensor->data = data;
    tensor->partition_type = DAHL_NONE;
    tensor->sub_handles = nullptr;
    tensor->nb_sub_handles = 0;

    return tensor;
}

dahl_tensor* tensor_init_from(dahl_shape4d const shape, dahl_fp const* data)
{
    dahl_tensor* tensor = tensor_init(shape);
    size_t const n_elems = shape.x * shape.y * shape.z * shape.t;

    for (int i = 0; i < n_elems; i++)
    {
        tensor->data[i] = data[i];
    }

    return tensor;
}

dahl_tensor* tensor_init_random(dahl_shape4d const shape)
{
    dahl_tensor* tensor = tensor_init(shape);
    size_t const n_elems = shape.x * shape.y * shape.z * shape.t;

    for (int i = 0; i < n_elems; i += 1)
    {
        tensor->data[i] = (dahl_fp)( 
            ( rand() % 2 ? 1 : -1 ) * ( (dahl_fp)rand() / (dahl_fp)(RAND_MAX / DAHL_MAX_RANDOM_VALUES)) 
        );
    }

    return tensor;
}

dahl_tensor* tensor_clone(dahl_tensor const* tensor)
{
    dahl_shape4d shape = tensor_get_shape(tensor);

    starpu_data_acquire(tensor->handle, STARPU_R);
    dahl_tensor* res = tensor_init_from(shape, tensor->data);
    starpu_data_release(tensor->handle);

    return res;
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

dahl_fp const* tensor_data_acquire(dahl_tensor const* tensor)
{
    starpu_data_acquire(tensor->handle, STARPU_R);
    return tensor->data;
}

dahl_fp* tensor_data_acquire_mutable(dahl_tensor* tensor)
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

void tensor_partition_along_t(dahl_tensor* tensor)
{
    // The tensor shouldn't be already partitionned
    assert(tensor->partition_type == DAHL_NONE);

    size_t const nparts = tensor_get_shape(tensor).t;

    tensor->partition_type = DAHL_BLOCK;
    tensor->sub_data.blocks = dahl_arena_alloc(nparts * sizeof(dahl_block));
    tensor->sub_handles = (starpu_data_handle_t*)dahl_arena_alloc(nparts * sizeof(starpu_data_handle_t));
    tensor->nb_sub_handles = nparts;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_tensor_filter_pick_block_t,
		.nchildren = nparts,
		.get_child_ops = starpu_tensor_filter_pick_block_child_ops
	};

	starpu_data_partition_plan(tensor->handle, &f, tensor->sub_handles);
    
    for (int i = 0; i < nparts; i++)
    {
        tensor->sub_data.blocks[i].handle = tensor->sub_handles[i];
        tensor->sub_data.blocks[i].data = (dahl_fp*)starpu_block_get_local_ptr(tensor->sub_handles[i]);
        // Children are not yet partitioned
        tensor->sub_data.blocks[i].partition_type = DAHL_NONE;
    }

    starpu_data_partition_submit(tensor->handle, nparts, tensor->sub_handles);
}

void tensor_unpartition(dahl_tensor* tensor)
{
    switch (tensor->partition_type)
    {
        case DAHL_NONE:
            printf("ERROR: Tried calling %s but the object is not partitioned", __func__);
            abort();
            break;
        case DAHL_TENSOR:
            printf("ERROR: got value %i in function %s, however tensor should only be partioned into block, matrix or vector", 
                   tensor->partition_type, __func__);
            abort();
            break;
        case DAHL_BLOCK:
            tensor->sub_data.blocks = nullptr;
            break;
        case DAHL_MATRIX:
            tensor->sub_data.matrices = nullptr;
            break;
        case DAHL_VECTOR:
            tensor->sub_data.vectors = nullptr;
            break;
    }

    tensor->partition_type = DAHL_NONE;
    starpu_data_unpartition_submit(tensor->handle, tensor->nb_sub_handles,
                            tensor->sub_handles, STARPU_MAIN_RAM);

    starpu_data_partition_clean(tensor->handle, tensor->nb_sub_handles, tensor->sub_handles);
}

size_t tensor_get_nb_children(dahl_tensor const* tensor)
{
    assert(tensor->partition_type != DAHL_NONE);
    return tensor->nb_sub_handles;
}

dahl_block* tensor_get_sub_block(dahl_tensor const* tensor, const size_t index)
{
    assert(tensor->partition_type == DAHL_BLOCK 
        && tensor->sub_data.blocks != nullptr 
        && index < tensor->nb_sub_handles);

    return &tensor->sub_data.blocks[index];
}

dahl_matrix* tensor_get_sub_matrix(dahl_tensor const* tensor, const size_t index)
{
    assert(tensor->partition_type == DAHL_MATRIX 
        && tensor->sub_data.matrices != nullptr 
        && index < tensor->nb_sub_handles);

    return &tensor->sub_data.matrices[index];
}

dahl_vector* tensor_get_sub_vector(dahl_tensor const* tensor, const size_t index)
{
    assert(tensor->partition_type == DAHL_VECTOR 
        && tensor->sub_data.vectors != nullptr 
        && index < tensor->nb_sub_handles);

    return &tensor->sub_data.vectors[index];
}

void tensor_print(dahl_tensor const* tensor)
{
    const dahl_shape4d shape = tensor_get_shape(tensor);
	const size_t ldy = starpu_tensor_get_local_ldy(tensor->handle);
	const size_t ldz = starpu_tensor_get_local_ldz(tensor->handle);
	const size_t ldt = starpu_tensor_get_local_ldt(tensor->handle);

	starpu_data_acquire(tensor->handle, STARPU_R);

    printf("tensor=%p nx=%zu ny=%zu nz=%zu ldy=%zu ldz=%zu ldt=%zu\n", tensor->data, shape.x, shape.y, shape.z, ldy, ldz, ldt);

	for(size_t t = 0; t < shape.t; t++)
    {
        for(size_t z = 0; z < shape.z; z++)
        {
            for(size_t y = 0; y < shape.y; y++)
            {
                for(size_t x = 0; x < shape.x; x++)
                {
                    printf("%f ", tensor->data[(t*ldt)+(z*ldz)+(y*ldy)+x]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
	printf("\n");

	starpu_data_release(tensor->handle);
}
