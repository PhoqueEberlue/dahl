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
    size_t n_elems = shape.x * shape.y * shape.z;
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
    tensor->sub_blocks = nullptr;
    tensor->sub_matrices = nullptr;
    tensor->sub_vectors = nullptr;
    tensor->is_partitioned = false;

    return tensor;
}

dahl_tensor* tensor_init_from(dahl_shape4d const shape, dahl_fp const* data)
{
    dahl_tensor* tensor = tensor_init(shape);
    size_t const n_elems = shape.x * shape.y * shape.z;

    for (int i = 0; i < n_elems; i++)
    {
        tensor->data[i] = data[i];
    }

    return tensor;
}

dahl_tensor* tensor_init_random(dahl_shape4d const shape)
{
    dahl_tensor* tensor = tensor_init(shape);
    size_t const n_elems = shape.x * shape.y * shape.z;

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

    dahl_fp* data = tensor_data_acquire(tensor);
    dahl_tensor* res = tensor_init_from(shape, data);
    tensor_data_release(tensor);

    return res;
}

dahl_tensor* tensor_add_padding_init(dahl_tensor const* tensor, dahl_shape4d const new_shape)
{
    dahl_shape4d shape = tensor_get_shape(tensor);

    starpu_data_acquire(tensor->handle, STARPU_R);
    dahl_fp* data = tensor->data;

    assert(new_shape.x >= shape.x && new_shape.y >= shape.y && new_shape.z >= shape.z);

    size_t diff_z = (new_shape.z - shape.z) / 2;
    size_t diff_y = (new_shape.y - shape.y) / 2;
    size_t diff_x = (new_shape.x - shape.x) / 2;

    dahl_tensor* res = tensor_init(new_shape);
    starpu_data_acquire(res->handle, STARPU_W);
    dahl_fp* res_data = res->data;

    for (size_t z = 0; z < shape.z; z++)
    {
        for (size_t y = 0; y < shape.y; y++)
        {
            for (size_t x = 0; x < shape.x; x++)
            {
                dahl_fp value = data[(z * shape.x * shape.y) + (y * shape.x) + x];
                // FIX PLEASE JUST DO AN ACCESSOR FUNCTION WITH X, Y, Z AS PARAMETERS SO WE CAN IGNORE LD
                res_data[((z + diff_z) * new_shape.x * new_shape.y) + ((y + diff_y) * new_shape.x) + (x + diff_x)] = value;
            }
        }

    }

    starpu_data_release(tensor->handle);
    starpu_data_release(res->handle);

    return res;
}

dahl_shape4d tensor_get_shape(dahl_tensor const* tensor)
{
    starpu_data_acquire(tensor->handle, STARPU_R);
    size_t nx = starpu_tensor_get_nx(tensor->handle);
	size_t ny = starpu_tensor_get_ny(tensor->handle);
	size_t nz = starpu_tensor_get_nz(tensor->handle);
	size_t nt = starpu_tensor_get_nt(tensor->handle);
    dahl_shape4d res = { .x = nx, .y = ny, .z = nz, .t = nt };
    starpu_data_release(tensor->handle);

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

dahl_fp* tensor_data_acquire(dahl_tensor const* tensor)
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
    assert(tensor->is_partitioned != true);

    dahl_shape4d const shape = tensor_get_shape(tensor);

    struct starpu_data_filter f =
	{
		.filter_func = starpu_tensor_filter_pick_block_t,
		.nchildren = shape.t,
		.get_child_ops = starpu_tensor_filter_pick_block_child_ops
	};

	starpu_data_partition(tensor->handle, &f);
    
    tensor->is_partitioned = true;
    tensor->sub_blocks = dahl_arena_alloc(shape.t * sizeof(dahl_block));

    for (int i = 0; i < starpu_data_get_nb_children(tensor->handle); i++)
    {
		starpu_data_handle_t sub_block_handle = starpu_data_get_sub_data(tensor->handle, 1, i);

        dahl_fp* data = (dahl_fp*)starpu_block_get_local_ptr(sub_block_handle);
        assert(data);

        tensor->sub_blocks[i].handle = sub_block_handle;
        tensor->sub_blocks[i].data = data;
        tensor->sub_blocks[i].is_sub_data = true;
        tensor->sub_blocks[i].is_partitioned = false;
    }
}

void tensor_unpartition(dahl_tensor* tensor)
{
    assert(tensor->is_partitioned);

    starpu_data_unpartition(tensor->handle, STARPU_MAIN_RAM);

    // Only one of sub_blocks, sub_matrices or sub_vectors should be defined
    if (tensor->sub_blocks != nullptr)
    {
        // Will be freed by the arena
        tensor->sub_blocks = nullptr;
    }
    else if (tensor->sub_matrices != nullptr)
    {
        // Will be freed by the arena
        tensor->sub_matrices = nullptr;
    }
    else if (tensor->sub_vectors != nullptr)
    {
        // Will be freed by the arena
        tensor->sub_vectors = nullptr;
    }
    else 
    {
        printf("ERROR: Neither sub_matrices or sub_vectors were defined, this is weird");
        abort();
    }

    tensor->is_partitioned = false;
}

size_t tensor_get_nb_children(dahl_tensor const* tensor)
{
    return starpu_data_get_nb_children(tensor->handle);
}

dahl_block* tensor_get_sub_block(dahl_tensor const* tensor, const size_t index)
{
    assert(tensor->is_partitioned 
        && tensor->sub_blocks != nullptr 
        && index < starpu_data_get_nb_children(tensor->handle));

    return &tensor->sub_blocks[index];
}

dahl_matrix* tensor_get_sub_matrix(dahl_tensor const* tensor, const size_t index)
{
    assert(tensor->is_partitioned 
        && tensor->sub_matrices != nullptr 
        && index < starpu_data_get_nb_children(tensor->handle));

    return &tensor->sub_matrices[index];
}

dahl_vector* tensor_get_sub_vector(dahl_tensor const* tensor, const size_t index)
{
    assert(tensor->is_partitioned 
        && tensor->sub_vectors != nullptr 
        && index < starpu_data_get_nb_children(tensor->handle));

    return &tensor->sub_vectors[index];
}

void tensor_print(dahl_tensor const* tensor)
{
    const dahl_shape4d shape = tensor_get_shape(tensor);
	const size_t ldy = starpu_tensor_get_local_ldy(tensor->handle);
	const size_t ldz = starpu_tensor_get_local_ldz(tensor->handle);

	starpu_data_acquire(tensor->handle, STARPU_R);

    printf("tensor=%p nx=%zu ny=%zu nz=%zu ldy=%zu ldz=%zu\n", tensor->data, shape.x, shape.y, shape.z, ldy, ldz);

	for(size_t t = 0; t < shape.t; t++)
    {
        for(size_t z = 0; z < shape.z; z++)
        {
            for(size_t y = 0; y < shape.y; y++)
            {
                for(size_t x = 0; x < shape.x; x++)
                {
                    printf("%f ", tensor->data[(z*ldz)+(y*ldy)+x]);
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
