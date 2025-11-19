#include "data_structures.h"
#include "starpu_data.h"
#include "starpu_data_interfaces.h"
#include <stdio.h>

dahl_partition* _partition_init(size_t nb_children, dahl_access access, dahl_traits* trait, 
                                struct starpu_data_filter* f, starpu_data_handle_t main_handle,
                                dahl_arena* origin_arena)
{
    dahl_partition* p = dahl_arena_alloc(
        origin_arena,
        // The partition object itself
        sizeof(dahl_partition) + 
        // + the children array with enough space to store their pointers
        (nb_children * sizeof(void*))
    );

    p->handles = (starpu_data_handle_t*)dahl_arena_alloc(
        origin_arena,
        nb_children * sizeof(starpu_data_handle_t));
    p->nb_children = nb_children;
    p->trait = trait;
    p->access = access;

    // Publish the partition plan, required to run before getting the local ptrs
	starpu_data_partition_plan(main_handle, f, p->handles);

    // Allocate the children and set their handle ptrs. Those will be instanciated by
    // StarPU when submitting the partition plan.
    for (size_t i = 0; i < nb_children; i++)
    {
        p->children[i] = trait->init_from_ptr(
            origin_arena,
            p->handles[i], 
            (dahl_fp*)starpu_data_get_local_ptr(p->handles[i])
        );
    }

    dahl_arena_attach_partition(origin_arena, main_handle, nb_children, p->handles);

    return p;
}

void _partition_enable_redux(dahl_partition* p)
{
    for (size_t i = 0; i < p->nb_children; i++)
    {
        p->trait->enable_redux(p->children[i]);
    }
}

void _partition_submit_if_needed(
        metadata* meta, int8_t index, dahl_access new_access, starpu_data_handle_t main_handle)
{
    dahl_partition* p = meta->partitions[index];

    // If requested partition is not active, or it changed access type
    if (meta->current_partition != index || new_access != p->access)
    { 
        meta->current_partition = index; 
        p->access = new_access;
    
        switch (p->access) {
            case DAHL_READ:
                starpu_data_partition_readonly_submit(main_handle, p->nb_children, p->handles);
                break;
            case DAHL_MUT:
                starpu_data_partition_submit(main_handle, p->nb_children, p->handles);
                break;
            case DAHL_REDUX:
                _partition_enable_redux(p);
                starpu_data_partition_submit(main_handle, p->nb_children, p->handles);
                break;
        }
    }
}

size_t get_nb_children(void const* object, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    return p->nb_children;
}

dahl_tensor* _get_sub_tensor(dahl_partition const* p, size_t index)
{
    assert(p->trait->type == DAHL_TENSOR
        && index < p->nb_children
        && p->children[index] != nullptr);

    return p->children[index];
}

dahl_tensor* get_sub_tensor_mut(void* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(p->access == DAHL_MUT || p->access == DAHL_REDUX);
    return _get_sub_tensor(p, index);
}

dahl_tensor const* get_sub_tensor(void const* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(p->access == DAHL_READ || p->access == DAHL_MUT);
    return _get_sub_tensor(p, index);
}

dahl_block* _get_sub_block(dahl_partition const* p, size_t index)
{
    assert(p->trait->type == DAHL_BLOCK
        && index < p->nb_children
        && p->children[index] != nullptr);

    return p->children[index];
}

dahl_block* get_sub_block_mut(void* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(p->access == DAHL_MUT || p->access == DAHL_REDUX);
    return _get_sub_block(p, index);
}

dahl_block const* get_sub_block(void const* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(p->access == DAHL_READ || p->access == DAHL_MUT);
    return _get_sub_block(p, index);
}

dahl_matrix* _get_sub_matrix(dahl_partition const* p, size_t index)
{
    assert(p->trait->type == DAHL_MATRIX
        && index < p->nb_children
        && p->children[index] != nullptr);
    return p->children[index];
}

dahl_matrix* get_sub_matrix_mut(void* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(p->access == DAHL_MUT || p->access == DAHL_REDUX);
    return _get_sub_matrix(p, index);
}

dahl_matrix const* get_sub_matrix(void const* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(p->access == DAHL_READ || p->access == DAHL_MUT);
    return _get_sub_matrix(p, index);
}

dahl_vector* _get_sub_vector(dahl_partition const* p, size_t index)
{
    assert(p->trait->type == DAHL_VECTOR
        && index < p->nb_children
        && p->children[index] != nullptr);

    return p->children[index];
}

dahl_vector* get_sub_vector_mut(void* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(p->access == DAHL_MUT || p->access == DAHL_REDUX);
    return _get_sub_vector(p, index);
}

dahl_vector const* get_sub_vector(void const* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(p->access == DAHL_READ || p->access == DAHL_MUT);
    return _get_sub_vector(p, index);
}
