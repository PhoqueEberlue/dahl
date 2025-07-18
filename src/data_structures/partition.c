#include "data_structures.h"
#include "starpu_data_interfaces.h"
#include <stdio.h>

dahl_partition* _partition_init(size_t nb_children, bool is_mut, dahl_traits* trait, 
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
    p->type = trait->type;
    p->is_mut = is_mut;

    // Publish the partition plan, required to run before getting the local ptrs
	starpu_data_partition_plan(main_handle, f, p->handles);

    // Allocate the children and set their handle ptrs. Those will be instanciated by
    // StarPU when submitting the partition plan.
    for (int i = 0; i < nb_children; i++)
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

void _partition_submit_if_needed(metadata* meta, int8_t index, bool should_be_mut, starpu_data_handle_t main_handle)
{
    dahl_partition* p = meta->partitions[index];

    bool need_resubmit = false;

    if (meta->current_partition != index)
    { 
        meta->current_partition = index; 
        need_resubmit = true; 
    }

    if (should_be_mut != p->is_mut)
    {
        p->is_mut = (bool)!p->is_mut; // toggle mutable status
        need_resubmit = true;
    }

    if (need_resubmit)
    {
        if (should_be_mut)
            starpu_data_partition_submit(main_handle, p->nb_children, p->handles);
        else
            starpu_data_partition_readonly_submit(main_handle, p->nb_children, p->handles);
    }
}

size_t get_nb_children(void const* object, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    return p->nb_children;
}

dahl_block* _get_sub_block(dahl_partition const* p, size_t index)
{
    assert(p->type == DAHL_BLOCK
        && index < p->nb_children
        && p->children[index] != nullptr);

    return p->children[index];
}

dahl_block* get_sub_block_mut(void* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(p->is_mut);
    return _get_sub_block(p, index);
}

dahl_block const* get_sub_block(void const* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(!p->is_mut);
    return _get_sub_block(p, index);
}

dahl_matrix* _get_sub_matrix(dahl_partition const* p, size_t index)
{
    assert(p->type == DAHL_MATRIX
        && index < p->nb_children
        && p->children[index] != nullptr);

    return p->children[index];
}

dahl_matrix* get_sub_matrix_mut(void* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(p->is_mut);
    return _get_sub_matrix(p, index);
}

dahl_matrix const* get_sub_matrix(void const* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(!p->is_mut);
    return _get_sub_matrix(p, index);
}

dahl_vector* _get_sub_vector(dahl_partition const* p, size_t index)
{
    assert(p->type == DAHL_VECTOR
        && index < p->nb_children
        && p->children[index] != nullptr);

    return p->children[index];
}

dahl_vector* get_sub_vector_mut(void* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(p->is_mut);
    return _get_sub_vector(p, index);
}

dahl_vector const* get_sub_vector(void const* object, size_t index, dahl_traits* traits)
{
    dahl_partition* p = traits->get_partition(object);
    assert(!p->is_mut);
    return _get_sub_vector(p, index);
}
