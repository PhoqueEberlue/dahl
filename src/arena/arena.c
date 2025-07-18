#include "arena.h"

#include <assert.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Including the arena backend implementation here to avoid leaking its functions outside this file.
#include "arena_tsoding.h"
#include "starpu_data.h"

// TODO: make it dynamic?
#define NMAX_HANDLES 1000
#define NMAX_PARTITIONS 50000

// Redifinition of dahl_partition with only the children handles and the parent data handle
typedef struct
{
    starpu_data_handle_t main_handle;
    size_t nb_children;
    starpu_data_handle_t* children;
} _partition;

// Defining dahl_arena here so it stays opaque in the public API
typedef struct _dahl_arena
{
    Arena arena;
    starpu_data_handle_t* handles;
    size_t handle_count;

    size_t partition_count;
    _partition partitions[];
} dahl_arena;

dahl_arena* dahl_arena_new()
{
    dahl_arena* arena = malloc(
        sizeof(dahl_arena) + 
        // Allocate by default the maximum number of partitions
        (NMAX_PARTITIONS * sizeof(_partition))
    );

    arena->arena.begin = nullptr;
    arena->arena.end = nullptr;

    arena->handles = (starpu_data_handle_t*)malloc(NMAX_HANDLES * sizeof(starpu_data_handle_t));
    arena->handle_count = 0;

    arena->partition_count = 0;

    return arena;
}

void* dahl_arena_alloc(dahl_arena* arena, size_t size)
{
    assert(arena);
    return arena_alloc(&arena->arena, size);
}

void dahl_arena_attach_handle(dahl_arena* arena, starpu_data_handle_t handle)
{
    // TODO: realloc if max number of handles is reached
    assert(arena->handle_count + 1 < NMAX_HANDLES);

    arena->handles[arena->handle_count] = handle;
    arena->handle_count++;
}

void dahl_arena_attach_partition(dahl_arena* arena, starpu_data_handle_t main_handle, 
                                 size_t nb_children_handle, starpu_data_handle_t* children_handles)
{
    // TODO: realloc if max number of handles is reached
    assert(arena->partition_count + 1 < NMAX_PARTITIONS);

    _partition *p = &arena->partitions[arena->partition_count];

    p->main_handle = main_handle;
    p->nb_children = nb_children_handle;
    p->children = children_handles;

    arena->partition_count++;
}

void dahl_arena_reset(dahl_arena* arena)
{
    // Make sure to clean the partitions before, otherwise it would attempt to unregister
    // parent handles that have children.
    // Also, loop through the partitions in reverse to clean nested partitions in the right order.
    // Use i+1 in the condition because size_t is unsigned and would otherwise overflow
    for (size_t i = arena->partition_count - 1; i + 1 > 0; i--)
    {
        _partition* p = &arena->partitions[i];

        starpu_data_partition_clean(p->main_handle, p->nb_children, p->children);
        p->main_handle = nullptr;
        p->nb_children = 0;
        p->children = nullptr;
    }

    arena->partition_count = 0;

    for (size_t i = arena->handle_count - 1; i + 1 > 0 ; i--)
    {
        // Using no coherency to prevent data to be copied in the main memory
        starpu_data_unregister_no_coherency(arena->handles[i]);
        arena->handles[i] = nullptr;
    }

    arena->handle_count = 0;

    arena_reset(&arena->arena);
}

void dahl_arena_delete(dahl_arena* arena)
{
    // Important to unregister handles and partitions
    dahl_arena_reset(arena);

    arena_free(&arena->arena);
    free((void*)arena->handles);
    free(arena);
}
