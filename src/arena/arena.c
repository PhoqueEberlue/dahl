#include "_arena.h"
#include "starpu_data_interfaces.h"
#include "starpu_task.h"

#include <assert.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

dahl_arena* dahl_arena_new()
{
    dahl_arena* arena = malloc(sizeof(dahl_arena));
    arena->arena.begin = nullptr;
    arena->arena.end = nullptr;

    arena->handles = (starpu_data_handle_t*)malloc(100 * sizeof(starpu_data_handle_t));
    arena->handle_count = 0;
    arena->handle_capacity = 100;

    return arena;
}

void* dahl_arena_alloc(size_t size)
{
    assert(dahl_context_arena);
    return arena_alloc(&dahl_context_arena->arena, size);
}

void dahl_arena_attach_handle(starpu_data_handle_t handle)
{
    // TODO: realloc if max number of handles is reached
    assert(dahl_context_arena->handle_count + 1 < dahl_context_arena->handle_capacity);

    dahl_context_arena->handles[dahl_context_arena->handle_count] = handle;
    dahl_context_arena->handle_count++;
}

void dahl_arena_reset(dahl_arena* arena)
{
    arena_reset(&arena->arena);

    for (size_t i = 0; i < arena->handle_count; i++)
    {
        starpu_data_unregister(arena->handles[i]);
        arena->handles[i] = nullptr;
    }

    arena->handle_count = 0;
}

// void arena_delete(dahl_arena* arena)
// {
//     free(arena->buffer);
//     free(arena);
// }

