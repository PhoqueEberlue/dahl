#include "arena.h"

#include <assert.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Including the arena backend implementation here to avoid leaking its functions outside this file.
#include "arena_tsoding.h"

// Defining dahl_arena here so it stays opaque in the public API
typedef struct _dahl_arena
{
    Arena arena;
    starpu_data_handle_t* handles;
    size_t handle_count;
    size_t handle_capacity;
} dahl_arena;

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

void dahl_arena_delete(dahl_arena* arena)
{
    // Important to unregister handles
    dahl_arena_reset(arena);

    arena_free(&arena->arena);
    free((void*)arena->handles);
    free(arena);
}
