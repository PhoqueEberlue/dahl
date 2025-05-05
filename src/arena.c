#include "../include/dahl_arena.h"

#include <assert.h>
#include <malloc.h>
#include <string.h>

dahl_arena* arena_new(size_t const size)
{
    dahl_arena* res = malloc(sizeof(dahl_arena));

    res->buffer = malloc(size);
    res->buffer_size = size;
    res->buffer_offset = 0;

    res->handles = (starpu_data_handle_t*)malloc(10000);
    res->handles_size = 10000;
    res->handles_offset = 0;

    return res;
}

void* arena_put(dahl_arena* arena, size_t size)
{
    assert(arena->buffer_offset + size < arena->buffer_size);

    uintptr_t* res = &arena->buffer[arena->buffer_offset];
    arena->buffer_offset += size;

    return res;
}

void arena_add_handle(dahl_arena* arena, starpu_data_handle_t handle)
{
    assert(arena->handles_offset + sizeof(starpu_data_handle_t) < arena->handles_size);

    arena->handles[arena->handles_offset] = handle;
    arena->buffer_offset += sizeof(starpu_data_handle_t);
}

void arena_reset(dahl_arena* arena)
{
    arena->buffer_offset = 0;
    memset(arena->buffer, 0, arena->buffer_size);

    for (size_t i = 0; i < arena->handles_size; i++)
    {
        // TODO: to allow pinning/unpinning we will have to change the data structure to save the start
        // of each memory block.
        // TODO: do I have to pin each data structure, or can I pin the whole arena buffer?
        // starpu_memory_unpin();
        starpu_data_release(arena->handles[i]);
    }
}

void arena_delete(dahl_arena* arena)
{
    free(arena->buffer);
    free(arena);
}
