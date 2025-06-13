#include "arena.h"

#include <assert.h>
#include <malloc.h>
#include <string.h>

dahl_arena* arena_new(size_t const size)
{
    dahl_arena* res = malloc(sizeof(dahl_arena));

    res->buffer = malloc(size);
    res->buffer_size = size;
    res->buffer_offset = 0;

    // Here its the maximum number of dahl_any, i.e. number of dahl data objects
    res->anys = malloc(10000 * sizeof(dahl_any));
    res->anys_capacity = 10000;
    res->anys_offset = 0;

    return res;
}

void* arena_put(dahl_arena* arena, size_t size)
{
    assert(arena->buffer_offset + size < arena->buffer_size);

    uintptr_t* res = &arena->buffer[arena->buffer_offset];
    arena->buffer_offset += size;

    return res;
}

void arena_attach_data(dahl_arena* arena, dahl_any any)
{
    assert(arena->anys_number < arena->anys_capacity);

    arena->anys[arena->anys_number] = any;
    arena->anys_number++;
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
