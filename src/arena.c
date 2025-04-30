#include "../include/dahl_arena.h"

#include <assert.h>
#include <malloc.h>
#include <string.h>

dahl_arena* arena_new(size_t const size)
{
    dahl_arena* res = malloc(sizeof(dahl_arena));

    res->buffer = malloc(size);
    res->buffer_size = size;
    res->offset = 0;

    return res;
}

void* arena_put(dahl_arena* arena, size_t size)
{
    assert(arena->offset + size < arena->buffer_size);

    uintptr_t* res = &arena->buffer[arena->offset];
    arena->offset += size;

    return res;
}

void arena_reset(dahl_arena* arena)
{
    arena->offset = 0;
    memset(arena->buffer, 0, arena->buffer_size);
}

void arena_delete(dahl_arena* arena)
{
    free(arena->buffer);
    free(arena);
}
