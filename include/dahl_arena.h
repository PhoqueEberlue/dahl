#ifndef DAHL_ARENA_H
#define DAHL_ARENA_H

#include <stddef.h>
#include <stdint.h>
#include "dahl_types.h"

typedef struct _dahl_arena
{
    uintptr_t* buffer;
    size_t buffer_size;
    size_t buffer_offset;

    void *anys;
    size_t anys_size;
    size_t anys_offset;
} dahl_arena;

dahl_arena* arena_new(size_t const size);
void* arena_put(dahl_arena* arena, size_t size);
void arena_reset(dahl_arena* arena);
void arena_delete(dahl_arena* arena);

#endif //!DAHL_ARENA_H
