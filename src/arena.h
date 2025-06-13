#ifndef DAHL_ARENA_H
#define DAHL_ARENA_H

#include <stddef.h>
#include <stdint.h>
#include "data_structures/data_structures.h"

typedef struct _dahl_arena
{
    uintptr_t* buffer;
    size_t buffer_size;
    size_t buffer_offset;

    dahl_any* anys;
    size_t anys_capacity;
    size_t anys_number;
} dahl_arena;

#endif //!DAHL_ARENA_H
