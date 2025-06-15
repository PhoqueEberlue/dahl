#ifndef __DAHL_ARENA_H
#define __DAHL_ARENA_H

#include "arena.h"

#include "arena_tsoding.h"

typedef struct _dahl_arena
{
    Arena arena;
    starpu_data_handle_t* handles;
    size_t handle_count;
    size_t handle_capacity;
} dahl_arena;

#endif //!__DAHL_ARENA_H
