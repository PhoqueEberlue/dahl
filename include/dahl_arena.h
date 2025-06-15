#ifndef DAHL_ARENA_H
#define DAHL_ARENA_H

#include <stddef.h>

typedef struct _dahl_arena dahl_arena;

extern dahl_arena* default_arena;
extern dahl_arena* temporary_arena;
extern dahl_arena* context_arena;
extern dahl_arena* context_arena_save;

dahl_arena* dahl_arena_new();
void* dahl_arena_alloc(size_t size);
void dahl_arena_reset(dahl_arena* arena);

#endif //!DAHL_ARENA_H
