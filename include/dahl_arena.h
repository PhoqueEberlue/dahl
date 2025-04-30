#include <stdint.h>
#include <stddef.h>

typedef struct
{
    uintptr_t* buffer;
    size_t buffer_size;

    size_t offset;
} dahl_arena;

dahl_arena* arena_new(size_t const size);
void* arena_put(dahl_arena* arena, size_t size);
void arena_reset(dahl_arena* arena);
void arena_delete(dahl_arena* arena);
