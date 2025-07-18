#ifndef DAHL_ARENA_H
#define DAHL_ARENA_H

#include <stddef.h>

// The `dahl_arena` enables managing memory by groups of objects.
// Alloc everything in one arena, and reset or delete everything at the same time. 
// Reset doesn't not deallocate anything, neither it sets memory to 0, it just tell
// the allocator that the memory buffer can be reused, thus previous data will be 
// overwritten.
// Delete actually free everything and deletes the arena.
typedef struct _dahl_arena dahl_arena;

// Return the pointer to a new arena
dahl_arena* dahl_arena_new();

// Allocate `size` of memory in the `arena`
void* dahl_arena_alloc(dahl_arena* arena, size_t size);

// Reset the given `arena`.
// No memory gets deallocated, the arena's buffer keeps its maximum size
// and is ready to welcome new data.
void dahl_arena_reset(dahl_arena* arena);

// Delete the given arena. Actually free the memory.
void dahl_arena_delete(dahl_arena* arena);

#endif //!DAHL_ARENA_H
