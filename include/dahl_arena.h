#ifndef DAHL_ARENA_H
#define DAHL_ARENA_H

#include <stddef.h>

// The `dahl_arena` enables managing memory by groups of objects.
// Alloc everything in one arena, and reset or delete everything at the same time. 
// Reset doesn't not deallocate anything, it just tell the allocator that the memory buffer 
// can be reused, previous data will be overwritten.
// Delete actually free everything and deletes the arena.
typedef struct _dahl_arena dahl_arena;

// Persistent arena that won't (and shouldn't) get reset or free before the end of the program.
// Created automatically by dahl
extern dahl_arena* dahl_persistent_arena;
// Temporary arena, it can be reseted at any time.
// Created automatically by dahl
extern dahl_arena* dahl_temporary_arena;

// Return the pointer to a new arena
dahl_arena* dahl_arena_new();

void dahl_arena_set_context(dahl_arena* arena);
dahl_arena* dahl_arena_get_context();
void dahl_arena_restore_context();

// Allocate memory in the current context arena
void* dahl_arena_alloc(size_t size);

// Reset the given arena.
// No memory gets deallocated, the arena's buffer keeps its maximum size
// and is ready to welcome new data.
void dahl_arena_reset(dahl_arena* arena);

// Delete the given arena. Actually free the memory.
void dahl_arena_delete(dahl_arena* arena);

#endif //!DAHL_ARENA_H
