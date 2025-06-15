#ifndef _DAHL_ARENA_H
#define _DAHL_ARENA_H

#include "../../include/dahl_arena.h"
#include <stddef.h>
#include <starpu.h>
#include <stdint.h>

// typedef struct
// {
//     starpu_data_handle_t *handles;
//     size_t handle_count;
//     size_t handle_capacity;
// } handle_list;

// Internal function not available in the public API
void dahl_arena_attach_handle(starpu_data_handle_t handle);
// void arena_delete(dahl_arena* arena);

// TODO: change tsoding's arena implem to use starpu malloc
//
// I still struggle to understand how I should use arenas, its not perfectly clear to me if I should expose them in my api?
// Maybe I could just sneak them into the layers implementations, e.g. conv have one layer for constant buffers (weigths + biases + output so it does not disapear) and one for temporary computation buffers that can be reset at each call......
// But if we do that, what's the point of using arena alloc (which does not perform alloc/free but only register/unregistering) because I could just be reusing the same handles each iteration? Furthermore it means that the memory will always be bounded to the sum of every layers.
// One way to tackle the last issue is to use a context arena that is instead used in every layers and reseted at the end of each function.
// That's a good idea actually, this way it should still be hidden and not exposed in the API.
//
// This works if I plan to continue using the layers, if I want more abstract task execution, or want to implement the "partition reading mechanism" it seems like it should behave differently.
// But for now it seems alright.

#endif //!_DAHL_ARENA_H
