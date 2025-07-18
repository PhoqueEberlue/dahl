#ifndef _DAHL_ARENA_H
#define _DAHL_ARENA_H

#include "../../include/dahl_arena.h"
#include <starpu.h>

// Private functions

// Attach a starpu handle to the current context arena.
// The handle will be unregistered upon reseting the arena.
void dahl_arena_attach_handle(dahl_arena* arena, starpu_data_handle_t handle);


// Attach a starpu partition that was allocated asynchronously with `starpu_data_partition_plan` to the current context arena.
// The whole partition handles will be cleaned up upon reseting the arena.
void dahl_arena_attach_partition(dahl_arena* arena, starpu_data_handle_t main_handle, 
                                 size_t nb_children_handle, starpu_data_handle_t* children_handles);

#endif //!_DAHL_ARENA_H
