#ifndef _DAHL_ARENA_H
#define _DAHL_ARENA_H

#include "../../include/dahl_arena.h"
#include <starpu.h>

// Private functions
// Attach a starpu handle to the current context arena.
// The handle will be unregistered upon reseting the arena.
void dahl_arena_attach_handle(starpu_data_handle_t handle);

#endif //!_DAHL_ARENA_H
