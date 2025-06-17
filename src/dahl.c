#include "../include/dahl.h"
#include <stdio.h>
#include <starpu.h>

dahl_arena* dahl_persistent_arena = nullptr;
dahl_arena* dahl_temporary_arena = nullptr;
dahl_arena* dahl_context_arena = nullptr;

void dahl_init()
{
    int ret = starpu_init(nullptr);
    if (ret != 0)
    {
        printf("Could not initialize starpu");
    }

    dahl_persistent_arena = dahl_arena_new();
    dahl_temporary_arena = dahl_arena_new();
    dahl_context_arena = dahl_persistent_arena;
}

void dahl_shutdown()
{
    dahl_arena_delete(dahl_persistent_arena);
    dahl_arena_delete(dahl_temporary_arena);
	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();
}
