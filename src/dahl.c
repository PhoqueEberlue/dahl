#include "../include/dahl.h"
#include <stdio.h>
#include <starpu.h>

dahl_arena* default_arena = nullptr;
dahl_arena* temporary_arena = nullptr;
dahl_arena* context_arena = nullptr;
dahl_arena* context_arena_save = nullptr;

void dahl_init()
{
    int ret = starpu_init(nullptr);
    if (ret != 0)
    {
        printf("Could not initialize starpu");
    }

    default_arena = dahl_arena_new();
    temporary_arena = dahl_arena_new();
    context_arena = default_arena;
}

void dahl_shutdown()
{
	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();
}
