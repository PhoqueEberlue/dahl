#include "../include/dahl.h"
#include <stdio.h>
#include <starpu.h>

void dahl_init()
{
    int ret = starpu_init(nullptr);
    if (ret != 0)
    {
        printf("Could not initialize starpu");
    }
}

void dahl_shutdown()
{
	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();
}
