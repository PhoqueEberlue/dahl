#include "../include/dahl.h"
#include <stdio.h>
#include <starpu.h>
#include "tasks/codelets.h"

void dahl_init()
{
    int ret = starpu_init(nullptr);
    if (ret != 0)
    {
        printf("Could not initialize starpu");
    }

    // starpu_cuda_set_device(0);

    // Force the codelet switch to be executed on the main RAM.
    // Required to refresh and synchronize buffers.
    cl_switch.specific_nodes = 1;
	for (int i = 0; i < STARPU_NMAXBUFS; i++)
	    cl_switch.nodes[i] = STARPU_MAIN_RAM;
}

void dahl_shutdown()
{
	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();
}
