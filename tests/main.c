#include "tests.h"
#include <stdio.h>

#define RANDOM_SEED 42
#define NUMBER_TESTS 1

// It it absolutly mandatory to have the following functions arguments to run starpu with simgrid
int main(int argc, char **argv)
{
    srand(RANDOM_SEED);

    int ret = starpu_init(nullptr);
    if (ret != 0)
    {
        return 1;
    }

    test_convolution();

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

    return 0;
}
