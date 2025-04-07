#include "tests.h"
#include <stdlib.h>

#define RANDOM_SEED 42

int main(int argc, char **argv)
{
    srand(RANDOM_SEED);

    dahl_init();

    // test_dahl_convolution();
    test_tasks();

    dahl_shutdown();

    return 0;
}
