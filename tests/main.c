#include "tests.h"
#include <stdio.h>
#include <stdlib.h>

#define RANDOM_SEED 42

int main(int argc, char **argv)
{
    srand(RANDOM_SEED);

    dahl_init();

    test_convolution();
    test_tasks();
    test_data();

    dahl_shutdown();

    return 0;
}
