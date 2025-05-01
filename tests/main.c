#include "tests.h"
#include <stdio.h>
#include <stdlib.h>

#define RANDOM_SEED 42

int main(int argc, char **argv)
{
    srand(RANDOM_SEED);

    dahl_init();

    dahl_arena* arena = arena_new(10'000);
    // test_dahl_convolution();
    // test_tasks();
    test_data(arena);

    dahl_shutdown();

    return 0;
}
