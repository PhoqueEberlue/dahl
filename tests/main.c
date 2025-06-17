#include "tests.h"
#include <stdio.h>
#include <stdlib.h>

#define RANDOM_SEED 42

dahl_arena* test_arena = nullptr;

int main(int argc, char **argv)
{
    srand(RANDOM_SEED);

    dahl_init();

    test_arena = dahl_arena_new();

    test_layers();
    test_tasks();
    test_data();

    dahl_arena_delete(test_arena);

    dahl_shutdown();

    return 0;
}
