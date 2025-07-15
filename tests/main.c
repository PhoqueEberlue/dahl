#include "tests.h"
#include <stdio.h>
#include <stdlib.h>

#define RANDOM_SEED 42

dahl_arena* testing_arena = nullptr;

int main(int argc, char **argv)
{
    srand(RANDOM_SEED);

    dahl_init();

    // Instanciate a testing arena and set it as context
    testing_arena = dahl_arena_new();
    dahl_arena_set_context(testing_arena);

    // test_layers();
    test_tasks();
    test_data();
    test_arena();

    dahl_arena_restore_context();
    dahl_arena_delete(testing_arena);

    dahl_shutdown();

    return 0;
}
