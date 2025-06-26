#include "tests.h"

void test_arena_reset()
{
    // Create a new arena to prevent interfering with the other arena
    dahl_arena* tmp_arena = dahl_arena_new();
    dahl_arena const* dahl_save = dahl_context_arena;
    dahl_context_arena = tmp_arena;

    // This vector and its handel will be initialized in our arena buffer
    dahl_vector* v1 = vector_init_from(5, (dahl_fp[5]){ 0, 1, 2, 3, 4 });

    // Restore the previous arena
    dahl_context_arena = dahl_save;

    // Init in the context arena
    dahl_vector* v2 = vector_init_from(5, (dahl_fp[5]){ 0, 1, 2, 3, 4 });

    // Now we want to check if reseting the arena while a task is working creates problems
    TASK_ADD_SELF(v2, v1);

    // The arena reset should wait upon any tasks that contains data allocated in the same arena
    dahl_arena_reset(tmp_arena);
    
    ASSERT_VECTOR_EQUALS(vector_init_from(5, (dahl_fp[5]){ 0, 2, 4, 6, 8 }), v2);
}

void test_arena()
{
    test_arena_reset();
}
