#include "tests.h"

void test_arena_reset()
{
    // Create a new arena to prevent interfering with the other arena
    dahl_arena* tmp_arena = dahl_arena_new();
    dahl_arena_set_context(tmp_arena);

    // This vector and its handel will be initialized in our arena buffer
    dahl_vector* v1 = vector_init_from(5, (dahl_fp[5]){ 0, 1, 2, 3, 4 });

    // Restore the previous context
    dahl_arena_restore_context();

    // Init in the context arena
    dahl_vector* v2 = vector_init_from(5, (dahl_fp[5]){ 0, 1, 2, 3, 4 });

    // Now we want to check if reseting the arena while a task is working creates problems
    TASK_WAIT(v1, 1000); // Lock v1 to really test that dahl_arena_reset() will wait completion
    TASK_ADD_SELF(v2, v1);

    // The arena reset should wait upon any tasks that contains data allocated in the same arena
    dahl_arena_reset(tmp_arena);
    
    ASSERT_VECTOR_EQUALS(vector_init_from(5, (dahl_fp[5]){ 0, 2, 4, 6, 8 }), v2);

    dahl_arena_delete(tmp_arena);

    dahl_arena_reset(testing_arena);
}

void test_arena_dangling_pointers()
{
    dahl_arena* tmp_arena = dahl_arena_new();
    dahl_arena_set_context(tmp_arena);

    dahl_vector* v = vector_init_from(3, (dahl_fp[3]){ 0, 1, 2 });

    vector_print(v);

    dahl_arena_reset(tmp_arena);

    vector_print(v);

    dahl_arena_restore_context();

    dahl_arena_reset(testing_arena);
}

void test_return_values()
{
    dahl_fp* a = dahl_arena_alloc(sizeof(dahl_fp));
    *a = 42.0F;

    dahl_arena_reset(testing_arena);

    dahl_fp* b = dahl_arena_alloc(sizeof(dahl_fp));

    // The memory is not reseted to 0, so the old value is still here
    ASSERT_FP_EQUALS(*b, 42.0F);

}

void test_arena()
{
    test_arena_reset();
    test_return_values();
    // We need a xfail mechanism
    // test_arena_dangling_pointers();
}
