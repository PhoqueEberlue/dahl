#include "arena.h"

#include <assert.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Including the arena backend implementation here to avoid leaking its functions outside this file.
#include "arena_tsoding.h"
#include "starpu_data.h"

size_t HANDLE_REGISTER_COUNT = 0;
size_t HANDLE_UNREGISTER_COUNT = 0;
// TODO: make it dynamic?
#define NMAX_HANDLES 1000
#define NMAX_PARTITIONS 50000
#define NMAX_CONTEXTS 256

static dahl_arena* context_stack[NMAX_CONTEXTS];
static size_t context_index = 0; // The next available space in the context stack

// Redifinition of dahl_partition with only the children handles and the parent data handle
typedef struct
{
    starpu_data_handle_t main_handle;
    size_t nb_children;
    starpu_data_handle_t* children;
} _partition;

// Defining dahl_arena here so it stays opaque in the public API
typedef struct _dahl_arena
{
    Arena arena;
    starpu_data_handle_t* handles;
    size_t handle_count;

    size_t partition_count;
    _partition partitions[];
} dahl_arena;

dahl_arena* dahl_arena_new()
{
    dahl_arena* arena = malloc(
        sizeof(dahl_arena) + 
        // Allocate by default the maximum number of partitions
        (NMAX_PARTITIONS * sizeof(_partition))
    );

    arena->arena.begin = nullptr;
    arena->arena.end = nullptr;

    arena->handles = (starpu_data_handle_t*)malloc(NMAX_HANDLES * sizeof(starpu_data_handle_t));
    arena->handle_count = 0;

    arena->partition_count = 0;

    return arena;
}

void dahl_arena_set_context(dahl_arena* arena)
{
    assert(arena != nullptr);
    assert(context_index < NMAX_CONTEXTS);

    context_stack[context_index] = arena;
    context_index++;
}

dahl_arena* dahl_arena_get_context()
{
    // Here we get (index - 1) because index is the next available space
    dahl_arena* ctx = context_stack[context_index - 1];
    assert(ctx);
    return ctx;
}

void dahl_arena_restore_context()
{
    // Optional: the context should be overwritten anyways
    // context_stack[context_index - 1] = nullptr;
    context_index--;
}

void* dahl_arena_alloc(size_t size)
{
    dahl_arena* ctx = dahl_arena_get_context();
    return arena_alloc(&ctx->arena, size);
}

void dahl_arena_attach_handle(starpu_data_handle_t handle)
{
    dahl_arena* ctx = dahl_arena_get_context();
    HANDLE_REGISTER_COUNT++;
    // TODO: realloc if max number of handles is reached
    assert(ctx->handle_count + 1 < NMAX_HANDLES);

    ctx->handles[ctx->handle_count] = handle;
    ctx->handle_count++;
}

void dahl_arena_attach_partition(starpu_data_handle_t main_handle, size_t nb_children_handle, starpu_data_handle_t* children_handles)
{
    dahl_arena* ctx = dahl_arena_get_context();
    // TODO: realloc if max number of handles is reached
    assert(ctx->partition_count + 1 < NMAX_PARTITIONS);

    _partition *p = &ctx->partitions[ctx->partition_count];

    p->main_handle = main_handle;
    p->nb_children = nb_children_handle;
    p->children = children_handles;

    ctx->partition_count++;
}

void dahl_arena_reset(dahl_arena* arena)
{
    // Make sure to clean the partitions before, otherwise it would attempt to unregister
    // parent handles that have children.
    // Also, loop through the partitions in reverse to clean nested partitions in the right order.
    // Use i+1 in the condition because size_t is unsigned and would otherwise overflow
    for (size_t i = arena->partition_count - 1; i + 1 > 0; i--)
    {
        _partition* p = &arena->partitions[i];

        starpu_data_partition_clean(p->main_handle, p->nb_children, p->children);
        p->main_handle = nullptr;
        p->nb_children = 0;
        p->children = nullptr;
    }

    arena->partition_count = 0;

    for (size_t i = arena->handle_count - 1; i + 1 > 0 ; i--)
    {
        // Using no coherency to prevent data to be copied in the main memory
        starpu_data_unregister_no_coherency(arena->handles[i]);
        arena->handles[i] = nullptr;
        HANDLE_UNREGISTER_COUNT++;
    }

    arena->handle_count = 0;

    arena_reset(&arena->arena);
}

void dahl_arena_delete(dahl_arena* arena)
{
    for (size_t i = 0; i < context_index; i++)
    {
        if (context_stack[i] == arena)
        {
            printf("error, trying to delete arena while it is still in the context stack\n");
            abort();
        }
    }

    // Important to unregister handles and partitions
    dahl_arena_reset(arena);

    arena_free(&arena->arena);
    free((void*)arena->handles);
    free(arena);

    printf("HANDLE_REGISTER_COUNT=%lu\n", HANDLE_REGISTER_COUNT);
    printf("HANDLE_UNREGISTER_COUNT=%lu\n", HANDLE_UNREGISTER_COUNT);
}
