---
tldr: To remove the barier required by arena reset at each end of batch, simply use two batch arenas
and switch between them.
commit: `3e4141a781ccca6abfb77da418e2b98124056ad4`
---

Before:
```c
for (size_t i = 0; i < n_batches_per_epoch; i++)
{
    dahl_tensor_p* pool_out_p = pooling_forward(batch_arena, pool, conv_out_p);
    dahl_tensor_p* pool_out_p = pooling_forward(batch_arena, pool, conv_out_p);
    // etc...

    dahl_arena_reset(batch_arena);
}
```

Here `dahl_arena_reset` is blocking because we must ensure tasks that were using arena's memory are
finished before clearing the arena.

This adds a small delay before launching the next batch.
A simple, yet quite effective fix, is simply to use two arenas, and switch them each batch:

Now:
```c
dahl_arena* batch_arenas[2] = { dahl_arena_new(), dahl_arena_new() };

for (size_t i = 0; i < n_batches_per_epoch; i++)
{
    dahl_arena* batch_arena = batch_arenas[i%2];

    dahl_tensor_p* pool_out_p = pooling_forward(batch_arena, pool, conv_out_p);
    dahl_tensor_p* pool_out_p = pooling_forward(batch_arena, pool, conv_out_p);
    // etc...

    dahl_arena_reset(batch_arenas[(i+1)%2]);
}
```

See ./no-arena-switching.trace and ./arena-switching.trace for comparison.
(both were tested with batch size 60)


Another remark: when launching multiple batch, we see that our two sleep barriers (that we observed
in the previous experiment) are disapearing after a few batches.
But our blocks of tasks are becoming less solid, by that I mean that we notice sleep triangles on
some cpus.
Pretty interesting!

Trying with batch size 120 (./arena-switching-batch-120.trace) we still notice the sleep barrier, on
the two first batches, then it disapears.
However overall scheduling is pretty great, and tasks are pretty solid.
