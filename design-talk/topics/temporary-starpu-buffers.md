# Temporary StarPU buffers

[Wed Sep  3 09:48:27 AM CEST 2025]

Context: Arena, StarPU temporary buffers

One problem that I face quite often is that I need some temporary buffer but can't manage to find a nice tradeoff for handling their memory in terms of API design.

Take this function for example:

```c
void task_cross_entropy_loss_batch(dahl_matrix const* prediction_batch, dahl_matrix const* target_batch, dahl_scalar* out)
{
    dahl_fp const epsilon = 1e-12F;
    dahl_matrix* tmp = matrix_init( <arena?> , matrix_get_shape(prediction_batch));
    TASK_CLIP(prediction_batch, tmp, epsilon, 1 - epsilon);

    int ret = starpu_task_insert(&cl_cross_entropy_loss_batch,
                                 STARPU_R, tmp->handle,
                                 STARPU_R, target_batch->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}
```

Here I just want to call the clip task (to delete values < epsilon and > 1 - epsilon) before submitting the cross entropy task.
If I directly clipped the values in the cross entropy task we wouldn't have any problem, but, well, I already implemented the clip task so we may reuse it no?
And especially I want to keep the variable `prediction_batch` unmodified.
So we need a temporary buffer, but who should own the memory?

Intuitively it should be owned by the function itself, because the buffer `tmp` will only be used to store partial results.

It makes sense but it is particularly difficult to do when using arena and to keep our function non-blocking.
So here the problem is related to which `<arena?>` should we chose.

## Creating an arena on the fly 

```c
void task_cross_entropy_loss_batch(dahl_matrix const* prediction_batch, dahl_matrix const* target_batch, dahl_scalar* out)
{
    dahl_arena* tmp_arena = dahl_arena_new();

    dahl_fp const epsilon = 1e-12F;
    dahl_matrix* tmp = matrix_init(tmp_arena, matrix_get_shape(prediction_batch));
    TASK_CLIP(prediction_batch, tmp, epsilon, 1 - epsilon);

    int ret = starpu_task_insert(&cl_cross_entropy_loss_batch,
                                 STARPU_R, tmp->handle,
                                 STARPU_R, target_batch->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");

    dahl_arena_delete(tmp_arena);
}
```

One way is to create an arena 'on the fly', yet it is very uneficient and pretty equivalent to what a malloc/free version would look like in fact.
The second reason, is that even if we did define a global arena made for partial results, it would be still blocking the function if we have to reset the arena at the end of the function.
So we could also decide to reset at certain given time but it may be hard to estimate when to do that (looks like a garbage collector too).

## Asking for a scratch arena explicitly

```c
void task_cross_entropy_loss_batch(dahl_arena* scratch_arena, dahl_matrix const* prediction_batch, dahl_matrix const* target_batch, dahl_scalar* out)
{
    dahl_fp const epsilon = 1e-12F;
    dahl_matrix* tmp = matrix_init(scratch_arena, matrix_get_shape(prediction_batch));
    TASK_CLIP(prediction_batch, tmp, epsilon, 1 - epsilon);

    int ret = starpu_task_insert(&cl_cross_entropy_loss_batch,
                                 STARPU_R, tmp->handle,
                                 STARPU_R, target_batch->handle,
                                 STARPU_W, out->handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}
```

We could also ask for a scratch arena to be passed as a parameter.
This works, but it is now the user's responsability to reset the arena at some point.
In my case this is sufficient because I have scratch arenas that are resetted upon each batch iterations.
However this does not look right in terms of API design and memory ownership.
Morever, if I implement this function in `_init` mode (instead of passing the result by pointer to an already existing object, the function allocates the return object itself) it means that we now have two arenas to be passed by parameter, which makes things confusing:

```c
dahl_scalar* task_cross_entropy_loss_batch_init(dahl_arena* scratch_arena, dahl_arena* arena, dahl_matrix const* prediction_batch, dahl_matrix const* target_batch)
{
    dahl_scalar* res = scalar_init(arena);
    task_cross_entropy_loss_batch(scratch_arena, prediction_batch, target_batch, res);
    return res;
}
```

Here `arena` will store the result object, whereas `scratch_arena` will contain the partial result.
These could be the same arenas, or totally different, anyways it doesn't matter to the result as they wont use the partial results.

At least this solution is non-blocking.

## Using StarPU temporary buffers

Another solution would be to use built-in StarPU temporary buffers.
From the documentation:

```c
// The following code examplifies both points: it registers the temporary data, submits three tasks accessing it, and records the data for automatic unregistration.
starpu_vector_data_register(&handle, -1, NULL, n, sizeof(float));
starpu_task_insert(&produce_data, STARPU_W, handle, 0);
starpu_task_insert(&compute_data, STARPU_RW, handle, 0);
starpu_task_insert(&summarize_data, STARPU_R, handle, STARPU_W, result_handle, 0);
starpu_data_unregister_submit(handle);
```

Using `-1` and `NULL` we don't need to allocate/deallocate any vector data, StarPU will handle it automatically.

However, to make it compatible with our API functions, it means that we should wrap the handle of the temporary buffer inside a `dahl_vector`.
This means having to handle heap allocated memory for the `dahl_vector` object itself, because right now it is not possible to instanciate a `dahl_vector` on the stack.
This restriction exists because I use opaque types to hide `starpu_data_handle_t` in my wrapper objects so that the starpu includes does not leak to the user application.
Maybe this is too strict, I don't know.
We could potentially do a trick to stack allocate a vector anyways?
