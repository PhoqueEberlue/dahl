# Clear API and ownership

~~ Rustify my C ~~
Currently in my API, it is not clear if a function allocate/deallocate memory, or from a higher point of view if it create/deletes
a data structure.

Let's compare two functions:
```c
dahl_block* vector_to_block(dahl_vector* vector, dahl_shape3d shape);
dahl_matrix* vector_as_categorical(dahl_vector const* const vector, size_t const num_classes);
```

Here the first function takes a vector and return it as a block.
Under the hood it deallocates the `dahl_vector ` (but not the real vector data) and allocate memory for the `dahl_block` which
will be pointing to the same data.
The advantage is that we didn't have to copy the actual data into a block, we just changed the wrapper object from vector to block.
However it is not perfectly clear that the vector is deleted by the function.
In rust we would just require the function to take ownership of the vector, but here we can't.

Compare that to the second function, here we cannot simply change the wrapper, we actually need to change the data so a copy is made.
Thus vector is not modified, this is indicated by the `dahl_vector const* const`.
Here the language indicates that, so it's fine.

So here my convention is to use `_to_` when a data structure is "morphed" into another, and `_as_` when it is cloned.
But it doesn't feel suitable enough.

Try to find better conventions? Also should I respect the convention in `dahl_tasks` which consists of explicitly writing `_init_` 
when data is allocated by the function?

This problem is also linked to the memory handling, see [memory managment](./memory-managment.md) and [manual partitionning](./data-structure-wrappers.md#getting-the-right-types-with-manual-partitionning) to create views.
