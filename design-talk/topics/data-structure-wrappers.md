# Data structure wrappers

## Problem

Make a wrapper for starpu matrix/block to provide nice accesors such as get(x, y, z), maybe we can even hide data acquiring under the hood, it can be nice.
=> This is trickier than it seems:
First solution is to do a function, it works very well and can even check with asserts if the index is oob, however it means that in CUDA
I should be able to import this function... Which may be possible?
```c
dahl_fp block_get(starpu_data_handle_t handle, size_t x, size_t y, size_t z)
{
    dahl_fp* block = (dahl_fp*)starpu_block_get_local_ptr(handle);
    size_t ldy = starpu_block_get_local_ldy(handle);
    size_t ldz = starpu_block_get_local_ldz(handle);

    size_t index = (z*ldz)+(y*ldy)+x;

    // TODO: add debug flags macros
    size_t nx = starpu_block_get_nx(handle);
    size_t ny = starpu_block_get_ny(handle);
    size_t nz = starpu_block_get_nz(handle);

    assert(index < nx * ny * nz);

    return block[index];
}
```

With macros we wouldn't have the problem however this looks very bad and its harder to get an assert in there
```c
#define block_get(p, x, y, z, ldy, ldz) (p[((z * ldz) + (y * ldy) + x)])

#define block_get(handle, x, y, z) (\
    ((dahl_fp*)starpu_block_get_local_ptr(handle))\
        [ (z * starpu_block_get_local_ldz(handle))\
        + (y * starpu_block_get_local_ldy(handle))\
        +  x]\
)
```

## Implementing wrappers around starpu_block/matrix/vector

[lun. 17 mars 2025 11:48:23 CET]
-> With my new wrapper (dahl_matrix, dahl_block) this is handled, however we still need to separate matrix and block functions in the codelets functions, even if the implementation could be the same. I'm not sure this is a problem though.
-> everything can be registered as blocks under the hood?
-> seems like a good idea, so we can route corresponding functions e.g.:
```
task_relu_block(dahl_block) -> calls cl_relu_block()
task_relu_matrix(dahl_matrix) -> calls cl_relu_block() // Same because dahl_matrix is under the hood a block and also because relu implementation is looping through all elements one by one so the dimensions does not matter
```
And probably that even for add functions, block and matrix functions could be the same in fact?

=> In the end this is what have been chosen, every data structure (dahl_vector, dahl_matrix, dahl_block) is represented by a StarPU blocks.
I'm not sure if it even adds overhead at all.

I came to the conclusion to only use StarPU blocks because:
Let a starpu block with dimensions (4,4,4), you partition this block into two (2,2,2) sub blocks.
Here the sub blocks will be of type "block" which makes sense.
However if I extract four (1,4,4) sub blocks, they are still blocks, but the idea of this partition is more of accessing sub matrices.
This leads to a problem if we define a function that takes a matrix as parameter.
If my function takes a - starpu - matrix, it won't be able to receive - sub blocks with a matrix shape - which is pretty unconvenient.

I tried to look at ways to maybe convert the data structures defined by starpu but it looked rather complicated and I had to dive into the implementations
details of starpu if I really wanted to do that.
Probably we should ask to the developpers to implement this functionnality, and maybe they could tell us why it isn't implemented, maybe it complexifies
too much the code.

So having a wrapper over the starpu block is very nice because I can virtually create new types based on that:
the user don't have to mind with implementations details, yet it gives my library a layer to do some little optimizations.
For example implementing a flatten function for matrices (or blocks) is as simple as changing the dimensions of the data, the memory isn't touched
as it is contiguously stored anyways.

[later]
Going back on the first topic of this choice, a get function would be possible and it would work on CUDA, however it does'nt make a lot of sense to send
our wrapper objects on CPU/GPU. We should keep the level low as we go down on the layers.
However the get functions could be defined to be accesed directly by the user (without calling a codelet).

## Getting the right types with manual partitionning

[Tue May 27 04:03:10 PM CEST 2025]

Ok, I finally found a way!
Instead of using the builtin partitionning function, in can simply create other handles pointing to the same data.
I just need to make sure to guarantee coherency. This could be done in the actual wrapper functions, so it wouldn't
change the user API, however it would make clearer codelet code with buffers having the correct type and not always starpu block.
```c
void relu_matrix(void* buffers[1], void* cl_arg)
{
    size_t const in_nx = STARPU_MATRIX_GET_NX(buffers[0]);
    size_t const in_ny = STARPU_MATRIX_GET_NY(buffers[0]);
    float* buf = (float*)STARPU_MATRIX_GET_PTR(buffers[0]);

    for (int i = 0; i < in_nx*in_ny; i++)
    {
        if (buf[i] < 0.0F)
        {
            buf[i] = 0.0F;
        }
    }
}

static struct starpu_codelet cl_relu_matrix = {
    .cpu_funcs = { relu_matrix },
    .nbuffers = 1,
    .modes = { STARPU_RW },
};

void relu_vector(void* buffers[1], void* cl_arg)
{
    size_t const in_nx = STARPU_VECTOR_GET_NX(buffers[0]);
    float* buf = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);

    for (int i = 0; i < in_nx; i++)
    {
        if (buf[i] < 0.0F)
        {
            buf[i] = 0.0F;
        }
    }
}

static struct starpu_codelet cl_relu_vector = {
    .cpu_funcs = { relu_vector },
    .nbuffers = 1,
    .modes = { STARPU_RW },
};

struct starpu_codelet cl_switch =
{
	.where = STARPU_NOWHERE,
	.nbuffers = STARPU_VARIABLE_NBUFFERS,
};

// See: https://files.inria.fr/starpu/doc/html/AdvancedDataManagement.html#ManualPartitioning
int main(int argc, char **argv)
{
    int ret = starpu_init(nullptr);
    if (ret != 0)
    {
        return 1;
    }

    // This seems mandatory
    cl_switch.specific_nodes = 1;
	for (int i = 0; i < STARPU_NMAXBUFS; i++)
		cl_switch.nodes[i] = STARPU_MAIN_RAM;

    float mat[3][3] = {
        { 1, -2, 3 },
        { 4, -5, 6 },
        { -7, 8, -9 },
    };

    // Initialize the "main" handle to the previously defined matrix
    starpu_data_handle_t mat_handle;
    starpu_matrix_data_register(&mat_handle, STARPU_MAIN_RAM, (uintptr_t)&mat, 3, 3, 3, sizeof(float));

    matrix_print(mat_handle);

    // Initialize vectors handle, views of the matrix: 
    // manual partitionning -> this way we can define vector data (contrary to the builtin partitionning system that keeps the same type as the parent interface).
    starpu_data_handle_t vectors[3];
    for (size_t i = 0; i < 3; i++)
    {
        starpu_vector_data_register(&vectors[i], STARPU_MAIN_RAM, (uintptr_t)&mat[i], 3, sizeof(float));
        // (optional) Instantly invalidate the vector if it isn't directly used after
		starpu_data_invalidate(vectors[i]);
    }

	struct starpu_data_descr vectors_descr[3];
    for (size_t i = 0; i < 3; i++)
    {
        vectors_descr[i].handle = vectors[i];
        vectors_descr[i].mode = STARPU_W;
    }

    // This function refreshes the vectors data so they become valid again
    // STARPU_W -> enables data
    // STARPU_RW -> does not enable data it seems
    // So here all the vectors in vectors_descr are enabled again
	ret = starpu_task_insert(&cl_switch, STARPU_RW, mat_handle, STARPU_DATA_MODE_ARRAY, vectors_descr, 3, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

    // Invalidate the main handle to prevent concurrent data accesses
	starpu_data_invalidate_submit(mat_handle);

    for (size_t i = 0; i < 3; i++)
    {
        // We can actually call a codelet that takes a vector and not a matrix thanks to our view
        ret = starpu_task_insert(&cl_relu_vector, STARPU_RW, vectors[i], 0);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    }

    // Refresh the main handle
	ret = starpu_task_insert(&cl_switch, STARPU_W, mat_handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
 
    // Print the result
    matrix_print(mat_handle);

    // Also works with the matrix type
    ret = starpu_task_insert(&cl_relu_matrix, STARPU_RW, mat_handle, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* terminate StarPU, no task can be submitted after */
	starpu_shutdown();

    return 0;
}
```

Pros:
- Should better transition to higher dimensions data structures
- Cleaner codelet code with the right types
- Leaves us the opportunity to implement a view mechanism based on the same principle, whereas with the previous implem I would
  have to define two separate mechanisms. However it means I have to pay attention to coherency?
Cons:
- Data should be contiguous for the views to work -> for now I didn't need it because I wisely used my data structures knowing
  how they would be accessed after.
- This also means I will have to do data partitionning myself (or reuse the builtin system and lose the "type changing" ability)

## Getting the right types using starpu builtin filters

[Wed May 28 11:49:42 AM CEST 2025]
Ok, there's a way to actually do it from the StarPU API... Which makes sense but wow how do I missed that.

Since the beginning I'm partitionning my data structures with options like: `starpu_block_filter_depth_block` but there also exist
`starpu_block_filter_pick_matrix_z` that returns a matrix interface...

Pros:
- Should better transition to higher dimensions data structures -> still valid
- Cleaner codelet code with the right types -> still valid
- ~~Leaves us the opportunity to implement a view mechanism based on the same principle, whereas with the previous implem I would
  have to define two separate mechanisms. However it means I have to pay attention to coherency?~~
  -> the view system will have to be implemented with the manual partitionning I think. I did not find a way to do it directly in StarPU.
Cons:
- ~~Data should be contiguous for the views to work -> for now I didn't need it because I wisely used my data structures knowing
  how they would be accessed after.~~
- ~~This also means I will have to do data partitionning myself (or reuse the builtin system and lose the "type changing" ability)~~

## Custom filter

[Thu Jun  5 02:10:27 PM CEST 2025]
It is also really easy to define a custom filter in starpu, for example here I wrote a filter that can be applied on a block
in order to partition the z dimension into flattened matrices, i.e. instead of having regular matrices with (x,y) we get vectors with x*y lenght.

```c
static void starpu_block_filter_pick_matrix_z_as_flat_vector(
    void* parent_interface, void* child_interface, 
    STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter* f,
    unsigned id, unsigned nparts)
{
	struct starpu_block_interface* block_parent = (struct starpu_block_interface *) parent_interface;
	struct starpu_vector_interface* vector_child = (struct starpu_vector_interface *) child_interface;

	unsigned blocksize;

	size_t nx = block_parent->nx;
	size_t ny = block_parent->ny;
	size_t nz = block_parent->nz;
	
    blocksize = block_parent->ldz;

	size_t elemsize = block_parent->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG(nparts <= nz, "cannot get %u vectors", nparts);
	STARPU_ASSERT_MSG((chunk_pos + id) < nz, "the chosen vector should be in the block");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(block_parent->id == STARPU_BLOCK_INTERFACE_ID, "%s can only be applied on a block data", __func__);
	vector_child->id = STARPU_VECTOR_INTERFACE_ID;

    vector_child->nx = nx*ny;

	vector_child->elemsize = elemsize;
	vector_child->allocsize = vector_child->nx * elemsize;

	if (block_parent->dev_handle)
	{
		if (block_parent->ptr)
			vector_child->ptr = block_parent->ptr + offset;
		
		vector_child->dev_handle = block_parent->dev_handle;
		vector_child->offset = block_parent->offset + offset;
	}
}

struct starpu_data_interface_ops *starpu_block_filter_pick_vector_child_ops(
    STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_vector_ops;
}
```
