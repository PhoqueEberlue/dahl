---
tldr: Just remove some partition calls? Why didn't I think about it earlier?
commit: `0c889d4bdde60b2fd3cebe6730b1921a8067aa6e`
---

In the convolution layer I have something like this:

```c
dahl_tensor* convolution_forward(...)
{
    // Initialize the result tensor
    dahl_tensor* output_batch = tensor_init(arena, conv->output_shape);
    
    tensor_partition_along_t(output_batch, DAHL_MUT);
    tensor_partition_along_t(input_batch, DAHL_READ);
    tensor_partition_along_t(conv->filters, DAHL_READ);

    for (size_t i = 0; i < batch_size; i++)
    {
        // launch tasks per sample
    }
    
    tensor_unpartition(output_batch);
    tensor_unpartition(input_batch);
    tensor_unpartition(conv->filters);

    return output_batch;
}
```

So I partition everything, launch task for each sample, then unpartition everything.
Unpartitioning also means, synchronizing everything so we get our complete tensor with every values
correct.

But then in the next layer I will do the same:
(note that I refactored RELU to be a layer itself, that what made me think about this in fact).

```c
void relu_forward(dahl_relu* relu, dahl_tensor* input_batch)
{
    tensor_partition_along_t(input_batch, DAHL_MUT);
    tensor_partition_along_t(relu->mask_batch, DAHL_MUT);
    size_t const batch_size = GET_NB_CHILDREN(input_batch);

    for (size_t i = 0; i < batch_size; i++)
    {
        dahl_block* input = GET_SUB_BLOCK_MUT(input_batch, i);
        dahl_block* mask = GET_SUB_BLOCK_MUT(relu->mask_batch, i);
        TASK_RELU_SELF(input, mask);
    }
    
    tensor_unpartition(input_batch);
    tensor_unpartition(relu->mask_batch);
}
```

So here I get the input_batch, which is the output from the convolution, partition it again, do
things, then unpartition again.
In fact from the beginning we could just, not unpartition, and prevent synchronizing at each
layer???

Looking at a comparison, we clearly see it changes a lot.
That's very good, however we will need to integrate this cleanly in the API: Should I always let the
output result partitioned? Does the user need to unpartition it by hand? How do I recognize which
layer is the first/last one?

After trying this with other layers I face another issue:

If I apply other types of partitionning on the data, it seems its not possible.
For example from relu to pooling, the data gets partitioned as such:
- relu partition batch dim
- pool partition batch dim, then filter dim for each sample

However If I apply the same trick as before, starpu crashes because we try to partition data that is
empty. That's a concerning limitation, yet it makes sense because I could in theory publish all the
partition plans of data and its sub data in one pass and I would mess up the results.

1. One solution is to remove partitioning on the filter dimension.
So we would rework codelets to produce multiple feature maps instead of one.
We would also need to reword the flatten layer, so that it returns partitioned data.
  - But concerning the flatten layer I think we may have problems anyways, because I invalidate the
    previous handle (tensor), create a new handle (matrix) but this last will lack synchronization with
    the tensor I think.
    A simple fix is to simply use a copy task, yet we lose the no_copy advantage of the other
    method. Of course we gain huge scheduling advantages, but it would be nice to have both!
  - Dense backward is not a problem because it only parallelizes on batch dim
  - Lastly we should revert loss, pred, and gradients tasks to what we had before:
    Tasks per sample, that gets parallelized using starpu partition.
    This way we keep using the same handles that are already partioned.
-> Nevertheless, we should really think of organizing similar tasks (but with different partition
granularity). For example, do I write two times cross_entropy_batch and cross_entropy? Can't I just
have a codelet that either launches cross_entropy_batch with matrices as arguments and cross_entropy
with vectors as arguments?
-> The advantage of this last point is that it will be very interesting for "self adjusting task
size scheduling".

2. Another solution, fast but not clean (maybe try this before hopping into big refactors from
   solution 1?), it to submit wait tasks (with 0 seconds of wait, we just want to acquire the data)
   so that we ensure that the partition is done after?

-> Turns out solution 2 is really great, and does work seamlessly
See ./removing_partition_call_conv_relu_pool_forward_and_backward.trace
We still notice sleep time between relu backward and conv backward, yet theere is no task to blame
for, which is pretty strange.
The other layers are pretty solid on the graph, it seems like this is a really great solution.

Now we just need to refactor loss,pred,grad and flatten layer so that the dense layer can use the
same trick too.
