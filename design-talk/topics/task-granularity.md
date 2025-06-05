# Task granularity

Should I make every operations as a task itself? Granularity problem
e.g. here I implemented vector_softmax in one codelet:
```c
void vector_softmax(void* buffers[2], void* cl_arg)
{
    size_t const in_len = STARPU_BLOCK_GET_NX(buffers[0]);
    dahl_fp const* const in = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[0]);

    size_t const out_len = STARPU_BLOCK_GET_NX(buffers[1]);
    dahl_fp* const out = (dahl_fp*)STARPU_BLOCK_GET_PTR(buffers[1]);

    assert(in_len == out_len);

    dahl_fp max_value = 0.0F;

    // Getting max value
    for (size_t i = 0; i < in_len; i++)
    {
        if (in[i] > max_value)
        {
            max_value = in[i];
        }
    }

    dahl_fp sum_values = 0.0F;

    // Shifting by the max value, computing exponent for each element, and summing
    for (size_t i = 0; i < in_len; i++)
    {
        out[i] = exp(in[i] - max_value);
        sum_values += out[i];
    }

    // Computing the probabilities
    for (size_t i = 0; i < in_len; i++)
    {
        out[i] = out[i] / sum_values;
    }
}
```
Here we could separate every loop into its own codelet function:
getting max value, substracting each values, exponate each values, and finally dividing each values.
By doing that we reduce granularity, and build small bricks to reuse code, however we lose optimization opportunities.
In this case substraction, exponent and summing can be done in the same loop.
=> I think granularity should be carefully chosen (not too big, not to small) in order to optimize computing.
```c
dahl_matrix* task_vector_softmax_derivative(dahl_vector const* const in)
{
    dahl_matrix* result = task_vector_diag(in);
    dahl_fp value = task_vector_dot_product(in, in);

    TASK_SUB_VALUE_SELF(result, value);

    return result;
}
```
In this example we have three operations, creating a diagonal matrix from a vector, a dot product and substraction by value.
In this case I think it does not make much sense to group all of those functions in the same codelet thought
