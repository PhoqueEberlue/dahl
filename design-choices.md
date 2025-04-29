-------------------------------------------------------------------------------
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


Going back on the first topic of this choice, a get function would be possible and it would work on CUDA, however it does'nt make a lot of sense to send
our wrapper objects on CPU/GPU. We should keep the level low as we go down on the layers.
However the get functions could be defined to be accesed directly by the user (without calling a codelet).

-------------------------------------------------------------------------------
Make dahl_matrix / dahl_block generic? they could use the type we want? -> seems hard, need to think about it.
```c
#include <stdio.h>
#include <stdlib.h>
#define N 500000000

enum type {
    type_int,
    type_double,
};

void* int_get(void* array, size_t index)
{
    return (void*) &( (int*)array )[index];
}

void int_add(void* a, void* b)
{
    *(int*)a += *(int*)b;
}

int main()
{
    int* array = malloc(N * sizeof(int));

    for (size_t i = 0; i < N; i++)
    {
        array[i] = 42;
    }

    void* array_v = (void*)array;
    enum type ty = type_int;
    
    // Solution 1: no genericity | avg: 1.50sc runtime
    for (size_t i = 0; i < N - 1; i++)
    {
        array[i+1] += array[i];
    }

    // Solution 2: function pointers | avg: 2.63sc runtime
    // Here we can do those assignements in if/else to chose the correct type
    if (ty == type_int)
    {
        void* (*get)(void*, size_t) = &int_get;
        void (*add)(void*, void*) = &int_add;
    }

    for (size_t i = 0; i < N - 1; i++)
    {
        void* val_a = get(array_v, i);
        void* val_b = get(array_v, i+1);

        int_add(val_b, val_a);
    }

    // Solution 3: enum and if else in the loop | avg: 1.62sc runtime
    // It seems that the branch predictor is doing a good job, yet it's still bellow the optimal.
    for (size_t i = 0; i < N - 1; i++)
    {
        if (ty == type_int)
        {
            ((int*)array_v)[i+1] += ((int*)array_v)[i];
        }
        else if (ty == type_double)
        {
            ((double*)array_v)[i+1] += ((double*)array_v)[i];
        }
    }
}
```
- Solution 1 is the actual one, this is the fastest because only one type can be used at runtime, so everything is perfectly tailor-made.
- Solution 2 keeps the code in a readable state, however the performance for calling the functions, passing by address and doing the good casts
does severely degrade the performance.
- Solution 3 makes the code awful but it is performing almost as great as solution 1, probably thanks to branch prediction.
- Solution 4 involves using C++ templates, however it is still mandatory to link everything together by hand for each codelet for each type... 
Nntile does that, the code is indeed less readable.

[lun. 24 mars 2025 14:57:44 CET]
I had to think about type generecity because I needed to store indexes into a block.
However I implies huge API changes and complexifies greatly the code.
Instead I used a mask, a block with 0 and 1 to enable/disable selected indexes, this way you can multiply the mask with another block to only keep
the indexes you want. The mask is still implemented with floating point values which makes sense when you do multiply it with another blocks.

[jeu. 24 avril 2025 16:43:38 CET]
In following to this problem, we can note that in reality the best way to implement that would be Solution 1, but with C++ templates.
I hesitated to transition my code to C++ at some point, however this is still not convenient for one reason:
StarPU is written in C. So, yes I can slap some templates on every of my functions, however I will still have
to write (or generate) my codelets and function linking for EVERY CODELET FOR EVERY TYPE.
This is basically what [nntile](https://github.com/nntile/nntile) is doing, they built a wrapper around the starpu codelets to automatically 
generate that.
For the sake of keeping the code simple and to differentiate from nntile, I kept my code "unityped".
This doesn't mean however that I am bounded to a single data type, every function is using a `dahl_fp` which is actually a typedef of double.
So you can use another type but the software should be recompiled.

rmq: some functions that take exclusively doubles will stop working if you change the type. This should be addressed later.

-------------------------------------------------------------------------------
[ven. 28 mars 2025 08:26:17 CET]
~ Macro nightmare ~
```c
#define DEFINE_STARPU_CODELET(func_name, num_buffers, ...)                         \
    void func_name(void *buffers[num_buffers], void *cl_arg);                      \
                                                                                   \
    static struct starpu_perfmodel perf_model_##func_name = {                      \
        .type = STARPU_HISTORY_BASED,                                              \
        .symbol = "perf_model_" #func_name                                         \
    };                                                                             \
                                                                                   \
    static struct starpu_codelet cl_##func_name = {                                \
        .cpu_funcs = { func_name },                                                \
        .nbuffers = num_buffers,                                                   \
        .modes = { __VA_ARGS__ },                                                  \
        .model = &perf_model_##func_name                                           \
    };                                                                             \
                                                                                   \
    void call_##func_name(SELECT_ARGS(num_buffers, starpu_data_handle_t))          \
    {                                                                              \
        int ret = starpu_task_insert(&cl_##func_name,                              \
                                     SELECT_ACCESS_HANDLE(num_buffers, __VA_ARGS__)\
                                     0);                                           \
                                                                                   \
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");                \
    }


#define GET_ARG_1(A, ...) A
#define GET_ARG_2(_1, A, ...) A
#define GET_ARG_3(_1, _2, A, ...) A
#define GET_ARG_4(_1, _2, _3, A, ...) A
#define GET_ARG_5(_1, _2, _3, _4, A, ...) A

#define ARG(N, T) T a##N
#define ARG_LIST_1(T) ARG(1, T)
#define ARG_LIST_2(T) ARG_LIST_1(T), ARG(2, T)
#define ARG_LIST_3(T) ARG_LIST_2(T), ARG(3, T)
#define ARG_LIST_4(T) ARG_LIST_3(T), ARG(4, T)
#define ARG_LIST_5(T) ARG_LIST_4(T), ARG(5, T)
#define SELECT_ARGS(N, T) ARG_LIST_##N(T)

#define ACCESS_HANDLE(N, ...) GET_ARG_##N(__VA_ARGS__), a##N,
#define ACCESS_HANDLE_LIST_1(...) ACCESS_HANDLE(1, __VA_ARGS__)
#define ACCESS_HANDLE_LIST_2(...) ACCESS_HANDLE_LIST_1(__VA_ARGS__) ACCESS_HANDLE(2, __VA_ARGS__)
#define ACCESS_HANDLE_LIST_3(...) ACCESS_HANDLE_LIST_2(__VA_ARGS__) ACCESS_HANDLE(3, __VA_ARGS__)
#define SELECT_ACCESS_HANDLE(N, ...) ACCESS_HANDLE_LIST_##N(__VA_ARGS__)

DEFINE_STARPU_CODELET(matrix_cross_correlation, 3, STARPU_R, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(matrix_max_pooling, 3, STARPU_R, STARPU_W, STARPU_W)
DEFINE_STARPU_CODELET(matrix_backward_max_pooling, 3, STARPU_R, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(relu, 2, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(block_sum_z_axis, 2, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(scal, 2, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(sub, 3, STARPU_R, STARPU_R, STARPU_W)
DEFINE_STARPU_CODELET(add, 3, STARPU_R, STARPU_R, STARPU_W)
```

Problem: for each codelet function, I need to write a codelet struct, a perfmodel struct and the signature of the function.
This can be easly generated using a macro.

[jeu. 24 avril 2025 17:12:02 CEST]
I also tried to do a wrapper function that actually makes the starpu task insert of the codelet but this is way too much because
we need to handle multiple arguments for the buffers and it becomes messy real quick.
Also in the current state of dahl, many of those functions hide some specific work in this wrapper function, e.g. if I do a matrix matrix
multiplication, my wrapper can automatically instanciate an output matrix with the right dimensions.

-------------------------------------------------------------------------------
Simplifying the api by providing a common type "any" that would let you pass anything (block, matrix, vector) so that the functions that performs elementary operations could be called by the same function (e.g. ADD is the same implementation for every data types).
Actualy if I did a simple function to convert a type, let's say block, to the any type, it would complexify the user code by adding a lot of local variables in their code for each "type casting".
Another solution is to only use any type as a stack allocated object, thus it can be passed directly by value, so we can call `operator(block_as_any())` where operator takes a any.
However it again adds functions with long names (block_as_any, vector_as_any etc.), and if an operator has a lot of parameters it can become a mess.
Instead we can use macros to handle that, as type is knowned at compilation anyways, giving us more syntax sugar
```c
struct a
{
    int i;
};

struct b
{
    int i;
};

struct any
{
    union data
    {
        struct a* a;
        struct b* b;
    } data;

    enum type
    {
        type_a,
        type_b,
    } type;
};

#define AS_ANY(x) _Generic((x),              \
    struct a*:                               \
        (struct any)                         \
        {                                    \
            .data = { .a = (struct a*)(x) }, \
            .type = type_a                   \
        },                                   \
    struct b*:                               \
        (struct any)                         \
        {                                    \
            .data = { .b = (struct b*)(x) }, \
            .type = type_b                   \
        }                                    \
)

void fn(struct any an)
{
    // actually checks with value any is
}

int main()
{
    struct a a = { .i = 42 };
    struct b b = { .i = 12 };

    fn(AS_ANY(&a));
    fn(AS_ANY(&b));
}
```

[jeu. 24 avril 2025 17:14:52 CEST]
Here AS_ANY is very practical it automatically guesses the type at compile time, however because it add too much verbosity in the function calls,
I directly implemented macros that do the AS_ANY by themselves. In this example it would give:

```c
void fn(struct any an)
{
    // actually checks with value any is
}

#define FN(X) fn(AS_ANY((X)))

int main()
{
    struct a a = { .i = 42 };
    struct b b = { .i = 12 };

    FN(&a);
    FN(&b);
}
```
This is more readable. We can argue that it hides the function definition, which is true but in dahl those macros are defined next to the 
original functions so you can always go to the macro definition and take a look at the function upside:

```c
// Performs `c` = `a` + `b`, where:
// - `+` is the value by value addition
// - `a`, `b` and `c` are dahl_any objects of the same shape
void task_add(dahl_any const a, dahl_any const b, dahl_any c);
#define TASK_ADD(A, B, C) task_add(AS_ANY(A), AS_ANY(B), AS_ANY(C))
```

As a bonus I also added the support for const in the AS_ANY macro.

Following this macro I also added FROM_ANY macro to unpack the result of a function that returns a dahl_any.
```c
#define FROM_ANY(IN, OUT) _Generic((IN), \
        dahl_block*:                     \
            (dahl_block*)                \
            {                            \
                (OUT).structure.block    \
            },                           \
        dahl_matrix*:                    \
            (dahl_matrix*)               \
            {                            \
                (OUT).structure.matrix   \
            },                           \
        //...
        dahl_matrix const*:              \
            (dahl_matrix*)               \
            {                            \
                (OUT).structure.matrix   \
            },                           \
        dahl_vector const*:              \
            (dahl_vector*)               \
            {                            \
                (OUT).structure.vector   \
            }                            \
    )
```
It works by checking the type of `IN` thus returning the same type by unpacking `OUT`.
It can be used like this:

```c
dahl_any any_clone(dahl_any const any);
#define ANY_CLONE(X) FROM_ANY(X, any_clone(AS_ANY(X)))
```

The disadvantage is that the function should always return the same type than the one passed as a parameter (but that makes sense).
Also we can argue that the code become more and more complex in terms of macro, and maybe it's too much.
I think that those macros are okay because they don't hide complex code execution / branches.
If something is wrong, it should be detected at compile time.
However one caveat that I noticed is that functions with multiple dahl_any parameters can be called with different types :)
which is not good.

One last remark: I probably shouldn't abuse of ANY. This is very useful for the task functions, but probably shouldn't be used
for defining common functions between the types themselves. Typically, I don't really want a ANY_PRINT that can print block, matrix, or vector.
In other words, the user should always know the types, but my task API can forgot them (because it's all starpu blocks in the back anyways).
I don't want the API to feel like python where you can call `np.dot(s, s.T)` where s could literally be anything, and based on that np.dot
will decide if it performs an inner product of vectors, matrix matrix multiplication, scalar multiplications, sum of products and so on.

-------------------------------------------------------------------------------
Simplify and/or homogeneize the way tasks returns values.
For example, let `add` that takes a and b parameters and write a + b in c
This function can be written as:
```c
// 1: Return the result via the pointer c
void add(void* a, void* b, void* c);

// 2: Simply return the result with the return keyword
void* add(void* a, void* b);

// 3: Writes the result directly in buffer a
void add_self(void* a_self, void* b);
```
All of those solutions are correct but they serve different purposes in terms of memory management.
1. forces the user to instanciate c on its own, which may reduce the possibility to forget calling finalize to free the memory.
Also it is very useful when another buffer can be reused to store the result into c. For example we might perform operations on
sub matrices of a block, and in this case we want to store every sub result in a common block buffer.
2. makes less writing for the user, the returned objected can be instanciated with the correct dimensions by the function.
Yet the user shouldn't forget to free the memory.
3. Also very useful when directly writing the result to the buffer a is not a problem.
For now, most of the tasks implement syntax 1 and 3 
- Should I implement the three ways for every task? 
- Will it cost a lot to maintain or add too much complexity?
- find a nice way to differentiate the 3 functions and add it in the doc

[sam. 12 avril 2025 10:34:19 CEST]
-> This would be very interesting not to allocate the same buffers over and over again: maybe try to reuse the buffers?
This is why it is important to have functions that take buffers pointers.
[jeu. 17 avril 2025 10:47:48 CEST]
-> "", "self", and "init" are good names to differentiate those three modes
[jeu. 24 avril 2025 18:27:47 CEST]
After all I'm not sure "init" should be added, because the function return already indicates that something is initialized right?
The problem is that in the task api and in the data api we might have diverging conventions.

-------------------------------------------------------------------------------
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

-------------------------------------------------------------------------------
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

This problem is also linked to the memory handling, see next problem

-------------------------------------------------------------------------------
Memory handling
Problem:

Currently (mar. 29 avril 2025 15:07:57 CEST) my API is implemented as follows:
dahl_data.h contains functions that are able to allocate memory dynamically e.g. matrix_init, block_init etc., and have their
corresponding finalize functions.
the codelets themself are not allocating data at all, just using the provided buffers.
However the dahl_tasks.h layer is able to allocate data by calling functions from dahl_data.h.
e.g. `task_vector_diag` takes a vector, and instanciates a new matrix with vector lenght * vector lenght as its dimensions.

Those functions are also used in the different layers of my CNN.
Which means that every epoch, and every sample, data is allocated.
This is not good for multiple reasons:
- First it can be difficult to track the data dependencies, and where a buffer should actually be finalized (lifetime issue).
- furthermore, starpu work asynchronously so I should await the data to be available (i.e. all tasks related to it ended) in order to free it.
- even if I manage to free my memory in the correct order, it would still be highly inneficient as the allocations require system calls.

What we could do instead, would be to reuse the buffers.
At each epoch/sample iteration, we perform exactly the same operations, with exactly the same needs in terms of memory.
So in theory, the maximal memory size can be known at compile time.

One strategy to allocate memory as such is called [Linear allocation/Arena/Region-based allocator].
It makes reasonning in terms of grouped elements instead of single elements.
And you can free all the elements in the group at the same time, without worrying about the data dependency order, pointers etc.

Some readings on that:
https://www.rfleury.com/p/untangling-lifetimes-the-arena-allocator
https://www.gingerbill.org/article/2019/02/01/memory-allocation-strategies-001/
About Rust and Cpp RAII:
"[...] Such languages also have the tendency to couple the concept of ownership with the concept of lifetime, **which are not necessarily linked.**"
I shouldn't forget to align my memory

Let's get back to my example.
I could use my arena in multiple ways:

```md
1.
create my arena with MEM_MAX bytes, actual memory allocation by calling malloc, nmap, windows virtual memory whatever.

for epoch in epochs
    for sample in sample
        allocate using the arena -> O(1)
        forward pass
        backward pass
        reset my arena -> fill the buffer with 0

delete my arena, actual memory deallocation

2.

create my arena with max(MEM_MAX_FORWARD, MEM_MAX_BACKWARD) bytes.

for epoch in epochs
    for sample in sample
        allocate using the arena
        forward pass

        reset my arena

        allocate using the arena
        backward pass

        reset my arena

delete my arena, actual memory deallocation
```

In 1., I allocate and deallocate at the beginnig and the end of the program respectively.
In fact, I could even omit deallocating the memory, the OS would do it for me anyways.
Yet it is not equivalent to malloc'ing everywhere and not freeing the memory because it's harder to manage (like my actual implem).
Here we allocate one buffer, and reuse it for all the iterations.
The memory consumption is constant, let's say MEM_MAX.
The program will be at MEM_MAX from beginning to end.
This obviously constrasts with my current version where the memory is continuously growing (huge memory leak).
So this is way better.

But this solution is not limited to a constant MEM_MAX, which could be problematic when your machine's memory is inferior than that. 
In solution 2. we show that the arena can be resetted between the forward and the backward pass.
This way, the minimum amount of memory is now the max(MEM_MAX_FORWARD, MEM_MAX_BACKWARD).
The memory granualarity can be chosen.
Another solution could also consists of declaring context arenas, e.g. one for the forward, one for the backward, or even lower in granularity.

Sadly, it is harder to do with starpu.
Apparently I can override `starpu_malloc` (and free) with `starpu_malloc_set_hooks` like in this [example](https://gitlab.inria.fr/starpu/starpu/-/blob/master/examples/basic_examples/hooks.c?ref_type=heads).
But it seems the data handles cannot be allocated freely wherever I want.
This is not a big deal in terms of performances because the handles are just the wrapper to the actual data (which is could allocate with arenas).
However it still complexify the data management, I still need to know when a buffer can be freed.
Because at some point I need to call starpu_data_unregister.

In theory I could implement an arena with the global buffer (everything is stored inside) + keeping track of every starpu_handles that have some data
in this buffer.
I could unregister all of that in the arena reset.
So it would imply actual memory dealloction (which is not dramatic for handles).
But, it makes my API more heterogenous, which I don't really like.
It means the GPU allocations will be treated differently, with starpu's care.

Do I need to manage GPU allocations myself? 
I think knowing the memory used be my program is very important in order to optimize it.
Especially for distributing computation on heterogeneous machines: I want to distribute the task in order to
decrease the energy consumption but I need to take into considerations limits of each machines.
This obviously includes the processing unit speed, number, and types (gpu, cpu), the bandwidth,
but also maximum memory of each processing unit (gpu, cpu).
As a side note I think memory managment is also important to reduce for energy consumption.
Also it should lead to better usage of the cache, thus speeding up the execution speed.

What do I lose from leaving starpu?
- Asynchronous handling of tasks for threads and gpu executions (which is an enormous point -> ⚠️ mutex/semaphore ⚠️)
- automatic perf models (which can be reimplemented pretty quickly I would say)
- Auto-managment of memory between ram and GPU integrated memory (but the point is to manage it ourselves)
- More work on developping with CUDA probably, however I think I still had to do most of stuff (other than  memory managment) with starpu.

What do I gain?
- I develop my own scheduler: I want to create an optimization software that takes a ML Model, a platform and return the theoritical optimum 
  for training this model with the minimum energy consumption (partition). With starpu the task scheduling will be done at runtime, this can be great
  but it will hurt reproducibility (I'm not sure I can simulate it), and I won't be able to execute my partition like the optimization software told us 
  to do. *But maybe this is a bad idea to do that from the beginning?*
- the code becomes more "deterministic" in terms of execution flow, which will lead to an easier time with simulating the runtime. (We could even try to separate the "execution flow" (So the ML API wrapper in fact?) from the implem, network and executions to integrate it in Simgrid)
- full control of GPU allocations, I can take into account the GPU allocation in the optimizing software
- I control my data structures myself, my three wrappers (block, matrix, vector) are no longer all pointing to a `starpu_block` just because the sub data of a block is a block. 
  I can specialize and decide how the data is stored. I can take every optimization in terms of data storing and use my sturctures cleanly
  in the codelets for cpu or CUDA.
