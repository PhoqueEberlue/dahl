# Any type

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

[Thu Jun  5 02:10:35 PM CEST 2025]

This solution worked really well when all the dahl structures were using starpu_blocks under the hood to hold the data.
In following to the changes made on the [data structure wrappers](./data-structure-wrappers.md#Getting-the-right-types-using-starpu-builtin-filters),
it is no longer the case.
So for now, commons functions have a codelet definition for every type.

For example for relu I implemented that as:

```c
void relu(dahl_fp const* in, dahl_fp* out, size_t const start, size_t const end)
{
    for (size_t i = start; i < end; i++)
    {
        if (in[i] < 0.0F)
        {
            out[i] = 0.0F;
        }
    }
}

void block_relu(void* buffers[2], void* cl_arg)  { /* load data and call relu() */ }
void matrix_relu(void* buffers[2], void* cl_arg) { /* load data and call relu() */ }
void vector_relu(void* buffers[2], void* cl_arg) { /* load data and call relu() */ }
```

and we now have 3 task functions too:

```c
void task_block_relu(dahl_block const* in, dahl_block *out)
{
    int ret = starpu_task_insert(&cl_block_relu,
                                 STARPU_R, in->handle, 
                                 STARPU_W, out->handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_block_submit");
}

void task_matrix_relu(dahl_matrix const* in, dahl_matrix *out)
{
    int ret = starpu_task_insert(&cl_matrix_relu,
                                 STARPU_R, in->handle, 
                                 STARPU_W, out->handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_matrix_submit");
}

void task_vector_relu(dahl_vector const* in, dahl_vector *out)
{
    int ret = starpu_task_insert(&cl_vector_relu,
                                 STARPU_R, in->handle, 
                                 STARPU_W, out->handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_vector_submit");
}
```
These task functions could be generated, or we could switch for the types at runtime using the dahl_any types
but it makes no sense to do the following:
- Create a matrix
- wrap it in a `any` type
- pass it to `relu_any()`
- `relu_any` switch for the type and directs to the good codelet

If we know the type from the start why wrapping it then unwrapping it?
Indeed the API is a bit bigger but at least it makes sense.

Also we can still use macros and `_Generic` but here in a way cleaner way, and without needing any wrappers.
This is still very useful for the `SELF` versions some functions.

```c
#define TYPES_MATCH(T1, T2) \
    (__builtin_types_compatible_p(typeof(*(T1)), typeof(*(T2))))

#define TASK_RELU(IN, OUT)                                 \
    _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                   "IN and OUT must be of the same type"); \
    _Generic((OUT),                                        \
        dahl_block*: task_block_relu,                      \
        dahl_matrix*: task_matrix_relu,                    \
        dahl_vector*: task_vector_relu                     \
    )(IN, OUT)

#define TASK_RELU_SELF(SELF) TASK_RELU(SELF, SELF)
```

Here we verify that IN and OUT have the same types, we also ignore const qualifiers with a little trick in TYPES_MATCH.
And here we switch the function based on the type of OUT because it will never be const qualified as it is the output buffer.
So it reduces the number of branches we need.

Implementing the SELF version of the function is straightforward.

Honestly the macros are way cleaner this way!

The advantage of having a codelet by each type is that we can identify which codelet type take more time to run for example.
(not convinced by my own argument? Note that nntile does that but with C++ template so they don't write it x times, whatever)

Also we can define different implementation for the different types (may be useful if we can use BLAS on matrix and vectors for example, coz I'm not sure
it exists equivalent for blocks).

If we really want to merge down the implementations together however, we could use transform every data passed in the arguments as a vector,
then we only have to implement the codelet for vectors.
This can be done with manual partitionning ([see](./data-structure-wrappers.md#getting-the-right-types-with-manual-partitionning)) but this is kind of a hassle.
Or we could implement a custom filter that return a single vector of the whole data structure -> but it means we have to do a data partition for just a single vector... Not very pretty.

[Mon Jun 30 10:58:41 AM CEST 2025]

So currently the API for common tasks looked like this: 

```
Generic arg          dahl type    handle + cl   starpu type   float array
              ┌──► task_block_fn  ──┐       ┌──► block_fn  ──┐ 
TASK MACRO FN ┼──► task_matrix_fn ──┼► task ┼──► matrix_fn ──┼► relu
              └──► task_vector_fn ──┘       └──► vector_fn ──┘ 
```

This does not make a lot of sense because we end up writing a macro that redirects to the correct task launcher function, that itself redirect to a common task launcher function, then to either of the codelets for each type, then to a common function implementation...

The problem here is that we were forced to implement 3 different codelets because we are no longer using the same data type block as a super data type like we did before.
To resolve that I created a new macro:

```c
#define STARPU_ANY_GET_PTR(interface) (((struct starpu_vector_interface *)(interface))->ptr)
```

This works for every type of StarPU because the `ptr` field is always in the third position, so it does not matter which interface we cast our object into.
Maybe this could create problems though... But for now it seems alright.

Plus in the codelet we need to pass the number of elements in the object:

```c
void relu(void* buffers[2], void* cl_arg)
{
    size_t nb_elem;
    starpu_codelet_unpack_args(cl_arg, &nb_elem);

    dahl_fp const* in = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[0]);
    dahl_fp* out = (dahl_fp*)STARPU_ANY_GET_PTR(buffers[1]);

    for (size_t i = 0; i < nb_elem; i++)
    {
        if (in[i] < 0.0F)
        {
            out[i] = 0.0F;
        }
    }
}
```

In the common tasks, every data types will be accessed elements by elements.


Now we only have to refactor the task launcher functions.
I like the idea of a single macro to launch a task on any type, one idea would be to pass the handle with the macro but this exposes the `starpu_handle_t` into my public API.

Instead I used a trait mechanism:

```c
// Defining the trait type, that hold function pointers to common functions of the data types.
typedef const struct _dahl_traits {
    starpu_data_handle_t (*get_handle)(void const*);
    size_t (*get_nb_elem)(void const*);
} dahl_traits;
```

```c
// Here we instanciate each type and choose the correct function pointers
// This makes it super easy to add a new type
dahl_traits dahl_traits_tensor = {
    .get_handle = _tensor_get_handle,
    .get_nb_elem = _tensor_get_nb_elem,
};

dahl_traits dahl_traits_block = {
    .get_handle = _block_get_handle,
    .get_nb_elem = _block_get_nb_elem,
};

dahl_traits dahl_traits_matrix = {
    .get_handle = _matrix_get_handle,
    .get_nb_elem = _matrix_get_nb_elem,
};

dahl_traits dahl_traits_vector = {
    .get_handle = _vector_get_handle,
    .get_nb_elem = _vector_get_nb_elem,
};
```

```c
// This trait can be passed to tasks common to every types
void task_relu(void const* in, void* out, dahl_traits* traits)
{
    // Get the number of elements
    size_t nb_elem = traits->get_nb_elem(out);
    int ret = starpu_task_insert(&cl_relu,
                                 // Pass the number of elements to the codelet
                                 STARPU_VALUE, &nb_elem, sizeof(&nb_elem),
                                           // Get the handles
                                 STARPU_R, traits->get_handle(in), 
                                 STARPU_W, traits->get_handle(out), 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_matrix_submit");
}
```

Finally, this gives us even simpler macros:

```c
// First define once the generic to match the corresponding trait object
#define GET_TRAITS(OBJECT) _Generic((OBJECT), \
    dahl_tensor*: &dahl_traits_tensor,        \
    dahl_block*:  &dahl_traits_block,         \
    dahl_matrix*: &dahl_traits_matrix,        \
    dahl_vector*: &dahl_traits_vector         \
)

// Define a types match static assert
#define TYPES_MATCH(T1, T2) \
    (__builtin_types_compatible_p(typeof(*(T1)), typeof(*(T2))))

// Actually define the relu macro
#define TASK_RELU(IN, OUT)                                     \
    do {                                                       \
        _Static_assert(TYPES_MATCH((IN), (OUT)),               \
                       "IN and OUT must be of the same type"); \
        task_relu(IN, OUT, GET_TRAITS(OUT));                   \
    } while (0)

// Update every value of `self` by applying a relu function.
#define TASK_RELU_SELF(SELF) TASK_RELU(SELF, SELF)
```
