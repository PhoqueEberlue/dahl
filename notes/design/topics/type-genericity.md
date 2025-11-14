# Type generecity

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

[Fri Nov 14 08:00:08 AM UTC 2025]

Should have clearly inlined + restrict the pointers to have a more fair benchmark.
