#include "../include/dahl_asserts.h"
#include <assert.h>
#include <stdio.h>

void assert_vector_equals(dahl_vector const* a, dahl_vector const* b, bool const rounding)
{
    if (!vector_equals(a, b, rounding))
    {
        printf("[DAHL][FAIL][%s:%d] Assert vector equals: A != B\n", __FILE__, __LINE__);
        printf("A: ");
        vector_print(a);
        printf("B: ");
        vector_print(b);
        printf("\n");
    }
}

void assert_matrix_equals(dahl_matrix const* a, dahl_matrix const* b, bool const rounding)
{
    if (!matrix_equals(a, b, rounding))
    {
        printf("[DAHL][FAIL][%s:%d] Assert matrix equals: A != B\n", __FILE__, __LINE__);
        printf("A: ");
        matrix_print(a);
        printf("B: ");
        matrix_print(b);
        printf("\n");
    }
}

void assert_block_equals(dahl_block const* a, dahl_block const* b, bool const rounding)
{
    if (!block_equals(a, b, rounding))
    {
        printf("[DAHL][FAIL][%s:%d] Assert block equals: A != B\n", __FILE__, __LINE__);
        printf("A: ");
        block_print(a);
        printf("B: ");
        block_print(b);
        printf("\n");
    }
}

void assert_fp_equals(dahl_fp const a, dahl_fp const b)
{
    if (a != b)
    {
        printf("[DAHL][FAIL][%s:%d] Assert fp equals: %f != %f\n\n", __FILE__, __LINE__, a, b);
    }
}

void assert_shape2d_equals(dahl_shape2d const a, dahl_shape2d const b)
{
    if(!shape2d_equals(a , b))
    {
        printf("[DAHL][FAIL][%s:%d] Assert shape2d equals: A != B\n", __FILE__, __LINE__);
        printf("A: ");
        shape2d_print(a);
        printf("B: ");
        shape2d_print(b);
        printf("\n");
    }
}

void assert_shape3d_equals(dahl_shape3d const a, dahl_shape3d const b)
{
    if(!shape3d_equals(a , b))
    {
        printf("[DAHL][FAIL][%s:%d] Assert shape3d equals: A != B\n", __FILE__, __LINE__);
        printf("A: ");
        shape3d_print(a);
        printf("B: ");
        shape3d_print(b);
        printf("\n");
    }
}
