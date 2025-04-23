#include "../include/dahl_asserts.h"
#include <stdio.h>

void assert_vector_equals(dahl_vector const* const a, dahl_vector const* const b, bool const rounding)
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

void assert_matrix_equals(dahl_matrix const* const a, dahl_matrix const* const b, bool const rounding)
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

void assert_block_equals(dahl_block const* const a, dahl_block const* const b, bool const rounding)
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
