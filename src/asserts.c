#include "../include/dahl_asserts.h"
#include <assert.h>
#include <stdio.h>

void assert_vector_equals(dahl_vector const* a, dahl_vector const* b, bool const rounding,
                          char const* file, int const line,
                          char const* a_expr, char const* b_expr)
{
    if (!vector_equals(a, b, rounding))
    {
        printf("[DAHL][FAIL][%s:%d] Assert vector equals: %s != %s\n", file, line, a_expr, b_expr);
        printf("%s: ", a_expr);
        vector_print(a);
        printf("%s: ", b_expr);
        vector_print(b);
        printf("\n");
    }
}

void assert_matrix_equals(dahl_matrix const* a, dahl_matrix const* b, bool const rounding,
                          char const* file, int const line,
                          char const* a_expr, char const* b_expr)
{
    if (!matrix_equals(a, b, rounding))
    {
        printf("[DAHL][FAIL][%s:%d] Assert matrix equals: %s != %s\n", file, line, a_expr, b_expr);
        printf("%s: ", a_expr);
        matrix_print(a);
        printf("%s: ", b_expr);
        matrix_print(b);
        printf("\n");
    }
}

void assert_block_equals(dahl_block const* a, dahl_block const* b, bool const rounding,
                          char const* file, int const line,
                          char const* a_expr, char const* b_expr)
{
    if (!block_equals(a, b, rounding))
    {
        printf("[DAHL][FAIL][%s:%d] Assert block equals: %s != %s\n", file, line, a_expr, b_expr);
        printf("%s: ", a_expr);
        block_print(a);
        printf("%s: ", b_expr);
        block_print(b);
        printf("\n");
    }
}

void assert_fp_equals(dahl_fp const a, dahl_fp const b,
                          char const* file, int const line,
                          char const* a_expr, char const* b_expr)
{
    if (a != b)
    {
        printf("[DAHL][FAIL][%s:%d] Assert dahl_fp equals: %s != %s\n", file, line, a_expr, b_expr);
        printf("%s = %f\n", a_expr, a);
        printf("%s = %f\n", b_expr, b);
        printf("\n");
    }
}

void assert_size_t_equals(size_t const a, size_t const b,
                          char const* file, int const line,
                          char const* a_expr, char const* b_expr)
{
    if (a != b)
    {
        printf("[DAHL][FAIL][%s:%d] Assert size_t equals: %s != %s\n", file, line, a_expr, b_expr);
        printf("%s = %zu\n", a_expr, a);
        printf("%s = %zu\n", b_expr, b);
        printf("\n");
    }
}

void assert_shape2d_equals(dahl_shape2d const a, dahl_shape2d const b,
                           char const* file, int const line,
                           char const* a_expr, char const* b_expr)
{
    if(!shape2d_equals(a , b))
    {
        printf("[DAHL][FAIL][%s:%d] Assert shape2d equals: %s != %s\n", file, line, a_expr, b_expr);
        printf("%s: ", a_expr);
        shape2d_print(a);
        printf("%s: ", b_expr);
        shape2d_print(b);
        printf("\n");
    }
}

void assert_shape3d_equals(dahl_shape3d const a, dahl_shape3d const b,
                           char const* file, int const line,
                           char const* a_expr, char const* b_expr)
{
    if(!shape3d_equals(a , b))
    {
        printf("[DAHL][FAIL][%s:%d] Assert shape3d equals: %s != %s\n", file, line, a_expr, b_expr);
        printf("%s: ", a_expr);
        shape3d_print(a);
        printf("%s: ", b_expr);
        shape3d_print(b);
        printf("\n");
    }
}
