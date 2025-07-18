#include "../include/dahl_asserts.h"
#include <assert.h>
#include <stdio.h>

void log_prefix(char const* file, int const line, char const* function)
{
    printf("[DAHL][FAIL][%s:%s:%d] ", file, function, line);
}

void assert_scalar_equals(dahl_scalar const* a, dahl_scalar const* b,
                          bool const rounding, u_int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (!scalar_equals(a, b, rounding, precision))
    {
        log_prefix(file, line, function);
        printf("Assert scalar equals: %s != %s\n", a_expr, b_expr);
        printf("%s: ", a_expr);
        scalar_print(a);
        printf("%s: ", b_expr);
        scalar_print(b);
        printf("\n");
    }
}

void assert_vector_equals(dahl_vector const* a, dahl_vector const* b,
                          bool const rounding, u_int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (!vector_equals(a, b, rounding, precision))
    {
        log_prefix(file, line, function);
        printf("Assert vector equals: %s != %s\n", a_expr, b_expr);
        printf("%s: ", a_expr);
        vector_print(a);
        printf("%s: ", b_expr);
        vector_print(b);
        printf("\n");
    }
}

void assert_matrix_equals(dahl_matrix const* a, dahl_matrix const* b,
                          bool const rounding, u_int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (!matrix_equals(a, b, rounding, precision))
    {
        log_prefix(file, line, function);
        printf("Assert matrix equals: %s != %s\n", a_expr, b_expr);
        printf("%s: ", a_expr);
        matrix_print(a);
        printf("%s: ", b_expr);
        matrix_print(b);
        printf("\n");
    }
}

void assert_block_equals(dahl_block const* a, dahl_block const* b,
                          bool const rounding, u_int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (!block_equals(a, b, rounding, precision))
    {
        log_prefix(file, line, function);
        printf("Assert block equals: %s != %s\n", a_expr, b_expr);
        printf("%s: ", a_expr);
        block_print(a);
        printf("%s: ", b_expr);
        block_print(b);
        printf("\n");
    }
}

void assert_tensor_equals(dahl_tensor const* a, dahl_tensor const* b,
                          bool const rounding, u_int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (!tensor_equals(a, b, rounding, precision))
    {
        log_prefix(file, line, function);
        printf("Assert block equals: %s != %s\n", a_expr, b_expr);
        printf("%s: ", a_expr);
        tensor_print(a);
        printf("%s: ", b_expr);
        tensor_print(b);
        printf("\n");
    }
}

void assert_fp_equals(dahl_fp const a, dahl_fp const b,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (a != b)
    {
        log_prefix(file, line, function);
        printf("Assert dahl_fp equals: %s != %s\n", a_expr, b_expr);
        printf("%s = %f\n", a_expr, a);
        printf("%s = %f\n", b_expr, b);
        printf("\n");
    }
}

void assert_size_t_equals(size_t const a, size_t const b,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (a != b)
    {
        log_prefix(file, line, function);
        printf("Assert size_t equals: %s != %s\n", a_expr, b_expr);
        printf("%s = %zu\n", a_expr, a);
        printf("%s = %zu\n", b_expr, b);
        printf("\n");
    }
}

void assert_shape2d_equals(dahl_shape2d const a, dahl_shape2d const b,
                           char const* file, int const line, char const* function,
                           char const* a_expr, char const* b_expr)
{
    if(!shape2d_equals(a , b))
    {
        log_prefix(file, line, function);
        printf("Assert shape2d equals: %s != %s\n", a_expr, b_expr);
        printf("%s: ", a_expr);
        shape2d_print(a);
        printf("%s: ", b_expr);
        shape2d_print(b);
        printf("\n");
    }
}

void assert_shape3d_equals(dahl_shape3d const a, dahl_shape3d const b,
                           char const* file, int const line, char const* function,
                           char const* a_expr, char const* b_expr)
{
    if(!shape3d_equals(a , b))
    {
        log_prefix(file, line, function);
        printf("Assert shape3d equals: %s != %s\n", a_expr, b_expr);
        printf("%s: ", a_expr);
        shape3d_print(a);
        printf("%s: ", b_expr);
        shape3d_print(b);
        printf("\n");
    }
}

void assert_shape4d_equals(dahl_shape4d const a, dahl_shape4d const b,
                           char const* file, int const line, char const* function,
                           char const* a_expr, char const* b_expr)
{
    if(!shape4d_equals(a , b))
    {
        log_prefix(file, line, function);
        printf("Assert shape3d equals: %s != %s\n", a_expr, b_expr);
        printf("%s: ", a_expr);
        shape4d_print(a);
        printf("%s: ", b_expr);
        shape4d_print(b);
        printf("\n");
    }
}
