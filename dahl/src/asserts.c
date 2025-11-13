#include "../include/dahl_asserts.h"
#include <assert.h>
#include <stdio.h>
#include "misc.h"
#include "stdlib.h"
#include "data_structures/data_structures.h"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_COLOR_FLUSH   "\33[0m"

void log_prefix_fail(char const* file, int const line, char const* function)
{
    printf("[DAHL][TESTS][" ANSI_COLOR_RED "FAIL" ANSI_COLOR_FLUSH "][%s:%s:%d] ", file, function, line);
}

void log_prefix_ok(char const* file, int const line, char const* function)
{
    printf("[DAHL][TESTS][" ANSI_COLOR_GREEN "OK" ANSI_COLOR_FLUSH "][%s]\n", function);
}

void print_diff(void const* a, void const* b, char const* a_expr, char const* b_expr, int8_t precision, dahl_traits* traits)
{
    char fname1[256];
    char fname2[256];

    snprintf(fname1, sizeof(fname1), "/tmp/%s_XXXXXX", a_expr);
    snprintf(fname2, sizeof(fname2), "/tmp/%s_XXXXXX", b_expr);

    FILE *fp1 = temp_file_create(fname1);
    FILE *fp2 = temp_file_create(fname2);

    // Set default precision when precision is equal to the special value -1
    if (precision == -1) { precision = DAHL_DEFAULT_PRINT_PRECISION ; }

    traits->print_file(a, fp1, precision);
    traits->print_file(b, fp2, precision);

    fflush(fp1);
    fflush(fp2);
    rewind(fp1);
    rewind(fp2);

    char cmd[256];
    // TODO: we could clearly implement that in C and directly integrate it into the code.
    // Compare outputs with a custom python script
    snprintf(cmd, sizeof(cmd), "python3 ../diff.py '%s' '%s'", fname1, fname2);
    system(cmd);

    temp_file_delete(fname1, fp1);
    temp_file_delete(fname2, fp2);
}

void assert_scalar_equals(dahl_scalar const* a, dahl_scalar const* b,
                          bool const rounding, int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (!scalar_equals(a, b, rounding, precision))
    {
        log_prefix_fail(file, line, function);
        printf("Assert scalar equals: " ANSI_COLOR_GREEN "%s" ANSI_COLOR_FLUSH " != " ANSI_COLOR_RED "%s" ANSI_COLOR_FLUSH "\n", a_expr, b_expr);
        PRINT_DIFF_EXPR(a, b, a_expr, b_expr, precision);
        printf("\n");
    }
    else
    {
        log_prefix_ok(file, line, function);
    }
}

void assert_vector_equals(dahl_vector const* a, dahl_vector const* b,
                          bool const rounding, int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (!vector_equals(a, b, rounding, precision))
    {
        log_prefix_fail(file, line, function);
        printf("Assert vector equals: " ANSI_COLOR_GREEN "%s" ANSI_COLOR_FLUSH " != " ANSI_COLOR_RED "%s" ANSI_COLOR_FLUSH "\n", a_expr, b_expr);
        PRINT_DIFF_EXPR(a, b, a_expr, b_expr, precision);
        printf("\n");
    }
    else
    {
        log_prefix_ok(file, line, function);
    }
}

void assert_matrix_equals(dahl_matrix const* a, dahl_matrix const* b,
                          bool const rounding, int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (!matrix_equals(a, b, rounding, precision))
    {
        log_prefix_fail(file, line, function);
        printf("Assert matrix equals: " ANSI_COLOR_GREEN "%s" ANSI_COLOR_FLUSH " != " ANSI_COLOR_RED "%s" ANSI_COLOR_FLUSH "\n", a_expr, b_expr);
        PRINT_DIFF_EXPR(a, b, a_expr, b_expr, precision);
        printf("\n");
    }
    else
    {
        log_prefix_ok(file, line, function);
    }
}

void assert_block_equals(dahl_block const* a, dahl_block const* b,
                          bool const rounding, int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (!block_equals(a, b, rounding, precision))
    {
        log_prefix_fail(file, line, function);
        printf("Assert block equals: " ANSI_COLOR_GREEN "%s" ANSI_COLOR_FLUSH " != " ANSI_COLOR_RED "%s" ANSI_COLOR_FLUSH "\n", a_expr, b_expr);
        PRINT_DIFF_EXPR(a, b, a_expr, b_expr, precision);
        printf("\n");
    }
    else
    {
        log_prefix_ok(file, line, function);
    }
}

void assert_tensor_equals(dahl_tensor const* a, dahl_tensor const* b,
                          bool const rounding, int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (!tensor_equals(a, b, rounding, precision))
    {
        log_prefix_fail(file, line, function);
        printf("Assert tensor equals: " ANSI_COLOR_GREEN "%s" ANSI_COLOR_FLUSH " != " ANSI_COLOR_RED "%s" ANSI_COLOR_FLUSH "\n", a_expr, b_expr);
        PRINT_DIFF_EXPR(a, b, a_expr, b_expr, precision);
        printf("\n");
    }
    else
    {
        log_prefix_ok(file, line, function);
    }
}

void assert_fp_equals(dahl_fp const a, dahl_fp const b,
                          bool const rounding, int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    bool match = false;

    if (rounding)
    {
        match = fp_round(a, precision) == fp_round(b, precision);
    }
    else
    {
        match = a == b;
    }
    
    if (!match)
    {
        log_prefix_fail(file, line, function);
        printf("Assert dahl_fp equals: %s != %s\n", a_expr, b_expr);
        printf("%s = %+.15f\n", a_expr, a);
        printf("%s = %+.15f\n", b_expr, b);
        printf("\n");
    }
    else
    {
        log_prefix_ok(file, line, function);
    }
}

void assert_size_t_equals(size_t const a, size_t const b,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr)
{
    if (a != b)
    {
        log_prefix_fail(file, line, function);
        printf("Assert size_t equals: %s != %s\n", a_expr, b_expr);
        printf("%s = %zu\n", a_expr, a);
        printf("%s = %zu\n", b_expr, b);
        printf("\n");
    }
    else
    {
        log_prefix_ok(file, line, function);
    }
}

void assert_shape2d_equals(dahl_shape2d const a, dahl_shape2d const b,
                           char const* file, int const line, char const* function,
                           char const* a_expr, char const* b_expr)
{
    if(!shape2d_equals(a , b))
    {
        log_prefix_fail(file, line, function);
        printf("Assert shape2d equals: %s != %s\n", a_expr, b_expr);
        printf("%s: ", a_expr);
        shape2d_print(a);
        printf("%s: ", b_expr);
        shape2d_print(b);
        printf("\n");
    }
    else
    {
        log_prefix_ok(file, line, function);
    }
}

void assert_shape3d_equals(dahl_shape3d const a, dahl_shape3d const b,
                           char const* file, int const line, char const* function,
                           char const* a_expr, char const* b_expr)
{
    if(!shape3d_equals(a , b))
    {
        log_prefix_fail(file, line, function);
        printf("Assert shape3d equals: %s != %s\n", a_expr, b_expr);
        printf("%s: ", a_expr);
        shape3d_print(a);
        printf("%s: ", b_expr);
        shape3d_print(b);
        printf("\n");
    }
    else
    {
        log_prefix_ok(file, line, function);
    }
}

void assert_shape4d_equals(dahl_shape4d const a, dahl_shape4d const b,
                           char const* file, int const line, char const* function,
                           char const* a_expr, char const* b_expr)
{
    if(!shape4d_equals(a , b))
    {
        log_prefix_fail(file, line, function);
        printf("Assert shape3d equals: %s != %s\n", a_expr, b_expr);
        printf("%s: ", a_expr);
        shape4d_print(a);
        printf("%s: ", b_expr);
        shape4d_print(b);
        printf("\n");
    }
    else
    {
        log_prefix_ok(file, line, function);
    }
}
