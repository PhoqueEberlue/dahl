#ifndef DAHL_ASSERTS_H
#define DAHL_ASSERTS_H

#include "dahl_data.h"

void assert_size_t_equals(size_t a, size_t b, 
                          char const* file, int line, char const* function,
                          char const* a_expr, char const* b_expr);

void assert_shape2d_equals(dahl_shape2d a, dahl_shape2d b,
                           char const* file, int line, char const* function,
                           char const* a_expr, char const* b_expr);

void assert_shape3d_equals(dahl_shape3d a, dahl_shape3d b,
                           char const* file, int line, char const* function,
                           char const* a_expr, char const* b_expr);

void assert_shape4d_equals(dahl_shape4d a, dahl_shape4d b,
                           char const* file, int line, char const* function,
                           char const* a_expr, char const* b_expr);

void assert_fp_equals(dahl_fp a, dahl_fp b,
                      bool rounding, int8_t precision,
                      char const* file, int line, char const* function,
                      char const* a_expr, char const* b_expr);

void assert_scalar_equals(dahl_scalar const* a, dahl_scalar const* b,
                          bool const rounding, int8_t const precision,
                          char const* file, int const line, char const* function,
                          char const* a_expr, char const* b_expr);

void assert_vector_equals(dahl_vector const* a, dahl_vector const* b,
                          bool rounding, int8_t precision,
                          char const* file, int line, char const* function,
                          char const* a_expr, char const* b_expr);

void assert_matrix_equals(dahl_matrix const* a, dahl_matrix const* b,
                          bool rounding, int8_t precision,
                          char const* file, int line, char const* function,
                          char const* a_expr, char const* b_expr);

void assert_block_equals(dahl_block const* a, dahl_block const* b,
                         bool rounding, int8_t precision,
                         char const* file, int line, char const* function,
                         char const* a_expr, char const* b_expr);

void assert_tensor_equals(dahl_tensor const* a, dahl_tensor const* b,
                         bool rounding, int8_t precision,
                         char const* file, int line, char const* function,
                         char const* a_expr, char const* b_expr);

void print_diff(void const* a, void const* b, char const* a_expr, char const* b_expr, int8_t const precision, dahl_traits* traits);

#define ASSERT_SIZE_T_EQUALS(a, b) assert_size_t_equals((a), (b), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_SHAPE2D_EQUALS(a, b) assert_shape2d_equals((a), (b), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_SHAPE3D_EQUALS(a, b) assert_shape3d_equals((a), (b), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_SHAPE4D_EQUALS(a, b) assert_shape4d_equals((a), (b), __FILE__, __LINE__, __func__, #a, #b)

#define ASSERT_FP_EQUALS(a, b) assert_fp_equals((a), (b), false, -1, __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_FP_EQUALS_ROUND(a, b, precision) assert_fp_equals((a), (b), true, (precision), __FILE__, __LINE__, __func__, #a, #b)

#define ASSERT_SCALAR_EQUALS(a, b) assert_scalar_equals((a), (b), false, -1, __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_VECTOR_EQUALS(a, b) assert_vector_equals((a), (b), false, -1, __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_MATRIX_EQUALS(a, b) assert_matrix_equals((a), (b), false, -1, __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_BLOCK_EQUALS(a, b) assert_block_equals((a), (b), false, -1, __FILE__, __LINE__, __func__,  #a, #b)
#define ASSERT_TENSOR_EQUALS(a, b) assert_tensor_equals((a), (b), false, -1, __FILE__, __LINE__, __func__,  #a, #b)

#define ASSERT_SCALAR_EQUALS_ROUND(a, b, precision) assert_scalar_equals((a), (b), true, (precision), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_VECTOR_EQUALS_ROUND(a, b, precision) assert_vector_equals((a), (b), true, (precision), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_MATRIX_EQUALS_ROUND(a, b, precision) assert_matrix_equals((a), (b), true, (precision), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_BLOCK_EQUALS_ROUND(a, b, precision) assert_block_equals((a), (b), true, (precision), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_TENSOR_EQUALS_ROUND(a, b, precision) assert_tensor_equals((a), (b), true, (precision), __FILE__, __LINE__, __func__, #a, #b)

#define PRINT_DIFF(A, B, PRECISION)                         \
    do {                                                    \
        _Static_assert(TYPES_MATCH((A), (B)),               \
                       "A and B must be of the same type"); \
        print_diff(A, B, #A, #B, PRECISION, GET_TRAITS(A)); \
    } while (0)

#define PRINT_DIFF_EXPR(A, B, A_EXPR, B_EXPR, PRECISION)            \
    do {                                                            \
        _Static_assert(TYPES_MATCH((A), (B)),                       \
                       "A and B must be of the same type");         \
        print_diff(A, B, A_EXPR, B_EXPR, PRECISION, GET_TRAITS(A)); \
    } while (0)

#endif //!DAHL_H
