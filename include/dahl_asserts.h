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
                      bool rounding, u_int8_t precision,
                      char const* file, int line, char const* function,
                      char const* a_expr, char const* b_expr);

void assert_vector_equals(dahl_vector const* a, dahl_vector const* b,
                          bool rounding, u_int8_t precision,
                          char const* file, int line, char const* function,
                          char const* a_expr, char const* b_expr);

void assert_matrix_equals(dahl_matrix const* a, dahl_matrix const* b,
                          bool rounding, u_int8_t precision,
                          char const* file, int line, char const* function,
                          char const* a_expr, char const* b_expr);

void assert_block_equals(dahl_block const* a, dahl_block const* b,
                         bool rounding, u_int8_t precision,
                         char const* file, int line, char const* function,
                         char const* a_expr, char const* b_expr);

void assert_tensor_equals(dahl_tensor const* a, dahl_tensor const* b,
                         bool rounding, u_int8_t precision,
                         char const* file, int line, char const* function,
                         char const* a_expr, char const* b_expr);

#define ASSERT_SIZE_T_EQUALS(a, b) assert_size_t_equals((a), (b), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_SHAPE2D_EQUALS(a, b) assert_shape2d_equals((a), (b), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_SHAPE3D_EQUALS(a, b) assert_shape3d_equals((a), (b), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_SHAPE4D_EQUALS(a, b) assert_shape4d_equals((a), (b), __FILE__, __LINE__, __func__, #a, #b)

#define ASSERT_FP_EQUALS(a, b) assert_fp_equals((a), (b), false, 0, __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_FP_EQUALS_ROUND(a, b, precision) assert_fp_equals((a), (b), true, (precision), __FILE__, __LINE__, __func__, #a, #b)

#define ASSERT_SCALAR_EQUALS(a, b) assert_scalar_equals((a), (b), false, 0, __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_VECTOR_EQUALS(a, b) assert_vector_equals((a), (b), false, 0, __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_MATRIX_EQUALS(a, b) assert_matrix_equals((a), (b), false, 0, __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_BLOCK_EQUALS(a, b) assert_block_equals((a), (b), false, 0, __FILE__, __LINE__, __func__,  #a, #b)
#define ASSERT_TENSOR_EQUALS(a, b) assert_tensor_equals((a), (b), false, 0, __FILE__, __LINE__, __func__,  #a, #b)

#define ASSERT_VECTOR_EQUALS_ROUND(a, b, precision) assert_vector_equals((a), (b), true, (precision), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_MATRIX_EQUALS_ROUND(a, b, precision) assert_matrix_equals((a), (b), true, (precision), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_BLOCK_EQUALS_ROUND(a, b, precision) assert_block_equals((a), (b), true, (precision), __FILE__, __LINE__, __func__, #a, #b)
#define ASSERT_TENSOR_EQUALS_ROUND(a, b, precision) assert_tensor_equals((a), (b), true, (precision), __FILE__, __LINE__, __func__, #a, #b)

#endif //!DAHL_H
