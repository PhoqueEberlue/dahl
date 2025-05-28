#ifndef DAHL_ASSERTS_H
#define DAHL_ASSERTS_H

#include "dahl_data.h"

void assert_size_t_equals(size_t const a, size_t const b, 
                          char const* file, int line,
                          char const* a_expr, char const* b_expr);

void assert_shape2d_equals(dahl_shape2d const a, dahl_shape2d const b,
                           char const* file, int const line,
                           char const* a_expr, char const* b_expr);

void assert_shape3d_equals(dahl_shape3d const a, dahl_shape3d const b,
                           char const* file, int const line,
                           char const* a_expr, char const* b_expr);


void assert_fp_equals(dahl_fp const a, dahl_fp const b,
                          char const* file, int const line,
                          char const* a_expr, char const* b_expr);

void assert_vector_equals(dahl_vector const* a, dahl_vector const* b, bool const rounding,
                          char const* file, int const line,
                          char const* a_expr, char const* b_expr);

void assert_matrix_equals(dahl_matrix const* a, dahl_matrix const* b, bool const rounding,
                          char const* file, int const line,
                          char const* a_expr, char const* b_expr);

void assert_block_equals(dahl_block const* a, dahl_block const* b, bool const rounding,
                         char const* file, int const line,
                         char const* a_expr, char const* b_expr);

#define ASSERT_SIZE_T_EQUALS(a, b) assert_size_t_equals((a), (b), __FILE__, __LINE__, #a, #b)
#define ASSERT_SHAPE2D_EQUALS(a, b) assert_shape2d_equals((a), (b), __FILE__, __LINE__, #a, #b)
#define ASSERT_SHAPE3D_EQUALS(a, b) assert_shape3d_equals((a), (b), __FILE__, __LINE__, #a, #b)
#define ASSERT_FP_EQUALS(a, b) assert_fp_equals((a), (b), __FILE__, __LINE__, #a, #b)
#define ASSERT_VECTOR_EQUALS(a, b, rounding) assert_vector_equals((a), (b), (rounding), __FILE__, __LINE__, #a, #b)
#define ASSERT_MATRIX_EQUALS(a, b, rounding) assert_matrix_equals((a), (b), (rounding), __FILE__, __LINE__, #a, #b)
#define ASSERT_BLOCK_EQUALS(a, b, rounding) assert_block_equals((a), (b), (rounding), __FILE__, __LINE__, #a, #b)

#endif //!DAHL_H
