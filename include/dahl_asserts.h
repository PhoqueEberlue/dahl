#ifndef DAHL_ASSERTS_H
#define DAHL_ASSERTS_H

#include "dahl_data.h"

void assert_fp_equals(dahl_fp const a, dahl_fp const b);
void assert_vector_equals(dahl_vector const* a, dahl_vector const* b, bool const rounding);
void assert_matrix_equals(dahl_matrix const* a, dahl_matrix const* b, bool const rounding);
void assert_block_equals(dahl_block const* a, dahl_block const* b, bool const rounding);

#endif //!DAHL_H
