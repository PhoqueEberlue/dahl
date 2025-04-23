#ifndef DAHL_ASSERTS_H
#define DAHL_ASSERTS_H

#include "dahl_data.h"

void assert_vector_equals(dahl_vector const* const a, dahl_vector const* const b, bool const rounding);
void assert_matrix_equals(dahl_matrix const* const a, dahl_matrix const* const b, bool const rounding);
void assert_block_equals(dahl_block const* const a, dahl_block const* const b, bool const rounding);

#endif //!DAHL_H
