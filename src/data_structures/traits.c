#include "data_structures.h"

// Linking traits with their corresponding function definition
dahl_traits dahl_traits_tensor = {
    .init_from_ptr = _tensor_init_from_ptr,
    .get_handle = _tensor_get_handle,
    .get_partition = _tensor_get_current_partition,
    .get_nb_elem = _tensor_get_nb_elem,
    .print_file = _tensor_print_file,
    .type = DAHL_TENSOR,
};

dahl_traits dahl_traits_block = {
    .init_from_ptr = _block_init_from_ptr,
    .get_handle = _block_get_handle,
    .get_partition = _block_get_current_partition,
    .get_nb_elem = _block_get_nb_elem,
    .print_file = _block_print_file,
    .type = DAHL_BLOCK,
};

dahl_traits dahl_traits_matrix = {
    .init_from_ptr = _matrix_init_from_ptr,
    .get_handle = _matrix_get_handle,
    .get_partition = _matrix_get_current_partition,
    .get_nb_elem = _matrix_get_nb_elem,
    .print_file = _matrix_print_file,
    .type = DAHL_MATRIX,
};

dahl_traits dahl_traits_vector = {
    .init_from_ptr = _vector_init_from_ptr,
    .get_handle = _vector_get_handle,
    .get_partition = nullptr, // Vectors cannot be partitioned
    .get_nb_elem = _vector_get_nb_elem,
    .print_file = _vector_print_file,
    .type = DAHL_VECTOR,
};

// FIX: Nullptr on the functions here make it though to debug...
dahl_traits dahl_traits_scalar = {
    .init_from_ptr = nullptr, // No function for scalar
    .get_handle = _scalar_get_handle,
    .get_partition = nullptr, // Scalars cannot be partitioned
    .get_nb_elem = nullptr, // No number 
    .print_file = _scalar_print_file,
    .type = DAHL_SCALAR,
};
