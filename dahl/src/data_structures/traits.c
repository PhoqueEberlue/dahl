#include "data_structures.h"

// Linking traits with their corresponding function definition
dahl_traits dahl_traits_tensor = {
    .init_from_ptr = _tensor_init_from_ptr,
    .get_handle = _tensor_get_handle,
    .get_nb_elem = _tensor_get_nb_elem,
    .print_file = _tensor_print_file,
    .get_is_redux = _tensor_get_is_redux,
    .enable_redux = _tensor_enable_redux,
    .get_partition = _tensor_get_partition,
    .type = DAHL_TENSOR,
};

dahl_traits dahl_traits_block = {
    .init_from_ptr = _block_init_from_ptr,
    .get_handle = _block_get_handle,
    .get_nb_elem = _block_get_nb_elem,
    .print_file = _block_print_file,
    .get_is_redux = _block_get_is_redux,
    .enable_redux = _block_enable_redux,
    .get_partition = _block_get_partition,
    .type = DAHL_BLOCK,
};

dahl_traits dahl_traits_matrix = {
    .init_from_ptr = _matrix_init_from_ptr,
    .get_handle = _matrix_get_handle,
    .get_nb_elem = _matrix_get_nb_elem,
    .print_file = _matrix_print_file,
    .get_is_redux = _matrix_get_is_redux,
    .enable_redux = _matrix_enable_redux,
    .get_partition = _matrix_get_partition,
    .type = DAHL_MATRIX,
};

dahl_traits dahl_traits_vector = {
    .init_from_ptr = _vector_init_from_ptr,
    .get_handle = _vector_get_handle,
    .get_nb_elem = _vector_get_nb_elem,
    .print_file = _vector_print_file,
    .get_is_redux = _vector_get_is_redux,
    .enable_redux = _vector_enable_redux,
    .get_partition = nullptr,
    .type = DAHL_VECTOR,
};

// FIX: Nullptr on the functions here make it though to debug...
dahl_traits dahl_traits_scalar = {
    .init_from_ptr = nullptr, // No function for scalar
    .get_handle = _scalar_get_handle,
    .get_nb_elem = _scalar_get_nb_elem, // Always 1
    .print_file = _scalar_print_file,
    .get_is_redux = _scalar_get_is_redux,
    .enable_redux = _scalar_enable_redux,
    .get_partition = nullptr,
    .type = DAHL_SCALAR,
};

void print_handle(void const* object, char const* name, dahl_traits* traits)
{
    printf("Handle for %s: %p\n", name, traits->get_handle(object));
}
