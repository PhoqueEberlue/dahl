#include "data_structures.h"

starpu_data_handle_t any_get_handle(dahl_any const any)
{
    switch (any.type)
    {
        case dahl_type_block:
            return any.structure.block->handle;
        case dahl_type_matrix:
            return any.structure.matrix->handle;
        case dahl_type_vector:
            return any.structure.vector->handle;
    }
}
