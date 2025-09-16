#include "data_structures.h"

dahl_scalar* scalar_init(dahl_arena* arena)
{
    dahl_scalar* scalar = dahl_arena_alloc(arena, sizeof(dahl_scalar));
    scalar->data = 0.0F;

    starpu_data_handle_t handle = nullptr;
    starpu_variable_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)&scalar->data,
        sizeof(dahl_fp)
    );

    scalar->handle = handle;

    dahl_arena_attach_handle(arena, handle);

    return scalar;
}

dahl_scalar* scalar_init_from(dahl_arena* arena, dahl_fp const value)
{
    dahl_scalar* res = scalar_init(arena);
    res->data = value;
    return res;
}

dahl_fp scalar_get_value(dahl_scalar const* scalar)
{
    starpu_data_acquire(scalar->handle, STARPU_R);
    dahl_fp res = scalar->data;
    starpu_data_release(scalar->handle);
    return res;
}

void scalar_set_value(dahl_scalar* scalar, dahl_fp value)
{
    starpu_data_acquire(scalar->handle, STARPU_R);
    scalar->data = value;
    starpu_data_release(scalar->handle);
}

starpu_data_handle_t _scalar_get_handle(void const* scalar)
{
    return ((dahl_scalar*)scalar)->handle;
}

inline size_t _scalar_get_nb_elem(__attribute__((unused))void const* scalar)
{
    return 1;
}

bool scalar_equals(dahl_scalar const* a, dahl_scalar const* b, bool const rounding, u_int8_t const precision)
{
    dahl_fp a_val = scalar_get_value(a);
    dahl_fp b_val = scalar_get_value(b);

    if (rounding) { return fp_equals_round(a_val, b_val, precision); }
    else          { return fp_equals(a_val, b_val);                  }
}

void _scalar_print_file(void const* vscalar, FILE* fp, u_int8_t const precision)
{
    dahl_fp value = scalar_get_value((dahl_scalar const*)vscalar);
    fprintf(fp, "scalar=%+.*f\n", precision, value);
}

void scalar_print(dahl_scalar const* scalar)
{
    _scalar_print_file(scalar, stdout, DAHL_DEFAULT_PRINT_PRECISION);
}
