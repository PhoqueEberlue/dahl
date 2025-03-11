#include "starpu_data_interfaces.h"
#include "tests.h"

#include "../src/convolution.h"
#include "../src/utils.h"
#include <stdio.h>

void test_convolution()
{
    const shape2d input_shape = { .x = 9, .y = 12 };
    const size_t num_filters = 10;
    const size_t filter_size = 3;
    convolution conv = create_convolution(input_shape, filter_size, num_filters);

    assert(conv.filter_shape.x == filter_size);
    assert(conv.filter_shape.y == filter_size);
    assert(conv.filter_shape.z == num_filters);

    assert(conv.output_shape.x == 7); // 9 - 3 + 1
    assert(conv.output_shape.y == 10); // 12 - 3 + 1
    assert(conv.output_shape.z == num_filters);

    assert(conv.filters_handle != nullptr);
    assert(conv.biases_handle != nullptr);

    starpu_data_handle_t input_handle = matrix_init(input_shape);
    matrix_fill_random(input_handle);
    
    starpu_data_handle_t output_handle = forward_pass(conv, input_handle);

    matrix_print_from_handle(input_handle);
    block_print_from_handle(conv.filters_handle);
    block_print_from_handle(output_handle);

    size_t ldy = starpu_block_get_local_ldy(output_handle);
    size_t ldz = starpu_block_get_local_ldz(output_handle);

    starpu_data_acquire(output_handle, STARPU_R);

    dahl_fp* output = (dahl_fp*)starpu_block_get_local_ptr(output_handle);

    // take value at 0, 0, with filter 1 applied
    dahl_fp res = output[(1 * ldz) + (0 * ldy) + 0];
    assert(res == 74);

    // take value at 1, 0, with filter 1 applied
    res = output[(1 * ldz) + (0 * ldy) + 1];
    assert(res == 158);

    starpu_data_release(output_handle);


}
