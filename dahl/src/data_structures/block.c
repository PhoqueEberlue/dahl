#include "data_structures.h"
#include "custom_filters.h"
#include "../misc.h"
#include "starpu_data.h"
#include "sys/types.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../tasks/codelets.h"
#include <jpeglib.h>

// Allocate a dahl_block structure from a handle, and a pointer to data.
void* _block_init_from_ptr(dahl_arena* arena, starpu_data_handle_t handle, dahl_fp* data)
{
    dahl_block* block = dahl_arena_alloc(arena, sizeof(dahl_block));
    block->handle = handle;
    block->data = data;
    block->origin_arena = arena;
    block->is_redux = false;
    block->partition = (dahl_partition**)dahl_arena_alloc(arena, sizeof(dahl_partition**));

    return block;
}

// Allocate enough space for the given `shape` into the `arena`.
dahl_fp* _block_data_alloc(dahl_arena* arena, dahl_shape3d const shape)
{
    size_t n_elems = shape.x * shape.y * shape.z;
    return dahl_arena_alloc(arena, n_elems * sizeof(dahl_fp));
}

// Registers some `data` array to starpu, returning a handle with block type and correct dimensions.
starpu_data_handle_t _block_data_register(
        dahl_arena* arena, dahl_shape3d const shape, dahl_fp* data)
{
    starpu_data_handle_t handle = nullptr;
    starpu_block_data_register(
        &handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        shape.x,
        shape.x*shape.y,
        shape.x,
        shape.y,
        shape.z,
        sizeof(dahl_fp)
    );

    dahl_arena_attach_handle(arena, handle); 
    return handle;
}

dahl_block* block_init(dahl_arena* arena, dahl_shape3d const shape)
{
    dahl_fp* data = _block_data_alloc(arena, shape);
    memset(data, 0, shape.x*shape.y*shape.z * sizeof(dahl_fp));
    starpu_data_handle_t handle = _block_data_register(arena, shape, data);
    dahl_block* block = _block_init_from_ptr(arena, handle, data);
    return block;
}

dahl_block* block_init_redux(dahl_arena* arena, dahl_shape3d const shape)
{
    dahl_block* block = block_init(arena, shape);
    // Enable redux mode
    _block_enable_redux(block);
    return block;
}

dahl_block* block_init_from(dahl_arena* arena, dahl_shape3d const shape, dahl_fp const* data)
{
    dahl_fp* block_data = _block_data_alloc(arena, shape);

    for (size_t i = 0; i < shape.x*shape.y*shape.z; i++)
         block_data[i] = data[i];

    starpu_data_handle_t handle = _block_data_register(arena, shape, block_data);
    return _block_init_from_ptr(arena, handle, block_data);
}

dahl_block* block_init_random(dahl_arena* arena, dahl_shape3d const shape, dahl_fp min, dahl_fp max)
{
    dahl_fp* block_data = _block_data_alloc(arena, shape);

    for (size_t z = 0; z < shape.z; z++)
    {
        for (size_t y = 0; y < shape.y; y++)
        {
            for (size_t x = 0; x < shape.x; x++)
            {
                size_t index = (z * shape.x * shape.y) + (y * shape.x) + x;
                block_data[index] = fp_rand(min, max);
            }
        }
    }

    starpu_data_handle_t handle = _block_data_register(arena, shape, block_data);
    return _block_init_from_ptr(arena, handle, block_data);
}

void _block_enable_redux(void* block)
{
    ((dahl_block*)block)->is_redux = true;
    starpu_data_set_reduction_methods(
            ((dahl_block*)block)->handle, &cl_block_accumulate, &cl_block_zero);
}

bool _block_get_is_redux(void const* block)
{
    return ((dahl_block const*)block)->is_redux;
}

void block_set_from(dahl_block* block, dahl_fp const* data)
{
    dahl_shape3d shape = block_get_shape(block);

    block_acquire(block);

    size_t i = 0;
    for (size_t z = 0; z < shape.z; z ++)
    {
        for (size_t y = 0; y < shape.y; y ++)
        {
            for (size_t x = 0; x < shape.x; x ++)
            {
                block_set_value(block, x, y, z, data[i]);
                i++;
            }
        }
    }

    block_release(block);
}

void block_read_jpeg(dahl_block* block, char const* filename)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Could not open %s\n", filename);
        return;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    size_t width    = cinfo.output_width;
    size_t height   = cinfo.output_height;
    size_t channels = cinfo.output_components;

    dahl_shape3d shape = block_get_shape(block);
    assert(width == shape.x);
    assert(height == shape.y);
    assert(channels == shape.z);

    size_t row_stride = width * channels;

    // Allocate buffer for a single scanline
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)(
        (j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1
    );

    // Iterate rows
    for (size_t y = 0; y < height; y++)
    {
        jpeg_read_scanlines(&cinfo, buffer, 1); // reads one row

        unsigned char *row = buffer[0];

        // Iterate pixels and channels
        for (size_t x = 0; x < width; x++)
        {
            for (size_t z = 0; z < channels; z++)
            {
                unsigned char tmp = row[(x * channels) + z];

                dahl_fp value = (dahl_fp)tmp / 255.0F;

                block_set_value(block, x, y, z, value);
            }
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);
}

inline dahl_fp block_get_value(dahl_block const* block, size_t x, size_t y, size_t z)
{
    size_t ldy = starpu_block_get_local_ldy(block->handle);
    size_t ldz = starpu_block_get_local_ldz(block->handle);
    return block->data[(z * ldz) + (y * ldy) + x];
}

inline void block_set_value(dahl_block* block, size_t x, size_t y, size_t z, dahl_fp value)
{
    size_t ldy = starpu_block_get_local_ldy(block->handle);
    size_t ldz = starpu_block_get_local_ldz(block->handle);
    block->data[(z * ldz) + (y * ldy) + x] = value;
}

dahl_shape3d block_get_shape(dahl_block const* block)
{
    size_t nx = starpu_block_get_nx(block->handle);
	size_t ny = starpu_block_get_ny(block->handle);
	size_t nz = starpu_block_get_nz(block->handle);
    dahl_shape3d res = { .x = nx, .y = ny, .z = nz };

    return res;
}

starpu_data_handle_t _block_get_handle(void const* block)
{
    return ((dahl_block*)block)->handle;
}

size_t _block_get_nb_elem(void const* block)
{
    dahl_shape3d shape = block_get_shape((dahl_block*)block);
    return shape.x * shape.y * shape.z;
}

void block_acquire(dahl_block const* block)
{
    starpu_data_acquire(block->handle, STARPU_R);
}

void block_acquire_mut(dahl_block* block)
{
    starpu_data_acquire(block->handle, STARPU_RW);
}

void block_release(dahl_block const* block)
{
    starpu_data_release(block->handle);
}

dahl_vector* block_flatten_no_copy(dahl_block const* block)
{
    dahl_shape3d shape = block_get_shape(block);
    size_t new_len = shape.x * shape.y * shape.z;

    // Registers our block data as a vector (with new shape), handle will be attached to the
    // block's origin arena.
    starpu_data_handle_t handle = _vector_data_register(
            block->origin_arena, new_len, block->data);
    
    dahl_vector* res = _vector_init_from_ptr(block->origin_arena, handle, block->data);

    // Here we use the same trick when doing manual partitioning:
    // Use cl_switch to force data refresh in our new handle from the block handle
	int ret = starpu_task_insert(&cl_switch, STARPU_RW, block->handle, STARPU_W, handle, 0);
	STARPU_CHECK_RETURN_VALUE(ret, "block_flatten_along_t_no_copy");

    // Then deactivate the block handle
    starpu_data_invalidate_submit(block->handle);

    return res;
}

bool block_equals(dahl_block const* a, dahl_block const* b, bool const rounding, int8_t const precision)
{
    dahl_shape3d shape_a = block_get_shape(a);
    dahl_shape3d shape_b = block_get_shape(b);

    assert(shape_a.x == shape_b.x 
        && shape_a.y == shape_b.y 
        && shape_a.z == shape_b.z);

    block_acquire(a);
    block_acquire(b);

    bool res = true;

    for (size_t z = 0; z < shape_a.z; z++)
    {
        for (size_t y = 0; y < shape_a.y; y++)
        {
            for (size_t x = 0; x < shape_a.x; x++)
            {
                dahl_fp a_val = block_get_value(a, x, y, z);
                dahl_fp b_val = block_get_value(b, x, y, z);

                if (rounding) { res = fp_equals_round(a_val, b_val, precision); }
                else          { res = fp_equals(a_val, b_val);                  }

                if (!res)     { goto RELEASE; }
            }
        }
    }

RELEASE:
    block_release(a);
    block_release(b);

    return res;
}

dahl_partition* _block_get_partition(void const* block)
{
    return *((dahl_block*)block)->partition;
}

void block_partition_along_z(dahl_block const* block, dahl_access access)
{
    size_t const nparts = block_get_shape(block).z;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_pick_matrix_z,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_matrix_child_ops
	};

    // Create and set the partition
    dahl_partition* p = _partition_init(nparts, access, &dahl_traits_matrix,
                                        &f, block->handle, block->origin_arena, 
                                        BLOCK_PARTITION_ALONG_Z);

    _partition_submit(p);
    *block->partition = p;
}

void block_partition_along_z_flat_matrices(dahl_block const* block, dahl_access access, bool is_row)
{
    size_t const nparts = block_get_shape(block).z;

    struct starpu_data_filter f =
	{
		.filter_func = is_row? starpu_block_filter_pick_matrix_z_as_row_matrix 
                             : starpu_block_filter_pick_matrix_z_as_col_matrix,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_matrix_child_ops
	};

    // Create and set the partition
    dahl_partition* p = _partition_init(nparts, access, &dahl_traits_matrix,
                                        &f, block->handle, block->origin_arena,
                                        BLOCK_PARTITION_ALONG_Z_FLAT_MATRICES);

    _partition_submit(p);
    *block->partition = p;
}

void block_partition_along_z_flat_vectors(dahl_block const* block, dahl_access access)
{
    size_t const nparts = block_get_shape(block).z;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_pick_matrix_z_as_flat_vector,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_vector_child_ops
	};

    // Create and set the partition
    dahl_partition* p = _partition_init(nparts, access, &dahl_traits_vector,
                                        &f, block->handle, block->origin_arena,
                                        BLOCK_PARTITION_ALONG_Z_FLAT_VECTORS);

    _partition_submit(p);
    *block->partition = p;
}

void block_partition_flatten_to_vector(dahl_block const* block, dahl_access access)
{
    // Only one vector here because we flatten the whole block into a vector
    size_t const nparts = 1;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_flatten_to_vector,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_vector_child_ops
	};

    // Create and set the partition
    dahl_partition* p = _partition_init(nparts, access, &dahl_traits_vector,
                                        &f, block->handle, block->origin_arena,
                                        BLOCK_PARTITION_FLATTEN_TO_VECTOR);

    _partition_submit(p);
    *block->partition = p;
}

void block_partition_along_z_batch(dahl_block const* block, dahl_access access, size_t batch_size)
{
    size_t const nparts = block_get_shape(block).z / batch_size;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_depth_block,
		.nchildren = nparts,
	};

    // Create and set the partition
    dahl_partition* p = _partition_init(nparts, access, &dahl_traits_block,
                                        &f, block->handle, block->origin_arena,
                                        BLOCK_PARTITION_ALONG_Z_BATCH);

    _partition_submit(p);
    *block->partition = p;
}

void block_unpartition(dahl_block_part const* block)
{
    dahl_partition* p = *block->partition;
    assert(p && p->is_active);
    _unpartition_submit(p);
}

void _block_print_file(void const* vblock, FILE* fp, int8_t const precision)
{
    auto block = (dahl_block const*)vblock;
    const dahl_shape3d shape = block_get_shape(block);
	const size_t ldy = starpu_block_get_local_ldy(block->handle);
	const size_t ldz = starpu_block_get_local_ldz(block->handle);

	block_acquire(block);

    fprintf(fp, "block=%p nx=%zu ny=%zu nz=%zu ldy=%zu ldz=%zu\n{\n", block->data, shape.x, shape.y, shape.z, ldy, ldz);
	for(size_t z = 0; z < shape.z; z++)
	{
        fprintf(fp, "\t{\n");
		for(size_t y = 0; y < shape.y; y++)
		{
            fprintf(fp, "\t\t{ ");

			for(size_t x = 0; x < shape.x; x++)
			{
				fprintf(fp, "%+.*f, ", precision, block_get_value(block, x, y, z));
			}
			fprintf(fp, "},\n");
		}
		fprintf(fp, "\t},\n");
	}
	fprintf(fp, "}\n");

	block_release(block);
}

void block_print(dahl_block const* block)
{
    _block_print_file(block, stdout, DAHL_DEFAULT_PRINT_PRECISION);
}

void block_image_display(dahl_block const* block, size_t const scale_factor)
{
    // TODO: use the min max function to make it easier, however how do we deal with the allocations?
    // We absolutely do not want to have an arena as a parameter, so maybe create a temp arena here?
    // This is not a big deal in term of code blocking, because we will be acquiring the vector anyways.
    // however here we'll be creating an arena for nothing. Maybe using a global "trash" arena like we did before?
    // not sure it makes sense though.
    // dahl_fp min = scalar_get_value(TASK_MIN())...
    
    dahl_shape3d shape = block_get_shape(block);
    block_acquire(block);

    char cmd[100] = {};
    // Create a command that uses ImageMagick to display our matrix into an image
    sprintf(cmd, "display -resize %lux%lu -", shape.x * scale_factor, shape.y * scale_factor);

    FILE *fp = popen(cmd, "w");
    // Use PPM format with P6 for RGB
    fprintf(fp, "P6\n%d %d\n255\n", (int)shape.x, (int)shape.y);

    // normalize to 0..255
    dahl_fp min = block_get_value(block, 0, 0, 0);
    dahl_fp max = min;
    for (size_t y = 0; y < shape.y; y++)
    {
        for (size_t x = 0; x < shape.x; x++)
        {
            for (size_t z = 0; z < shape.z; z++)
            {
                dahl_fp value = block_get_value(block, x, y, z);
                if (value < min) min = value;
                if (value > max) max = value;
            }
        }
    }
    dahl_fp range = max - min;

    for (size_t y = 0; y < shape.y; y++)
    {
        for (size_t x = 0; x < shape.x; x++)
        {
            for (size_t z = 0; z < shape.z; z++)
            {
                unsigned char val = (unsigned char)(255.0F * (block_get_value(block, x, y, z) - min) / (range+1e-8F));
                fwrite(&val, 1, 1, fp);
            }
        }
    }

    pclose(fp);
    block_release(block);
}
