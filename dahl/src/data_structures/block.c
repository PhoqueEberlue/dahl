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
    metadata* md = dahl_arena_alloc(
        arena,
        // Metadata struct itself
        sizeof(metadata) + 
        // + a flexible array big enough partition pointers to 
        // store all kinds of block partioning
        (BLOCK_NB_PARTITION_TYPE * sizeof(dahl_partition*)
    ));

    for (size_t i = 0; i < BLOCK_NB_PARTITION_TYPE; i++) { md->partitions[i] = nullptr; }

    md->current_partition = -1;
    md->origin_arena = arena; // Saves where the block have been allocated

    dahl_block* block = dahl_arena_alloc(arena, sizeof(dahl_block));
    block->handle = handle;
    block->data = data;
    block->meta = md;
    block->is_redux = false;

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
    for (size_t y = 0; y < height; y++) {

        jpeg_read_scanlines(&cinfo, buffer, 1); // reads one row

        unsigned char *row = buffer[0];

        // Iterate pixels and channels
        for (size_t x = 0; x < width; x++) {
            for (size_t z = 0; z < channels; z++) {

                unsigned char tmp = row[(x * channels) + z];

                // Normalize to 0..1 like your example
                dahl_fp value = (dahl_fp)tmp / 255.0F;

                block_set_value(block, x, y, z, value);
            }
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);
}

dahl_fp block_get_value(dahl_block const* block, size_t x, size_t y, size_t z)
{
    size_t ldy = starpu_block_get_local_ldy(block->handle);
    size_t ldz = starpu_block_get_local_ldz(block->handle);
    return block->data[(z * ldz) + (y * ldy) + x];
}

void block_set_value(dahl_block* block, size_t x, size_t y, size_t z, dahl_fp value)
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

dahl_partition* _block_get_current_partition(void const* block)
{
    metadata* m = ((dahl_block const*)block)->meta;
    assert(m-> current_partition >= 0 && 
           m->current_partition < BLOCK_NB_PARTITION_TYPE);

    assert(m->partitions[m->current_partition] != nullptr);
    return m->partitions[m->current_partition];
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

void block_partition_along_z(dahl_block const* block, dahl_access access)
{
    assert(block->meta->current_partition == -1);
    block_partition_type t = BLOCK_PARTITION_ALONG_Z;

    // If the partition already exists, no need to create it.
    if (block->meta->partitions[t] != nullptr)
        goto submit; 

    size_t const nparts = block_get_shape(block).z;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_pick_matrix_z,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_matrix_child_ops
	};

    // Create and set the partition
    block->meta->partitions[t] = _partition_init(nparts, access, &dahl_traits_matrix,
                                        &f, block->handle, block->meta->origin_arena);

submit:
    _partition_submit_if_needed(block->meta, t, access, block->handle);
}

void block_partition_along_z_flat_matrices(dahl_block const* block, dahl_access access, bool is_row)
{
    assert(block->meta->current_partition == -1);
    block_partition_type t = BLOCK_PARTITION_ALONG_Z_FLAT_MATRICES;

    // If the partition already exists, no need to create it.
    if (block->meta->partitions[t] != nullptr)
        goto submit;

    size_t const nparts = block_get_shape(block).z;

    struct starpu_data_filter f =
	{
		.filter_func = is_row? starpu_block_filter_pick_matrix_z_as_row_matrix 
                             : starpu_block_filter_pick_matrix_z_as_col_matrix,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_matrix_child_ops
	};

    // Create and set the partition
    block->meta->partitions[t] = _partition_init(nparts, access, &dahl_traits_matrix,
                                        &f, block->handle, block->meta->origin_arena);

submit:
    _partition_submit_if_needed(block->meta, t, access, block->handle);
}

void block_partition_along_z_flat_vectors(dahl_block const* block, dahl_access access)
{
    assert(block->meta->current_partition == -1);
    block_partition_type t = BLOCK_PARTITION_ALONG_Z_FLAT_VECTORS;

    // If the partition already exists, no need to create it.
    if (block->meta->partitions[t] != nullptr)
        goto submit;

    size_t const nparts = block_get_shape(block).z;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_pick_matrix_z_as_flat_vector,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_vector_child_ops
	};

    // Create and set the partition
    block->meta->partitions[t] = _partition_init(nparts, access, &dahl_traits_vector,
                                        &f, block->handle, block->meta->origin_arena);

submit:
    _partition_submit_if_needed(block->meta, t, access, block->handle);
}

void block_partition_flatten_to_vector(dahl_block const* block, dahl_access access)
{
    assert(block->meta->current_partition == -1);
    block_partition_type t = BLOCK_PARTITION_FLATTEN_TO_VECTOR;

    // If the partition already exists, no need to create it.
    if (block->meta->partitions[t] != nullptr)
        goto submit;

    // Only one vector here because we flatten the whole block into a vector
    size_t const nparts = 1;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_flatten_to_vector,
		.nchildren = nparts,
		.get_child_ops = starpu_block_filter_pick_vector_child_ops
	};

    // Create and set the partition
    block->meta->partitions[t] = _partition_init(nparts, access, &dahl_traits_vector,
                                        &f, block->handle, block->meta->origin_arena);

submit:
    _partition_submit_if_needed(block->meta, t, access, block->handle);

}

void block_partition_along_z_batch(dahl_block const* block, dahl_access access, size_t batch_size)
{
    assert(block->meta->current_partition == -1);
    block_partition_type t = BLOCK_PARTITION_ALONG_Z_BATCH;

    size_t const nparts = block_get_shape(block).z / batch_size;

    dahl_partition* p = block->meta->partitions[t];
    // If the partition already exists AND had the same batch size, no need to create it. 
    // FIX Warning, here the memory is lost if we create many partitions with different batch size
    if (p != nullptr && p->nb_children == nparts)
        goto submit;

    struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_depth_block,
		.nchildren = nparts,
	};

    // Create and set the partition
    block->meta->partitions[t] = _partition_init(nparts, access, &dahl_traits_block,
                                        &f, block->handle, block->meta->origin_arena);

submit:
    _partition_submit_if_needed(block->meta, t, access, block->handle);
}

void block_unpartition(dahl_block const* block)
{
    dahl_partition* p = block->meta->partitions[block->meta->current_partition];
    assert(p); // Shouldn't crash, an non-active partition is identified by the if bellow
    assert(block->meta->current_partition >= 0 && 
           block->meta->current_partition < BLOCK_NB_PARTITION_TYPE);

    block->meta->current_partition = -1;
    starpu_data_unpartition_submit(block->handle, p->nb_children,
                                   p->handles, STARPU_MAIN_RAM);
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
