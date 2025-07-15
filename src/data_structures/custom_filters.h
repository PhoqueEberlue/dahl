// This file contains custom filters that are not available by default in StarPU
#ifndef DAHL_CUSTOM_FILTERS_H
#define DAHL_CUSTOM_FILTERS_H

#include "starpu_data_filters.h"

// ---------------------------------------- GETTERS ----------------------------------------
// To get a vector from a block
static struct starpu_data_interface_ops *starpu_block_filter_pick_vector_child_ops(
    STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_vector_ops;
}

// To get a matrix from a matrix
static struct starpu_data_interface_ops *starpu_matrix_filter_pick_matrix_child_ops(
    STARPU_ATTRIBUTE_UNUSED struct starpu_data_filter *f, STARPU_ATTRIBUTE_UNUSED unsigned child)
{
	return &starpu_interface_matrix_ops;
}

// ---------------------------------------- TENSOR ----------------------------------------

// ---------------------------------------- BLOCK ----------------------------------------
// Pick the matrices along z axis, but instanciate them vith the vector interface so they can be read as vectors
static void starpu_block_filter_pick_matrix_z_as_flat_vector(
    void* parent_interface, void* child_interface, 
    struct starpu_data_filter* f,
    unsigned id, unsigned nparts)
{
	struct starpu_block_interface* block_parent = (struct starpu_block_interface *) parent_interface;
	struct starpu_vector_interface* vector_child = (struct starpu_vector_interface *) child_interface;

	unsigned blocksize;

	size_t nx = block_parent->nx;
	size_t ny = block_parent->ny;
	size_t nz = block_parent->nz;
	
    blocksize = block_parent->ldz;

	size_t elemsize = block_parent->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG(nparts <= nz, "cannot get %u vectors", nparts);
	STARPU_ASSERT_MSG((chunk_pos + id) < nz, "the chosen vector should be in the block");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(block_parent->id == STARPU_BLOCK_INTERFACE_ID, "%s can only be applied on a block data", __func__);
	vector_child->id = STARPU_VECTOR_INTERFACE_ID;

    vector_child->nx = nx*ny;

	vector_child->elemsize = elemsize;
	vector_child->allocsize = vector_child->nx * elemsize;

	if (block_parent->dev_handle)
	{
		if (block_parent->ptr)
			vector_child->ptr = block_parent->ptr + offset;
		
		vector_child->dev_handle = block_parent->dev_handle;
		vector_child->offset = block_parent->offset + offset;
	}
}

static void starpu_block_filter_pick_matrix_z_as_matrix(
    void* parent_interface, void* child_interface, 
    struct starpu_data_filter* f,
    unsigned id, unsigned nparts, unsigned mode)
{
	struct starpu_block_interface* block_parent = (struct starpu_block_interface *) parent_interface;
	struct starpu_matrix_interface* matrix_child = (struct starpu_matrix_interface *) child_interface;

	unsigned blocksize;

	size_t nx = block_parent->nx;
	size_t ny = block_parent->ny;
	size_t nz = block_parent->nz;
	
    blocksize = block_parent->ldz;

	size_t elemsize = block_parent->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG(nparts <= nz, "cannot get %u matrices", nparts);
	STARPU_ASSERT_MSG((chunk_pos + id) < nz, "the chosen matrix should be in the block");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(block_parent->id == STARPU_BLOCK_INTERFACE_ID, "%s can only be applied on a block data", __func__);
	matrix_child->id = STARPU_MATRIX_INTERFACE_ID;

    switch (mode) 
    {
        case 0:
            matrix_child->nx = nx*ny;
            matrix_child->ny = 1;
            break;
        case 1:
            matrix_child->nx = 1;
            matrix_child->ny = nx*ny;
            break;
        default:
            STARPU_ASSERT_MSG(mode <= 1 && mode >= 0, "%s wrong mode used, please use 0=row, 1=col", __func__);
            break;
    }

	matrix_child->elemsize = elemsize;
	matrix_child->allocsize = matrix_child->nx * matrix_child->ny * elemsize;

	if (block_parent->dev_handle)
	{
		if (block_parent->ptr)
			matrix_child->ptr = block_parent->ptr + offset;
		
		matrix_child->dev_handle = block_parent->dev_handle;
		matrix_child->offset = block_parent->offset + offset;
	}
}

static void starpu_block_filter_pick_matrix_z_as_row_matrix(
    void* parent_interface, void* child_interface, 
    struct starpu_data_filter* f,
    unsigned id, unsigned nparts)
{
    starpu_block_filter_pick_matrix_z_as_matrix(parent_interface, child_interface, f, id, nparts, 0);
}

static void starpu_block_filter_pick_matrix_z_as_col_matrix(
    void* parent_interface, void* child_interface, 
    struct starpu_data_filter* f,
    unsigned id, unsigned nparts)
{
    starpu_block_filter_pick_matrix_z_as_matrix(parent_interface, child_interface, f, id, nparts, 1);
}

// Picks the full block as a single vector (flattened view of the block)
static void starpu_block_filter_flatten_to_vector(
    void* parent_interface, void* child_interface,
    struct starpu_data_filter* f,
    unsigned id, STARPU_ATTRIBUTE_UNUSED unsigned nparts)
{
	struct starpu_block_interface* block_parent = (struct starpu_block_interface *) parent_interface;
	struct starpu_vector_interface* vector_child = (struct starpu_vector_interface *) child_interface;

	unsigned blocksize;

	size_t nx = block_parent->nx;
	size_t ny = block_parent->ny;
	size_t nz = block_parent->nz;
	
    blocksize = block_parent->ldz;

	size_t elemsize = block_parent->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	STARPU_ASSERT_MSG((chunk_pos + id) < nz, "the chosen vector should be in the block");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(block_parent->id == STARPU_BLOCK_INTERFACE_ID, "%s can only be applied on a block data", __func__);
	vector_child->id = STARPU_VECTOR_INTERFACE_ID;

    // We return only one vector, so it's equal to the product of every dimension size
    vector_child->nx = nx*ny*nz;

	vector_child->elemsize = elemsize;
	vector_child->allocsize = vector_child->nx * elemsize;

	if (block_parent->dev_handle)
	{
		if (block_parent->ptr)
			vector_child->ptr = block_parent->ptr + offset;
		
		vector_child->dev_handle = block_parent->dev_handle;
		vector_child->offset = block_parent->offset + offset;
	}
}
// ---------------------------------------- MATRIX ----------------------------------------
// Pick matrices along y axis
static void starpu_matrix_filter_vertical_matrix(
    void* parent_interface, void* child_interface, 
    struct starpu_data_filter* f,
    unsigned id, unsigned nparts)
{
	struct starpu_matrix_interface* matrix_parent = (struct starpu_matrix_interface *) parent_interface;
	struct starpu_matrix_interface* matrix_child = (struct starpu_matrix_interface *) child_interface;

	unsigned blocksize;

	size_t nx = matrix_parent->nx;
	size_t ny = matrix_parent->ny;
	
    blocksize = matrix_parent->ld;

	size_t elemsize = matrix_parent->elemsize;

	size_t chunk_pos = (size_t)f->filter_arg_ptr;

	size_t new_ny = matrix_parent->ny / nparts;

	STARPU_ASSERT_MSG(nparts <= ny, "cannot get %u vectors", nparts);
	STARPU_ASSERT_MSG((chunk_pos + id) < ny, "the chosen sub matrix should be in the matrix");

	size_t offset = (chunk_pos + id) * blocksize * elemsize;

	STARPU_ASSERT_MSG(matrix_parent->id == STARPU_MATRIX_INTERFACE_ID, "%s can only be applied on a block data", __func__);
	matrix_child->id = STARPU_MATRIX_INTERFACE_ID;

    matrix_child->nx = nx;
    matrix_child->ny = new_ny;

	matrix_child->elemsize = elemsize;
	matrix_child->allocsize = matrix_child->nx * matrix_child->ny * elemsize;

	if (matrix_parent->dev_handle)
	{
		if (matrix_parent->ptr)
			matrix_child->ptr = matrix_parent->ptr + offset;
		
		matrix_child->dev_handle = matrix_parent->dev_handle;
		matrix_child->offset = matrix_parent->offset + offset;
	}
}

// ---------------------------------------- VECTOR ----------------------------------------
//
//
#endif //!DAHL_CUSTOM_FILTERS_H
