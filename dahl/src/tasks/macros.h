#ifndef DAHL_MACROS_H
#define DAHL_MACROS_H

// ---------------------------------------- Custom macros ----------------------------------------
// Get the ptr of any StarPU data type. Does not perform any check.
// This works because ptr is always the second field in the struct for vector, matrix, block and tensor,
// so it does not matter what we cast `interface` into. 
// This may be risky though, especially if the field order changes...
#define STARPU_ANY_GET_PTR(interface) (((struct starpu_vector_interface *)(interface))->ptr)

#if defined(STARPU_HAVE_STATEMENT_EXPRESSIONS) && defined(STARPU_DEBUG)
#define STARPU_TENSOR_GET(interface) (                   \
	{                                                    \
		STARPU_TENSOR_CHECK(interface);                  \
		(*(struct starpu_tensor_interface const*)(interface)); \
	})
#define STARPU_BLOCK_GET(interface) (                    \
	{                                                    \
		STARPU_BLOCK_CHECK(interface);                   \
		(*(struct starpu_block_interface const*)(interface));  \
	})
#define STARPU_MATRIX_GET(interface) (                   \
	{                                                    \
		STARPU_MATRIX_CHECK(interface);                  \
		(*(struct starpu_matrix_interface const*)(interface)); \
	})
#define STARPU_VECTOR_GET(interface) (                   \
	{                                                    \
		STARPU_VECTOR_CHECK(interface);                  \
		(*(struct starpu_vector_interface const*)(interface)); \
	})
#else
#define STARPU_TENSOR_GET(interface) (*(struct starpu_tensor_interface const*)(interface))
#define STARPU_BLOCK_GET (interface) (*(struct starpu_block_interface  const*)(interface))
#define STARPU_MATRIX_GET(interface) (*(struct starpu_matrix_interface const*)(interface))
#define STARPU_VECTOR_GET(interface) (*(struct starpu_vector_interface const*)(interface))
#endif

#endif //!DAHL_MACROS_H
