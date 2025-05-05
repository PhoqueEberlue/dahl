
- Building "flattened views": because data are contiguously stored for every types, we can see a block or a matrix as a vector without
    changing anything in strapu. TODO: finish that 
    [Update sam. 12 avril 2025 09:42:22 CEST]:
    In Starpu you can resize NX, NY, and LD but only for matrices and vectors.
    So instead I decided to create another handle for the vector and release the previous one, which means that we don't have a view
    but an onwed object instead. Still fine, we don't perform data copy.

-------------------------------------------------------------------------------
- Add dahl prefix to every public functions/macros
- need to rethink the `vector_to_matrix` etc. functions -> should it be a view of the data? (-> implies refcount) should it take ownernship of the data? (easier to manage but implies more clones). I think we might need both actually? => MAKE SURE TO ADD `INIT` POST FIX TO EVERY FUNCTION THAT INSTANCIATE DATA THAT NEED TO BE FREED (And homogeneize init_from, init, clone etc. etc.)

- Add consts to unmodified buffers in the layers.
- FORGOT TO ADD BIASES IN THE CONVOLUTION FORWARD

- Add compilation condition to enable/disable debugg asserts
- Building common functions (that are not tasks) for the data structures, probably using getter functions taking dahl_any and performing a switch inside
- check memory leaks, don't forget to call the finalize() at some point :)
- Improve asserts to show context messages when something crashes -> change the strings of the STARPU_CHECK_RETURN_VALUE()
-  currently it is possible to call dahl_any macro with different types of data structures (e.g. TASK_ADD(vec, mat, block) ) -> avoid that
- less important but print always the same numbers of character in pretty print e.g. "42.00", " 8.00"...
- Should filter values be negative?
- test backward pass
- cmake if macos ignore cuda

- Investigate on that: if I partition a starpu block, and send tasks with the sub blocks on the GPU, does StarPU copies the whole block on the GPU, or
  every single sub block one by one?

- Hide starpu_wait_for_all into my API -> is this even needed though?
