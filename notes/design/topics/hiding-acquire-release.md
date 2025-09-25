# Hiding acquire release

Here my problem is that `vector_data_acquire()` is exposed in my public API, but other vector functions might also acquire the data
internally, and the user of the API could easily create a deadlock situation that may be hard to debug.
My idea would be to completely hide the acquires in the API itself.

One way would be to provide get(i) and set(i, value) functions and the locking calls could be hidden there, 
but it means locking and unlocking for each element access. I want a better way to loop through the elements and still having auto lock/unlock.

This macro solution does the job:
```c
#define acquire(vector, data, expr) \
    do\
    {\
        dahl_fp* (data) = vector_data_acquire(vector);\
        expr\
        vector_data_release(vector);\
    }\
    while (0)

void test_vector_acquire()
{
    dahl_fp data[4] = { 1.0F, 2.0F, 3.0F, 4.0F };
    dahl_vector* vector = vector_init_from(4, data);

    dahl_fp data_bis[4] = { 1.0F, 2.0F, 3.0F, 4.0F };
    dahl_vector* vector_bis = vector_init_from(4, data_bis);

    acquire(vector, data,
        acquire(vector_bis, data_bis,
            for (size_t i = 0; i < 4; i++) 
            {
                 data[i] += data_bis[i];
            }
        );
    );

    dahl_fp expect[4] = { 2.0F, 4.0F, 6.0F, 8.0F };
    dahl_vector* expect_vector = vector_init_from(4, expect);

    ASSERT_VECTOR_EQUALS(expect_vector, vector);
}
```
However `vector_data_acquire` should remain public as it is called by the macro.
Furthermore it is still possible to create a deadlock by calling `vector_get_len` inside the macro calls.
