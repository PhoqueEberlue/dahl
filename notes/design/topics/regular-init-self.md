# Regular Init Self

Simplify and/or homogeneize the way tasks returns values.
For example, let `add` that takes a and b parameters and write a + b in c
This function can be written as:
```c
// 1: Return the result via the pointer c
void add(void* a, void* b, void* c);

// 2: Simply return the result with the return keyword
void* add(void* a, void* b);

// 3: Writes the result directly in buffer a
void add_self(void* a_self, void* b);
```
All of those solutions are correct but they serve different purposes in terms of memory management.
1. forces the user to instanciate c on its own, which may reduce the possibility to forget calling finalize to free the memory.
Also it is very useful when another buffer can be reused to store the result into c. For example we might perform operations on
sub matrices of a block, and in this case we want to store every sub result in a common block buffer.
2. makes less writing for the user, the returned objected can be instanciated with the correct dimensions by the function.
Yet the user shouldn't forget to free the memory.
3. Also very useful when directly writing the result to the buffer a is not a problem.
For now, most of the tasks implement syntax 1 and 3 
- Should I implement the three ways for every task? 
- Will it cost a lot to maintain or add too much complexity?
- find a nice way to differentiate the 3 functions and add it in the doc

[sam. 12 avril 2025 10:34:19 CEST]
-> This would be very interesting not to allocate the same buffers over and over again: maybe try to reuse the buffers?
This is why it is important to have functions that take buffers pointers.
[jeu. 17 avril 2025 10:47:48 CEST]
-> "", "self", and "init" are good names to differentiate those three modes
[jeu. 24 avril 2025 18:27:47 CEST]
After all I'm not sure "init" should be added, because the function return already indicates that something is initialized right?
The problem is that in the task api and in the data api we might have diverging conventions.
