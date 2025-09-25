Use snake case for naming:

```c
void do_something();
int my_super_variable;

#define AN_AMAZING_MACRO() amazing_function() // macros are in caption
#define GLOBAL_VARIABLE 42
```

Brackets are always on a new line:

```c
void do_something()
{
    if (true)
    {

    }
    else if (false)
    {

    }
    else
    {

    }
}
```

Except when initializing:

```c
struct test
{
    int b;
    int c;
};

struct test t = { // here it's ok
    .b = 1,
    .c = 2
};
```

Qualifiers are written right-to-left. This is easier to read `const* const`.

Altought sometimes `constexpr` is sometimes written `static constexpr elem`. Should probably change that to stay consistent.

```c
int const* a; // `a` is a pointer to const int
int const* const b; // `b` is a const pointer to a const int
int* const c; // `c` is a const pointer to a int
int constexpr c;
```

When arguments are passed by values, const is only written in the source file, not the header.
I think this simplifies the header API, because the user doesn't care that an argument passed by value will be changed. 
However in the implemantation it can be useful to state that we won't change it.

```c
// Definition in the header file
void do_something(int a);

// Implementation in the source file
void do_something(int const a)
{
    // ...
}
```
