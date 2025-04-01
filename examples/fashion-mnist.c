#include "../include/dahl.h"

int main()
{
    load_mnist("../fashion-mnist/train-images-idx3-ubyte", 
               "../fashion-mnist/train-labels-idx1-ubyte");

    return 0;
}
