mkdir cifar-10
cd cifar-10
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xf cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* .
rm -r cifar-10-batches-bin
