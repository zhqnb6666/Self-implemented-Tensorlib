
#include "tensor.h"
#include "tensor.cpp"
#include <iostream>

void test_tensor() {
    // Test the default constructor
    Tensor<int> tensor1;
    tensor1.print();

    // Test the constructor with dimensions and values
    Tensor<int> tensor2({2,3}, {1, 2, 3, 4, 5, 6});
    std::cout << tensor2 ;

    // Test the constructor with dimensions and a default value
    Tensor<int> tensor3({2, 2, 3, 2}, 0);
    tensor3.print();

    // Test the copy constructor
    Tensor<int> tensor4(tensor2);
    tensor4.print();

    // Test the move constructor
    Tensor<int> tensor5(std::move(tensor4));
    tensor5.print();

    // Test the copy assignment operator
    Tensor<int> tensor6;
    tensor6 = tensor2;
    tensor6.print();

    // Test the move assignment operator
    Tensor<int> tensor7;
    tensor7 = std::move(tensor6);
    tensor7.print();

    // Test the order function
    std::cout << "Order: " << tensor7.order() << std::endl;

    // Test the size function
    std::cout << "Size: " << tensor7.size() << std::endl;

    // Test the dimensions function
    std::vector<size_t> dims = tensor7.dimensions();
    std::cout << "Dimensions: ";
    for (size_t dim : dims) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // Test the values function
    std::vector<int> values = tensor7.values();
    std::cout << "Values: ";
    for (int value : values) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Test the operator() function
    std::cout << "Element at index {1, 2}: " << tensor7({1, 2}) << std::endl;

    // Test the operator[] function
    std::cout << "Element at linear index 5: " << tensor7[5] << std::endl;
}

int main() {
    test_tensor();
    return 0;
}