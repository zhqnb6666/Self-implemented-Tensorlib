
#include "tensor.h"
#include "tensor.cpp"
#include <iostream>

void test_tensor() {
    // Test the default constructor
    Tensor<int> tensor1;
    tensor1.print();

    // Test the constructor with dimensions and values
    Tensor<int> tensor2({2, 3}, {1, 2, 3, 4, 5, 6});
    std::cout << tensor2;

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
    for (size_t dim: dims) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // Test the values function
    std::vector<int> values = tensor7.values();
    std::cout << "Values: ";
    for (int value: values) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Test the operator() function
    std::cout << "Element at index {1, 2}: " << tensor7[{1, 2}] << std::endl;

    // Test the data_ptr function
std::cout << "Data pointer: " << tensor7.data_ptr() << std::endl;

// Test the type function
    std::cout << "Type: " << tensor7.type() << std::endl;
}

void test_rand() {
    // Test the rand function with double type
    Tensor<double>* tensor_double = Tensor<double>::rand({2, 3});
    std::cout << "Double tensor:" << std::endl;
    tensor_double->print();
    delete tensor_double;

    // Test the rand function with int type
    Tensor<int>* tensor_int = Tensor<int>::rand({2, 3});
    std::cout << "Int tensor:" << std::endl;
    tensor_int->print();
    delete tensor_int;

    // Test the rand function with bool type
    Tensor<bool>* tensor_bool = Tensor<bool>::rand({2, 3});
    std::cout << "Bool tensor:" << std::endl;
    tensor_bool->print();
    delete tensor_bool;
}

void test_operator() {
    // Create a Tensor object
    Tensor<int> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    std::cout << "Original Tensor:" << std::endl;
    tensor.print();

    // Use the overloaded operator() function to get a subset of the Tensor
    Tensor<int>* subset1 = tensor(2);
    std::cout << "Subset1 Tensor:" << std::endl;
    subset1->print();

    // Use the overloaded operator() function to get a subset of the Tensor
    Tensor<int>* subset2 = tensor(1, {0, 1});
    std::cout << "Subset2 Tensor:" << std::endl;
    subset2->print();

    // Change an element in the subset
    (*subset1)[{0}] = 100;
    std::cout << "Subset Tensor after change:" << std::endl;
    subset1->print();
    std::cout << "Original Tensor after subset change:" << std::endl;
    tensor.print();

    delete subset1;
    delete subset2;
}

void test_cat() {
    // Create two tensors
    std::vector<int> data1 = {1, 2, 3, 4};
    std::vector<int> data2 = {7, 8, 9, 10};
    Tensor<int> t1({2,2}, data1);
    Tensor<int> t2({2, 2}, data2);

    // Concatenate the tensors along the first dimension
    Tensor<int> *t3 = Tensor<int>::cat({t1, t2}, 1);
    std::cout << "Concatenated tensor along the first dimension:" << std::endl;
    t3->print();
    delete t3;
}

void test_tile() {
    // Create a tensor
    std::vector<int> data = {1, 2, 3, 4};
    Tensor<int> t({2, 2}, data);

    // Print the original tensor
    std::cout << "Original tensor:" << std::endl;
    t.print();

    // Tile the tensor along the first dimension
    Tensor<int> *tiled = Tensor<int>::tile(t, 1, 3);

    // Print the tiled tensor
    std::cout << "Tiled tensor:" << std::endl;
    tiled->print();

    delete tiled;
}

void test_mutate() {
    // Create a tensor
    std::vector<int> data = {1, 2, 3, 4, 1,1,1,1};
    Tensor<int> t({2, 4}, data);

    // Print the original tensor
    std::cout << "Original tensor:" << std::endl;
    t.print();
    std::cout << "t(1):" << std::endl;
    t(1)->print();
    std::cout<<"Mutated tensor:"<<std::endl;
    *t(1) = 2;
    t.print();

   Tensor<int> t2({2, 4}, data);
    std::cout<<"t2:"<<std::endl;
    t2.print();
    // Use the overloaded assignment operator to set all elements to a new array
    *t2(1,{0,4}) = {5,6,7,8};
    // Print the modified tensor
    std::cout << "Modified tensor:" << std::endl;
    t2.print();
}


int main() {
    test_tensor();
    test_rand();
    test_operator();
    test_cat();
    test_tile();
    test_mutate();
    return 0;
}
