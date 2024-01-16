#include "tensor.h"
#include "tensor.cpp"
#include "save_load.cpp"
#include <iostream>
#include "tensor_math.h"
#include "tensor_math.cpp"
#include "tensor_bool.h"
#include "tensor_einsum.h"

//#include "tensor_einsum.cpp"

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

void test_slice() {
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
    std::vector<int> data = {1, 2, 3, 4, 1, 1, 1, 1};
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

void test_view() {
    // Create a tensor
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor<int> t({2, 4}, data);

    // Print the original tensor
    std::cout << "Original tensor:" << std::endl;
    t.print();

    // Use the view member function to create a new tensor
    Tensor<int> t_view = t.view({4, 2});
    std::cout << "View tensor (member function):" << std::endl;
    t_view.print();

    // Use the view static function to create a new tensor
    Tensor<int>* t_view_static = Tensor<int>::view(t, {4, 2});
    std::cout << "View tensor (static function):" << std::endl;
    t_view_static->print();

    // Change an element in the view tensor
    t_view[{0, 0}] = 100;
    std::cout << "View tensor after change (member function):" << std::endl;
    t_view.print();
    std::cout << "Original tensor after view tensor change (member function):" << std::endl;
    t.print();

    // Change an element in the view tensor (static function)
    (*t_view_static)[{0, 0}] = 200;
    std::cout << "View tensor after change (static function):" << std::endl;
    t_view_static->print();
    std::cout << "Original tensor after view tensor change (static function):" << std::endl;
    t.print();

    delete t_view_static;
}

void test_permute() {
    // Create a tensor
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor<int> t({2, 2, 2}, data);

    // Print the original tensor
    std::cout << "Original tensor:" << std::endl;
    t.print();

    // Use the permute member function to create a new tensor
    Tensor<int> t_permute = t.permute({1, 0, 2});
    std::cout << "Permuted tensor (member function):" << std::endl;
    t_permute.print();

    // Use the permute static function to create a new tensor
    Tensor<int>* t_permute_static = Tensor<int>::permute(t, {1, 0, 2});
    std::cout << "Permuted tensor (static function):" << std::endl;
    t_permute_static->print();

    // Change an element in the permuted tensor
    t_permute[{0, 0, 0}] = 100;
    std::cout << "Permuted tensor after change (member function):" << std::endl;
    t_permute.print();
    std::cout << "Original tensor after permuted tensor change (member function):" << std::endl;
    t.print();

    // Change an element in the permuted tensor (static function)
    (*t_permute_static)[{0, 0, 0}] = 200;
    std::cout << "Permuted tensor after change (static function):" << std::endl;
    t_permute_static->print();
    std::cout << "Original tensor after permuted tensor change (static function):" << std::endl;
    t.print();
}

void test_transpose() {
    // Create a tensor
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor<int> t({2, 2, 2}, data);

    // Print the original tensor
    std::cout << "Original tensor:" << std::endl;
    t.print();

    // Use the transpose member function to create a new tensor
    Tensor<int> t_transpose = t.transpose(0, 1);
    std::cout << "Transposed tensor (member function):" << std::endl;
    t_transpose.print();

    // Use the transpose static function to create a new tensor
    Tensor<int>* t_transpose_static = Tensor<int>::transpose(t, 0, 1);
    std::cout << "Transposed tensor (static function):" << std::endl;
    t_transpose_static->print();

    // Change an element in the transposed tensor
    t_transpose[{0, 0, 0}] = 100;
    std::cout << "Transposed tensor after change (member function):" << std::endl;
    t_transpose.print();
    std::cout << "Original tensor after transposed tensor change (member function):" << std::endl;
    t.print();

    // Change an element in the transposed tensor (static function)
    (*t_transpose_static)[{0, 0, 0}] = 200;
    std::cout << "Transposed tensor after change (static function):" << std::endl;
    t_transpose_static->print();
    std::cout << "Original tensor after transposed tensor change (static function):" << std::endl;
    t.print();

    delete t_transpose_static;
}

void test_bool_Tensor() {
    // Create a bool tensor
    std::vector<size_t> dimensions = {2, 3};
    std::vector<bool> values = {true, false, true, false, true, false};
    Tensor<bool> tensor(dimensions, values);

    // Print the tensor
    tensor.print();

    // Test the overloaded [] operator
    std::vector<size_t> indices = {1, 2};
    std::cout << "Element at [1, 2]: " << (tensor[indices] ? "true" : "false") << std::endl;

    // Test the set method
    std::vector<size_t> set_indices = {0, 1};
    tensor.set(set_indices, true);
    std::cout << "After setting element at [0, 1] to true:" << std::endl;
    tensor.print();

    // Test the overloaded () operator
    Tensor<bool>* subTensor = tensor(1);
    std::cout << "Sub-tensor at index 1: " << std::endl;
    subTensor->print();
    delete subTensor;
}

void test_save_load() {
    // Create a tensor
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor<int> original_tensor({2, 2, 2}, data);

    // Save the tensor to a file
    std::string filename = "test_tensor.txt";
    save(original_tensor, filename);

    // Load the tensor from the file
    Tensor<int> loaded_tensor = load<int>(filename);

    // Print the original and loaded tensors
    std::cout << "Original tensor:" << std::endl;
    original_tensor.print();
    std::cout << "Loaded tensor:" << std::endl;
    loaded_tensor.print();
}

void test_einsum() {
    // Example tensors for testing
    Tensor<int> vector1({3}, {1, 2, 3}); // 1D vector
    Tensor<int> vector2({3}, {4, 5, 6}); // 1D vector
    Tensor<int> matrix1({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}); // 2D matrix
    Tensor<int> matrix2({3, 2}, {7, 8, 9, 10, 11, 12}); // 2D matrix
    Tensor<int> matrix3({2, 3}, {13, 14, 15, 16, 17, 18}); // 2D matrix
    Tensor<int> tensor1({2, 3, 2}, {1, 2, 3, 4, 5, 6 ,7 ,8 ,9 ,10 ,11 ,12}); // 3D tensor
    Tensor<int> tensor2({2, 3, 2}, {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}); // 3D tensor

    // Test Summation
    std::cout << "Summation Test: " << std::endl;
    auto sum_result = einsum("i->", vector1);
    std::cout << "Sum: " << sum_result << std::endl;

    // Test Element-wise Multiplication
    std::cout << "\nElement-wise Multiplication Test: " << std::endl;
    auto element_wise_result = einsum("i,i->i", vector1, vector2);
    element_wise_result.print();

    // Test Matrix Multiplication
    std::cout << "\nMatrix Multiplication Test: " << std::endl;
    auto matrix_mul_result = einsum("ij,jk->ik", matrix1, matrix2);
    matrix_mul_result.print();

    // Test Dot Product
    std::cout << "\nDot Product Test: " << std::endl;
    auto dot_product_result = einsum("i,i->", vector1, vector2);
    std::cout << "Dot product: " << dot_product_result << std::endl;

    // Test Outer Product
    std::cout << "\nOuter Product Test: " << std::endl;
    auto outer_product_result = einsum("i,j->ij", vector1, vector2);
    outer_product_result.print();

    // Test Transpose
    std::cout << "\nTranspose Test: " << std::endl;
    auto transpose_result = einsum("ijk->jki", tensor1);
    std::cout << transpose_result << std::endl;

    // Test Diagonal
    std::cout << "\nDiagonal Test: " << std::endl;
    auto diagonal_result = einsum("ii->i", matrix1);
    diagonal_result.print();

    // Test Trace
    std::cout << "\nTrace Test: " << std::endl;
    auto trace_result = einsum("ii", matrix1);
    std::cout << "Trace: " << trace_result << std::endl;

    // Test inner product
    std::cout << "\nInner Product Test: " << std::endl;
    auto inner_product_result = einsum("ijk,kjl->il",tensor1, tensor2);
    std::cout << "Inner product:\n ";
    inner_product_result.print();
}

void test_math(){
    std::cout<< "math test:"<<std::endl;
    Tensor<int> matrix1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<int> matrix2({3, 2}, {7, 8, 9, 10, 11, 12}); // 2D matrix
    Tensor<int> matrix3({2, 2}, {1,2,3,4}); // 2D matrix
    Tensor<int> matrix4({2, 2}, {1,0,0,1}); // 2D matrix
    Tensor<int> t1 = matrix1.subtract(matrix2);
    Tensor<int> t2 = max(matrix1,1);
    t1.print();
    t2.print();
}

int main() {
//    test_tensor();
//    test_rand();
//    test_slice();
//    test_cat();
//    test_tile();
//    test_mutate();
//    test_view();
//    test_permute();
//    test_transpose();
//    test_bool_Tensor();
//    test_save_load();
//    test_einsum();
    test_math();
    return 0;
}
