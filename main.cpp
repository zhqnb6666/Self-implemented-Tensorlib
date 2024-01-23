#include "tensor.h"
#include "tensor.cpp"
#include "save_load.cpp"
#include <iostream>
#include "tensor_math.h"
#include "tensor_math.cpp"
#include "tensor_bool.h"
#include "tensor_einsum.h"
#include "gradient.h"

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

void test_diagonal() {
    std::vector<int> diagonal_values = {1, 2, 3};
    Tensor<int>* diagonal_matrix = Tensor<int>::diagonal(diagonal_values);
    diagonal_matrix->print();
    delete diagonal_matrix;
}


void test_rand() {
    std::cout << "Test the rand function:" << std::endl;
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

void test_index() {
    // Create a Tensor object
    Tensor<int> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    std::cout << "Original Tensor:" << std::endl;
    tensor.print();

    // Use the overloaded operator() function to get a subset of the Tensor
    Tensor<int>* subset1 = tensor(2);
    std::cout << "Subset1 Tensor:" << std::endl;
    subset1->print();

    // Change an element in the subset
    (*subset1)[{0}] = 100;
    std::cout << "Subset Tensor after change:" << std::endl;
    subset1->print();
    std::cout << "Original Tensor after subset change:" << std::endl;
    tensor.print();

    delete subset1;
}

void test_slice() {
    // Create a Tensor object
    Tensor<int> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    std::cout << "Original Tensor:" << std::endl;
    tensor.print();

    // Use the overloaded operator() function to get a subset of the Tensor
    Tensor<int>* subset2 = tensor(1, {0, 1});
    std::cout << "Subset2 Tensor:" << std::endl;
    subset2->print();

    // Change an element in the subset
    (*subset2)[{0}] = 100;
    std::cout << "Subset Tensor after change:" << std::endl;
    subset2->print();
    std::cout << "Original Tensor after subset change:" << std::endl;
    tensor.print();

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

    // Get a reference to an element in the tensor
    Tensor<int>* element = t(1);
    std::cout << "Element before mutation:" << std::endl;
    element->print();

    // Mutate the element
    *element = 2;
    std::cout << "Element after mutation:" << std::endl;
    element->print();

    // Print the tensor after mutation
    std::cout << "Tensor after mutation:" << std::endl;
    t.print();

    // Create another tensor
    Tensor<int> t2({2, 4}, data);
    std::cout << "Original tensor2:" << std::endl;
    t2.print();

    // Get a slice of the tensor
    Tensor<int>* slice = t2(1, {0, 4});
    std::cout << "Slice before mutation:" << std::endl;
    slice->print();

    // Mutate the slice
    *slice = {5, 6, 7, 8};
    std::cout << "Slice after mutation:" << std::endl;
    slice->print();

    // Print the tensor after mutating the slice
    std::cout << "Tensor2 after slice mutation:" << std::endl;
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
    Tensor<int> matrix4({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}); // 2D matrix
    Tensor<int> tensor1({2, 3, 2}, {1, 2, 3, 4, 5, 6 ,7 ,8 ,9 ,10 ,11 ,12}); // 3D tensor
    Tensor<int> tensor2({2, 3, 2}, {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}); // 3D tensor
    Tensor<int>* tensor3 = Tensor<int>::rand({2, 3, 4, 5}); // "pqrs"
    Tensor<int>* tensor4 = Tensor<int>::rand({2, 2, 3, 2, 4}); //"tuqvr"
    Tensor<int> tensor5({3,3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor<int> tensor6({3,3,3}, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                  1, 2, 3, 4, 5, 6, 7, 8, 9,
                                  1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor<int> tensor7({3,3},{1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor<int> tensor8({2,2,3},{1,2,3,4,5,6,7,8,9,10,11,12});

    // Test Diagonal
    std::cout << "\nDiagonal Test: " << std::endl;
    auto diagonal_result = einsum("ii->i", matrix1);
    diagonal_result.print();

    // Test Transpose
    std::cout << "\nTranspose Test: " << std::endl;
    auto transpose_result = einsum("ijk->jki", tensor1);
    std::cout << transpose_result << std::endl;

    // Permute Test
    std::cout << "\nPermute Test: " << std::endl;
    auto permute_result = einsum("...ij->...ji", tensor1);
    std::cout << permute_result << std::endl;

    // Test Reduce sum
    std::cout << "Reduce Sum Test: " << std::endl;
    auto sum_result = einsum("ij->", matrix1);
    std::cout << "Reduce Sum: " << sum_result << std::endl;

    // Test Sum along dimension
    std::cout << "\nSum along dimension Test: " << std::endl;
    auto sum_along_dim_result = einsum("ij->j", matrix1);
    sum_along_dim_result.print();

    // Test Matrix and vector mul
    std::cout << "\nMatrix and vector mul Test: " << std::endl;
    auto matrix_vector_mul_result = einsum("ij,j->i", matrix1, vector1);
    matrix_vector_mul_result.print();

    // Test Matrix Multiplication
    std::cout << "\nMatrix Multiplication Test: " << std::endl;
    auto matrix_mul_result = einsum("ij,jk->ik", matrix1, matrix2);
    matrix_mul_result.print();

    // Test Dot Product
    std::cout << "\nDot Product Test: " << std::endl;
    auto dot_product_result = einsum("i,i->", vector1, vector2);
    std::cout << "Dot product: " << dot_product_result << std::endl;

    // Test Pointwise Multiplication and Sum
    std::cout << "\nPointwise Multiplication and Sum Test: " << std::endl;
    auto pointwise_mul_sum_result = einsum("ij,ij->", matrix1, matrix4);
    std::cout << "Pointwise multiplication and sum: " << pointwise_mul_sum_result << std::endl;

    // Test Outer Product
    std::cout << "\nOuter Product Test: " << std::endl;
    auto outer_product_result = einsum("i,j->ij", vector1, vector2);
    outer_product_result.print();

    // Test Batch Matrix Multiplication
    std::cout << "\nBatch Matrix Multiplication Test: " << std::endl;
    auto batch_matrix_mul_result = einsum("bij,bjk->bik", tensor1, tensor8);
    batch_matrix_mul_result.print();


    // Test contraction
    std::cout << "\nContraction Test: " << std::endl;
    auto contraction_result = einsum("ijk,kli->jl",tensor1, tensor2);
    std::cout << "Contraction product:\n ";
    contraction_result.print();
    Tensor<int>contraction_result2 = einsum("pqrs,tuqvr->pstuv", *tensor3, *tensor4);
    contraction_result2.print();

    // Test bilinear transformation
    std::cout << "\nBilinear Transformation Test: " << std::endl;
    auto bilinear_result = einsum("ik,jkl,jl->ij", tensor5, tensor6, tensor7);
    std::cout << "Bilinear product:\n ";
    bilinear_result.print();

    // Test Element-wise Multiplication
    std::cout << "\nElement-wise Multiplication Test: " << std::endl;
    auto element_wise_result = einsum("i,i->i", vector1, vector2);
    element_wise_result.print();

    // Test Trace
    std::cout << "\nTrace Test: " << std::endl;
    auto trace_result = einsum("ii", matrix1);
    std::cout << "Trace: " << trace_result << std::endl;
}

void test_math_pointwise(){
    using namespace std;
    cout << "!!!test_math_pointwise test" << endl;
    // std::cout<< "3d test: " << std::endl;
    Tensor<int> add1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<int> add2({3, 2}, {7, 8, 9, 10, 11, 12}); // 2D matrix
    //addition test
    cout << "add test" << endl;
    Tensor<int> add3 = add1 + add2;
    cout << "add3: \n"; 
    add3.print();

    Tensor<int> add4 = add(add1, add2);
    cout << "add4: \n"; 
    add4.print();

    Tensor<int> add5 = add1.add(add2);
    cout << "add5: \n"; 
    add5.print();

    Tensor<int> add6 = add(add1, 1);
    cout << "add6: \n"; 
    add6.print();

    Tensor<int> add7 = add1.add((int)1.0);
    cout << "add7: \n";
    add7.print();

    Tensor<int> add8 = add1 + 1;
    cout << "add8: \n";
    add8.print();



    // //substraction test
    cout << "sub test" << endl;
    Tensor<int> sub1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<int> sub2({3, 2}, {7, 8, 9, 10, 11, 12}); // 2D matrix

    Tensor<int> sub3 = sub1 - sub2;
    cout << "sub3: \n"; 
    sub3.print();

    Tensor<int> sub4 = subtract(sub1, sub2); // 假设`subtract`是相应的减法函数
    cout << "sub4: \n"; 
    sub4.print();

    Tensor<int> sub5 = sub1.subtract(sub2); // 假设`Tensor`类有一个`subtract`成员函数
    cout << "sub5: \n"; 
    sub5.print();

    Tensor<int> sub6 = subtract(sub1, 1); // 这里假设`subtract`函数可以接受一个标量作为第二个参数
    cout << "sub6: \n"; 
    sub6.print();

    Tensor<int> sub7 = sub1.subtract((int)1.0); // 强制类型转换以匹配函数的期望参数类型
    cout << "sub7: \n";
    sub7.print();

    Tensor<int> sub8 = sub1 - 1; // 强制类型转换以匹配函数的期望参数类型
    cout << "sub8: \n";
    sub8.print();



    //multiply test
    cout << "mul test" << endl;
    Tensor<int> mul1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<int> mul2({2, 3}, {7, 8, 9, 10, 11, 12}); // 2D matrix

    Tensor<int> mul3 = mul1 * mul2;
    // cout << "mul3: \n"; 
    mul3.print();

    Tensor<int> mul4 = multiply(mul1, mul2); // 假设`multiply`是相应的乘法函数
    cout << "mul4: \n"; 
    mul4.print();

    Tensor<int> mul5 = mul1.multiply(mul2); // 假设`Tensor`类有一个`multiply`成员函数
    cout << "mul5: \n"; 
    mul5.print();

    Tensor<int> mul6 = multiply(mul1, 1); // 这里假设`multiply`函数可以接受一个标量作为第二个参数
    cout << "mul6: \n"; 
    mul6.print();

    Tensor<int> mul7 = mul1.multiply((int)1.0); // 强制类型转换以匹配函数的期望参数类型
    cout << "mul7: \n";
    mul7.print();

    Tensor<int> mul8 = mul1 * 2; // 强制类型转换以匹配函数的期望参数类型
    cout << "mul8: \n";
    mul8.print();



    //div test
    cout << "div test" << endl;
    Tensor<double> div1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<double> div2({3, 2}, {7, 8, 9, 10, 11, 12}); // 2D matrix

    Tensor<double> div3 = div1 / div2;
    cout << "div3: \n"; 
    div3.print();

    Tensor<double> div4 = divide(div1, div2); // 假设`divide`是相应的除法函数
    cout << "div4: \n"; 
    div4.print();

    Tensor<double> div5 = div1.divide(div2); // 假设`Tensor`类有一个`divide`成员函数
    cout << "div5: \n"; 
    div5.print();

    Tensor<double> div6 = divide(div1, (double)2.0); // 这里假设`divide`函数可以接受一个标量作为第二个参数
    cout << "div6: \n"; 
    div6.print();

    Tensor<double> div7 = div1.divide((double)2.0); // 强制类型转换以匹配函数的期望参数类型
    cout << "div7: \n";
    div7.print();

    Tensor<double> div8 = div1 / 2.0; // 强制类型转换以匹配函数的期望参数类型
    cout << "div8: \n";
    div8.print();



    //log test
    cout << "log test" << endl;
    Tensor<double> log1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<double> log2({3, 2}, {7, 8, 9, 10, 11, 12}); // 2D matrix

    Tensor<double> log3 = log(log1);
    cout << "log3: log(log1)\n"; 
    log3.print();

    Tensor<double> log4 = log1.log();
    cout << "log4: log1.log()\n"; 
    log4.print();

    //project introduction examples for addition
    // official example
    // std::vector<double> testArray1 =  {0.1, 1.2, 2.2, 3.1, 4.9, 5.2};
    // std::vector<double> testArray2 =  {0.2, 1.3, 2.3, 3.2, 4.8, 5.1};
    // Tensor<double> t1({3,2},testArray1);
    // Tensor<double> t2({3,2},testArray2);
    // std::cout << t1 + t2 << std::endl << add(t1, 1.0) << std::endl; //是否应该考虑不同数据类型之间的相加
    // Output
    // [[ 0.3000, 2.5000],
    // [ 4.5000, 6.3000],
    // [ 9.7000, 10.3000]]
    // [[ 1.1000, 2.2000],
    // [ 3.2000, 4.1000],
    // [ 5.9000, 6.2000]]

    // //similiar sub test
    // std::cout << t1 - t2 << std::endl << subtract(t1,t2) << std::endl;
}

void test_math_reduction(){
    using namespace std;
    cout << "!!!test_math_reduction" << endl;
    //sum mean max min test 
    cout << "sum mean max min test" << endl;
    Tensor<int> test1({2 , 3, 2}, {1, 2, 3, 4, 5, 6,7, 8, 9, 10, 11, 12}); // 3D matrix

    cout << "sum " << endl;
    cout << "sum along 0 dimension" << endl;
    std::cout << sum(test1, 0) << std::endl << test1.sum(0) << std::endl;
    cout << "sum along 1 dimension" << endl;
    std::cout << sum(test1, 1) << std::endl << test1.sum(1) << std::endl;
    cout << "sum along 2 dimension" << endl;
    std::cout << sum(test1, 2) << std::endl << test1.sum(2) << std::endl;


    std::cout << "mean" << std::endl;
    std::cout << "mean along 0 dimension" << std::endl;
    std::cout << mean(test1, 0) << std::endl << test1.mean(0) << std::endl;
    std::cout << "mean along 1 dimension" << std::endl;
    std::cout << mean(test1, 1) << std::endl << test1.mean(1) << std::endl;
    std::cout << "mean along 2 dimension" << std::endl;
    std::cout << mean(test1, 2) << std::endl << test1.mean(2) << std::endl;

    std::cout << "max" << std::endl;
    std::cout << "max along 0 dimension" << std::endl;
    std::cout << max(test1, 0) << std::endl << test1.max(0) << std::endl;
    std::cout << "max along 1 dimension" << std::endl;
    std::cout << max(test1, 1) << std::endl << test1.max(1) << std::endl;
    std::cout << "max along 2 dimension" << std::endl;
    std::cout << max(test1, 2) << std::endl << test1.max(2) << std::endl;

    std::cout << "min" << std::endl;
    std::cout << "min along 0 dimension" << std::endl;
    std::cout << min(test1, 0) << std::endl << test1.min(0) << std::endl;
    std::cout << "min along 1 dimension" << std::endl;
    std::cout << min(test1, 1) << std::endl << test1.min(1) << std::endl;
    std::cout << "min along 2 dimension" << std::endl;
    std::cout << min(test1, 2) << std::endl << test1.min(2) << std::endl;
    return;
}

void test_math_comparision(){
    using namespace std;
    cout << "!!!test_math_comparision" << endl;
    //equal test
    // Tensor<bool> t3 = eq(t1, t2);// This compares t1 and t2 element-wise.
    // Tensor<bool> t4= t1.eq(t2);// Another way to compare t1 and t2 element-wise.
    // Tensor<bool> t5= t1 == t2;

    //comparision test 
    std::vector<double> equalArray1 =  {0.1, 1.2, 2.2, 3.1, 4.9, 5.2};
    std::vector<double> equalArray2 =  {0.2, 1.3, 2.2, 3.2, 4.8, 5.2};
    Tensor<double> eq1 = Tensor<double>({3,2},equalArray1);
    Tensor<double> eq2 = Tensor<double>({3,2},equalArray2);
    cout << "eq ne gt... test" << endl;

    cout << "eq" << endl;
    std::cout << eq(eq1,eq2) << std::endl;
    cout << eq1.eq(eq2) << endl;
    cout << (eq1 == eq2) << endl;

    cout << "not eq" << endl;
    std::cout << ne(eq1,eq2) << std::endl;
    cout << eq1.ne(eq2) << endl;
    cout << (eq1 != eq2) << endl;

    cout << "greater than" << endl;
    cout << gt(eq1,eq2) << endl;// 0 0 0 0 1 0
    cout << eq1.gt(eq2) << endl;
    cout << (eq1 > eq2) << endl;

    cout << "greater than or equal" << endl;
    cout << ge(eq1,eq2) << endl; // 0 0 1 0 1 1
    cout << eq1.ge(eq2) << endl;
    cout << (eq1 >= eq2) << endl;

    cout << "less than " << endl;
    cout << lt(eq1,eq2) << endl; // 1 1 0 1 0 0
    cout << eq1.lt(eq2) << endl;
    cout << (eq1 < eq2) << endl;

    cout << "less than or equal " << endl;
    cout << le(eq1,eq2) << endl; // 1 1 1 1 0 1
    cout << eq1.le(eq2) << endl;
    cout << (eq1 <= eq2) << endl;
}

void test_gradient(){
    std::cout << "test_gradient: " << std::endl;
    Tensor<double> matrix({1,1}, {2});
    Tensor<double> matrix_gradient = gradient(matrix,"x*x");
    matrix_gradient.print();
}

void test_clone() {
    Tensor<int> t1({2, 2}, {1, 2, 3, 4});
    Tensor<int> t2 = Tensor<int>::clone(t1);
    t1[{0,0}] = 0;
    std::cout << "t1: " << std::endl;
    t1.print();
    std::cout << "t2: " << std::endl;
    t2.print();
}

void test_print() {
    // 0D tensor
    Tensor<int> tensor0;
    tensor0.print();

    // 1D tensor
    Tensor<int> tensor1({5}, {1, 2, 3, 4, 5});
    tensor1.print();

    // 2D tensor
    Tensor<int> tensor2({2, 3}, {1, 2, 3, 4, 5, 6});
    tensor2.print();

    // 3D tensor
    Tensor<int> tensor3({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    tensor3.print();

    // 4D tensor
    Tensor<int> tensor4({2, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    tensor4.print();

    // 5D tensor
    Tensor<int> tensor5({2, 2, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
    tensor5.print();
}



int main() {
    test_diagonal();//item 1
    test_index();//item 2,3
    test_slice();//item 4,5
    test_mutate();//item 6,7
    test_transpose();//item 8
    test_permute();//item 9
    test_view();//item 10
    test_clone();//item 11
    test_einsum();//item 12
    test_save_load();//item 15
    test_gradient();  //item 18
    test_print();//item 20
    test_tensor();
    test_rand();
    test_cat();
    test_tile();
    test_bool_Tensor();
    test_math_comparision(); //comparision == != > < >= <=
    return 0;
}
