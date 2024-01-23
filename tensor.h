//
// Created by Lenovo on 2023/12/2.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <memory>


using namespace std;
// a class template for tensors of arbitrary rank and size
template<typename T>
class Tensor {
private:
    // a vector to store the tensor elements
    std::shared_ptr<std::vector<T>> data;

    // a vector to store the tensor dimensions
    std::vector<size_t> dims;

    // a vector to store the strides
    std::vector<size_t> strides;

    //an int to store the offset
    size_t offset = 0;

    // a helper function to calculate the linear index from a multi-index
    size_t linear_index(const std::vector<size_t> &indices) const;

public:
    // a default constructor that creates an empty tensor
    Tensor();

    // a constructor that creates a tensor with given dimensions and values
    Tensor(const std::vector<size_t> &dimensions, const std::vector<T> &values);

    // a constructor that creates a tensor with given dimensions and a default value
    explicit Tensor(const std::vector<size_t> &dimensions, const T &value = T());

    // a copy constructor that creates a tensor from another tensor
    Tensor(const Tensor<T> &other);

    //a constructor that creates a tensor from a shared_ptr
    Tensor(const std::vector<size_t> &dimensions, std::shared_ptr<std::vector<T>> values);

    // a constructor that creates a tensor with given dimensions  values and offset
    Tensor(const std::vector<size_t> &dimensions, std::shared_ptr<std::vector<T>> values, size_t offset);

    //create a tensor from another tensor
    Tensor(Tensor<T> &other);

    // a copy assignment operator that assigns a tensor from another tensor
    Tensor<T> &operator=(const Tensor<T> &other);

    // a move assignment operator that assigns a tensor from another tensor
    Tensor<T> &operator=(Tensor<T> &other);

    // a constructor that creates a tensor with all information
    Tensor<T>(const std::vector<size_t> &dimensions, const std::vector<T> &values, const std::vector<size_t> &strides, size_t offset) ;

        //set all the elements of the tensor to a given value
    Tensor<T> &operator=(const T &value);

    //set all the elements of the tensor to a given value array
    Tensor<T> &operator=(const std::vector<T> &values);

    // a destructor that destroys the tensor
    ~Tensor();

    //Create a tensor with a given shape and data type and initialize it randomly
    static Tensor<T> *rand(const std::vector<size_t> &dimensions);

    static Tensor<T> *Zeros(const std::vector<size_t> &dimensions);

    static Tensor<T> *Ones(const std::vector<size_t> &dimensions);

    //diagonal, a function that returns a diagonal tensor
    static Tensor<T> *diagonal(const std::vector<T> &values);

    // a function that returns the eye matrix
    static Tensor<T> *eye(size_t n);

    // a function that returns the order (rank) of the tensor
    size_t order() const;

    // a function that returns the size (number of elements) of the tensor
    size_t size() const;

    // a function that returns the stride of the tensor
    std::vector<size_t> getstrides() const ;

    // a function that returns the offset of the tensor
    size_t getoffset() const ;

    // a function that returns the dimensions of the tensor
    std::vector<size_t> dimensions() const;

    // a function that returns the const data of the tensor
    const T *data_ptr() const;

    // a function that returns the data_ptr of the tensor
    T *data_ptr();

    // a function that returns the type of the tensor
    string type() const;

    // a function that returns the data of the tensor
    std::vector<T> values() const;

    // a function that prints the tensor elements in a formatted way
    void print(std::ostream &os, const std::vector<size_t> &indices, size_t dim) const;

    void print() const;

    // an operator that prints the tensor elements in a formatted way
    template<typename U>
    friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

    // an operator that returns a reference to the element at a given multi-index
    T &operator[](const std::vector<size_t> &indices);

    // an operator that returns a const reference to the element at a given multi-index
    const T &operator[](const std::vector<size_t> &indices) const;

    // an operator that returns a reference to the element at a given index as a tensor
    Tensor<T>* operator()(size_t index);

    Tensor<T>* operator()(size_t index, const std::pair<size_t, size_t> &range);

    //concatenate,an operator that joins two tensors along a given dimension
    static Tensor<T> *cat(const std::vector<Tensor<T>>& tensors, int dim);

    //tile,an operator that repeatedly joins the tensor along a given dimension
    static Tensor<T> *tile(const Tensor<T>& tensor, int dim, int n);

    //transpose,an operator that returns a transposed tensor
    static Tensor<T> * transpose( Tensor<T>& tensor, int dim1, int dim2);

    Tensor<T> transpose( int dim1, int dim2) ;

    //permute,an operator that returns a permuted tensor
    static Tensor<T> *permute( Tensor<T>& tensor, const std::vector<int>& dims);

    Tensor<T> permute( const std::vector<int>& dims);

    Tensor<T> permute( const std::vector<int>& dims) const;

    static Tensor<T> clone(const Tensor<T>& tensor);


    //view,an operator that returns a reshaped tensor
    static Tensor<T> *view( Tensor<T>& tensor, const std::vector<size_t>& dims);

    Tensor<T> view( const std::vector<size_t>& dims);

    //add as a member function
    Tensor<T> add(const Tensor<T> &rhs) const;

    Tensor<T> add(const T &rhs) const;

    Tensor<T> subtract(const T &rhs) const;

    Tensor<T> subtract(const Tensor<T> &rhs) const;

    Tensor<T> multiply(const T &rhs) const;

    Tensor<T> multiply(const Tensor<T> &rhs) const;

    Tensor<T> divide(const T &rhs) const;

    Tensor<T> divide(const Tensor<T> &rhs) const;

    Tensor<T> log() const;

    Tensor<T> mean(int dim) const;

    bool increment_indices(vector<size_t> &indices, const vector<size_t> &dims) const;

    Tensor<T> min(int dim) const;

    Tensor<T> max(int dim) const;

    Tensor<T> sum(int dim) const;

    Tensor<bool> eq(const Tensor<T> &rhs) const;
    Tensor<bool> ne(const Tensor<T> &rhs) const;
    Tensor<bool> gt(const Tensor<T> &rhs) const;
    Tensor<bool> lt(const Tensor<T> &rhs) const;
    Tensor<bool> ge(const Tensor<T> &rhs) const;
    Tensor<bool> le(const Tensor<T> &rhs) const;
};
// include the implementation file
//#include "tensor.cpp"

#endif
