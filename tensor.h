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


    //create a tensor from another tensor
    Tensor(Tensor<T> &other);

    // a copy assignment operator that assigns a tensor from another tensor
    Tensor<T> &operator=(const Tensor<T> &other);

    // a move assignment operator that assigns a tensor from another tensor
    Tensor<T> &operator=(Tensor<T> &other);

    // a destructor that destroys the tensor
    ~Tensor();


    //Create a tensor with a given shape and data type and initialize it randomly
    static Tensor<T> *rand(const std::vector<size_t> &dimensions);

    static Tensor<T> *Zeros(const std::vector<size_t> &dimensions);

    static Tensor<T> *Ones(const std::vector<size_t> &dimensions);

    // a function that returns the eye matrix
    static Tensor<T> *eye(size_t n);

    // a function that returns the order (rank) of the tensor
    size_t order() const;

    // a function that returns the size (number of elements) of the tensor
    size_t size() const;

    // a function that returns the dimensions of the tensor
    std::vector<size_t> dimensions() const;

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
    friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    }

    // an operator that returns a reference to the element at a given multi-index
    T &operator[](const std::vector<size_t> &indices);

    // an operator that returns a const reference to the element at a given multi-index
    const T &operator[](const std::vector<size_t> &indices) const;

    // an operator that returns a reference to the element at a given index as a tensor
    Tensor<T>* operator()(size_t index);

    Tensor<T>* operator()(size_t index, const std::pair<size_t, size_t> &range);


};

// include the implementation file
//#include "tensor.cpp"

#endif
