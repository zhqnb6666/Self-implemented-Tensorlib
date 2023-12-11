//
// Created by Lenovo on 2023/12/2.
//

#include "tensor.h"

// a helper function to calculate the linear index from a multi-index
template<typename T>
size_t Tensor<T>::linear_index(const std::vector<size_t> &indices) const {
    size_t index = 0;
    size_t factor = 1;
    for (size_t i = dims.size() - 1; i < dims.size(); i--) {
        // check if the indices are valid
        if (indices[i] >= dims[i]) {
            throw std::out_of_range("Index out of range");
        }
        // calculate the linear index
        index += indices[i] * factor;
        factor *= dims[i];
    }
    return index;
}

// a default constructor that creates an empty tensor
template<typename T>
Tensor<T>::Tensor() {}

// a constructor that creates a tensor with given dimensions and values
template<typename T>
Tensor<T>::Tensor(const std::vector<size_t> &dimensions, const std::vector<T> &values) {
    // check if the dimensions and values match
    if (dimensions.empty() || values.empty()) {
        throw std::invalid_argument("Empty dimensions or values");
    }
    size_t size = 1;
    for (size_t dim: dimensions) {
        // check if the dimensions are positive
        if (dim == 0) {
            throw std::invalid_argument("Zero dimension");
        }
        size *= dim;
    }
    if (size != values.size()) {
        throw std::invalid_argument("Dimensions and values do not match");
    }
    // copy the dimensions and values to the class members
    dims = dimensions;
    data = values;
}

// a constructor that creates a tensor with given dimensions and a default value
template<typename T>
Tensor<T>::Tensor(const std::vector<size_t> &dimensions, const T &value) {
    // check if the dimensions are valid
    if (dimensions.empty()) {
        throw std::invalid_argument("Empty dimensions");
    }
    size_t size = 1;
    for (size_t dim: dimensions) {
        // check if the dimensions are positive
        if (dim == 0) {
            throw std::invalid_argument("Zero dimension");
        }
        size *= dim;
    }
    // resize the data vector and fill it with the default value
    data.resize(size, value);
    // copy the dimensions to the class member
    dims = dimensions;
}

// a copy constructor that creates a tensor from another tensor
template<typename T>
Tensor<T>::Tensor(const Tensor<T> &other) {
    // copy the data and dimensions from the other tensor
    data = other.data;
    dims = other.dims;
}

// a move constructor that creates a tensor from another tensor
template<typename T>
Tensor<T>::Tensor(Tensor<T> &&other) noexcept {
    // move the data and dimensions from the other tensor
    data = std::move(other.data);
    dims = std::move(other.dims);
}

// a copy assignment operator that assigns a tensor from another tensor
template<typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other) {
    // check for self-assignment
    if (this != &other) {
        // copy the data and dimensions from the other tensor
        data = other.data;
        dims = other.dims;
    }
    return *this;
}

// a move assignment operator that assigns a tensor from another tensor
template<typename T>
Tensor<T> &Tensor<T>::operator=(Tensor<T> &&other) {
    // check for self-assignment
    if (this != &other) {
        // move the data and dimensions from the other tensor
        data = std::move(other.data);
        dims = std::move(other.dims);
    }
    return *this;
}

// a destructor that destroys the tensor
template<typename T>
Tensor<T>::~Tensor() {}

// a function that returns the order (rank) of the tensor
template<typename T>
size_t Tensor<T>::order() const {
    return dims.size();
}

// a function that returns the size (number of elements) of the tensor
template<typename T>
size_t Tensor<T>::size() const {
    return data.size();
}

// a function that returns the dimensions of the tensor
template<typename T>
std::vector<size_t> Tensor<T>::dimensions() const {
    return dims;
}

// a function that returns the data of the tensor
template<typename T>
std::vector<T> Tensor<T>::values() const {
    return data;
}

// a function that prints the tensor elements in a formatted way
template<typename T>
void Tensor<T>::print(std::ostream &os, const std::vector<size_t> &indices, size_t dim) const {
    if(dim == 1&&indices[0]!=0){
        os<<endl;
        if(indices.size()==3)os<<endl;
    }
    if(dim == 2&&indices[1]!=0)os<<endl;
    os <<"[";
    for (size_t i = 0; i < dims[dim]; i++) {
        std::vector<size_t> new_indices = indices;
        new_indices[dim] = i;
        if (dim == dims.size() - 1) {
            // print the element
            os << data[linear_index(new_indices)];
        } else {
            // recursively print the sub-tensor
            print(os, new_indices, dim + 1);
        }
        if (i < dims[dim] - 1) {
            os << ", ";
        }
    }
    os << "]";
}

template<typename T>
void Tensor<T>::print() const {
    // check if the tensor is void
    if(data.empty()){
        std::cout << "Empty tensor" << std::endl;
        return;
    }
    print(std::cout, std::vector<size_t>(dims.size(), 0), 0);
    std::cout << std::endl;
    std::cout <<"----------------------------------------"<< std::endl;
}

// an operator that prints the tensor elements in a formatted way
template<typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    if(tensor.data.empty()){
        os << "Empty tensor" << std::endl;
        return os;
    }
    tensor.print(os, std::vector<size_t>(tensor.dims.size(), 0), 0);
    os << std::endl;
    return os;
}


// an operator that returns a reference to the element at a given multi-index
template<typename T>
T &Tensor<T>::operator()(const std::vector<size_t> &indices) {
    // check if the indices are valid
    if (indices.size() != dims.size()) {
        throw std::invalid_argument("Indices and dimensions do not match");
    }
    // return a reference to the element at the linear index
    return data[linear_index(indices)];
}

// an operator that returns a const reference to the element at a given multi-index
template<typename T>
const T &Tensor<T>::operator()(const std::vector<size_t> &indices) const {
    // check if the indices are valid
    if (indices.size() != dims.size()) {
        throw std::invalid_argument("Indices and dimensions do not match");
    }
    // return a const reference to the element at the linear index
    return data[linear_index(indices)];
}

// an operator that returns a reference to the element at a given linear index
template<typename T>
T &Tensor<T>::operator[](size_t index) {
    // check if the index is valid
    if (index >= data.size()) {
        throw std::out_of_range("Index out of range");
    }
    // return a reference to the element at the index
    return data[index];
}

// an operator that returns a const reference to the element at a given linear index
template <typename T>
const T& Tensor<T>::operator[](size_t index) const {
    // check if the index is valid
    if (index >= data.size()) {
        throw std::out_of_range("Index out of range");
    }
    // return a const reference to the element at the index
    return data[index];
}

