//
// Created by Lenovo on 2023/12/2.
//

#include "tensor.h"
#include <random>
#include <algorithm>

// a helper function to calculate the linear index from a multi-index
template<typename T>
size_t Tensor<T>::linear_index(const std::vector<size_t> &indices) const {
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides[i];
    }
    return index;
}

// a default constructor that creates an empty tensor
template<typename T>
Tensor<T>::Tensor() {}

//a constructor that creates a tensor from a shared_ptr
template<typename T>
Tensor<T>::Tensor(const std::vector<size_t> &dimensions, std::shared_ptr<std::vector<T>> values) {
    dims = dimensions;
    data = values;
    strides.resize(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}

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
    // copy the dimensions to the class members
    dims = dimensions;
    // create a new shared_ptr with the values and assign it to the data member
    data = std::make_shared<std::vector<T>>(values);
    strides.resize(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
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
    // create a new shared_ptr with the default value and assign it to the data member
    data = std::make_shared<std::vector<T>>(size, value);
    // copy the dimensions to the class member
    dims = dimensions;
    strides.resize(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}

// a copy constructor that creates a tensor from another tensor
template<typename T>
Tensor<T>::Tensor(const Tensor<T> &other) {
    // copy the dimensions from the other tensor
    dims = other.dims;
    // create a new shared_ptr with a copy of the other tensor's data and assign it to the data member
    data = std::make_shared<std::vector<T>>(*other.data);
    strides = other.strides;
}


template<typename T>
Tensor<T>::Tensor(Tensor<T> &other) {
    // share the data and dimensions with the other tensor
    data = other.data;
    dims = other.dims;
    strides = other.strides;
}

// a copy assignment operator that assigns a tensor from another tensor
template<typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other) {
    // check for self-assignment
    if (this != &other) {
        // copy the data and dimensions from the other tensor
        data = std::make_shared<std::vector<T>>(*other.data);
        dims = other.dims;
        strides = other.strides;
    }
    return *this;
}

// a move assignment operator that assigns a tensor from another tensor
template<typename T>
Tensor<T> &Tensor<T>::operator=(Tensor<T> &other) {
    // check for self-assignment
    if (this != &other) {
        // move the data and dimensions from the other tensor
        data = other.data;
        dims = other.dims;
        strides = other.strides;
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
    return data->size();
}

// a function that returns the data_ptr of the tensor
template<typename T>
T *Tensor<T>::data_ptr() {
    return data->data();
}

//Create a tensor with a given shape and data type and initialize it randomly
template<typename T>
Tensor<T>* Tensor<T>::rand(const std::vector<size_t> &dimensions) {
    // Calculate the total size of the tensor
    size_t size = 1;
    for (size_t dim: dimensions) {
        if (dim == 0) {
            throw std::invalid_argument("Zero dimension");
        }
        size *= dim;
    }
    // Create a data vector and fill it with random numbers
    std::vector<T> data(size);
    if (std::is_same<T, int>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(std::rand() % 101); // Generate random int between 0 and 100
        }
    } else if (std::is_same<T, float>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(std::rand() / (RAND_MAX + 1.0)); // Generate random float between 0 and 1
        }
    } else if (std::is_same<T, char>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(std::rand() % 256); // Generate random char between 0 and 255
        }

    } else if (std::is_same<T, double>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(std::rand() / (RAND_MAX + 1.0)); // Generate random double between 0 and 1
        }
    } else if (std::is_same<T, bool>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(std::rand() % 2); // Generate random bool
        }
    } else {
        throw std::invalid_argument("Unsupported data type");
    }
    Tensor<T> *tensor = new Tensor<T>(dimensions, data);
    return tensor;
}

//Create a tensor with a given shape and data type and initialize it with zeros
template<typename T>
Tensor<T>* Tensor<T>::Zeros(const std::vector<size_t> &dimensions) {
    auto *tensor = new Tensor<T>(dimensions, 0);
    return tensor;
}

//Create a tensor with a given shape and data type and initialize it with ones
template<typename T>
Tensor<T>* Tensor<T>::Ones(const std::vector<size_t> &dimensions) {
    auto *tensor = new Tensor<T>(dimensions, 1);
    return tensor;
}
//Create a identity matrix with a given shape and data type
template<typename T>
Tensor<T>* Tensor<T>::eye(size_t n) {
    // Create a data vector and fill it with zeros
    std::vector<T> data(n * n, 0);

    // Set the diagonal elements to 1
    for (size_t i = 0; i < n; i++) {
        data[i * n + i] = 1;
    }

    // Create a new Tensor object with the given dimensions and data
    Tensor<T>* tensor = new Tensor<T>({n, n}, data);

    // Return the pointer to the new Tensor object
    return tensor;
}



// a function that returns the type of the tensor
template<typename T>
string Tensor<T>::type() const {
    string type_id_name = typeid(T).name();
    if (type_id_name == "i") {
        return "int";
    } else if (type_id_name == "b") {
        return "bool";
    } else if (type_id_name == "c") {
        return "char";
    } else if (type_id_name == "d") {
        return "double";
    } else if (type_id_name == "f") {
        return "float";
    } else {
        return "unknown";
    }
}

// a function that returns the dimensions of the tensor
template<typename T>
std::vector<size_t> Tensor<T>::dimensions() const {
    return dims;
}

// a function that returns the data of the tensor
template<typename T>
std::vector<T> Tensor<T>::values() const {
    return *data;
}

// a function that prints the tensor elements in a formatted way
template<typename T>
void Tensor<T>::print(std::ostream &os, const std::vector<size_t> &indices, size_t dim) const {
    if (dim == 1 && indices[0] != 0) {
        os << endl;
        if (indices.size() == 3)os << endl;
    }
    if (dim == 2 && indices[1] != 0)os << endl;
    os << "[";
    for (size_t i = 0; i < dims[dim]; i++) {
        std::vector<size_t> new_indices = indices;
        new_indices[dim] = i;
        if (dim == dims.size() - 1) {
            // print the element
            os << (*data)[linear_index(new_indices)];
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
    if (!data||data->empty()) {
        std::cout << "Empty tensor" << std::endl;
        return;
    }
    print(std::cout, std::vector<size_t>(dims.size(), 0), 0);
    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

// an operator that prints the tensor elements in a formatted way
template<typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    if (!tensor.data || tensor.data->empty()) {
        os << "Empty tensor" << std::endl;
        return os;
    }
    tensor.print(os, std::vector<size_t>(tensor.dims.size(), 0), 0);
    os << std::endl;
    return os;
}


// an operator that returns a reference to the element at a given multi-index
template<typename T>
T &Tensor<T>::operator[](const std::vector<size_t> &indices) {
    // check if the indices are valid
    if (indices.size() != dims.size()) {
        throw std::invalid_argument("Indices and dimensions do not match");
    }
    // return a reference to the element at the linear index
    return (*data)[linear_index(indices)];
}

// an operator that returns a const reference to the element at a given multi-index
template<typename T>
const T &Tensor<T>::operator[](const std::vector<size_t> &indices) const {
    // check if the indices are valid
    if (indices.size() != dims.size()) {
        throw std::invalid_argument("Indices and dimensions do not match");
    }
    // return a const reference to the element at the linear index
    return (*data)[linear_index(indices)];
}
// indexing,an operator that returns a reference to the element at a given index as a tensor
template<typename T>
Tensor<T>* Tensor<T>::operator()(size_t index)  {
    // Check if the index is valid
    if (index >= dims[0]) {
        throw std::out_of_range("Index out of range");
    }
    // Calculate the new dimensions
    std::vector<size_t> new_dims(dims.begin() + 1, dims.end());
    // Calculate the offset of the data pointer
    size_t offset = index * std::accumulate(new_dims.begin(), new_dims.end(), 1, std::multiplies<size_t>());
    // Create a new Tensor object with the new dimensions and shared data
    Tensor<T>* result = new Tensor<T>(new_dims, data);
    // Shift the data pointer by the offset
    std::rotate(result->data->begin(), result->data->begin() + offset, result->data->end());
    // Resize the data to match the new dimensions
    result->data->resize(std::accumulate(new_dims.begin(), new_dims.end(), 1, std::multiplies<size_t>()));
    return result;
}

// slicing,an operator that returns a reference to the element at a given index and range as a tensor
template<typename T>
Tensor<T>* Tensor<T>::operator()(size_t index, const std::pair<size_t, size_t>& range)  {
    // Check if the index and range are valid
    if (index >= dims[0] || range.first > range.second || range.second > dims[1]) {
        throw std::out_of_range("Index or range out of range");
    }
    // Calculate the new dimensions
    std::vector<size_t> new_dims = {range.second - range.first};
    // Calculate the offset of the data pointer
    size_t offset = (index * dims[1] + range.first);
    // Create a new Tensor object with the new dimensions and shared data
    Tensor<T>* result = new Tensor<T>(new_dims, data);
    // Shift the data pointer by the offset
    std::rotate(result->data->begin(), result->data->begin() + offset, result->data->end());
    // Resize the data to match the new dimensions
    result->data->resize(range.second - range.first);
    return result;
}

//concatenate,an operator that joins two tensors along a given dimension
template<typename T>
Tensor<T> * Tensor<T>::cat(const std::vector<Tensor<T>>& tensors, int dim) {
    // Check if there is at least one tensor
    if (tensors.empty()) {
        throw std::invalid_argument("No tensors to concatenate");
    }

    // Check if the dimension is valid
    if (dim < 0 || dim >= tensors[0].dims.size()) {
        throw std::invalid_argument("Invalid dimension");
    }

    // Calculate the size of the new tensor
    std::vector<size_t> new_dims = tensors[0].dims;
    for (size_t i = 1; i < tensors.size(); ++i) {
        new_dims[dim] += tensors[i].dims[dim];
    }
    // Create the new tensor
    Tensor<T>* result = new Tensor<T>(new_dims,0);
    vector<size_t> offset(result->dims.size(),0);
    vector<size_t> indices(result->dims.size(), 0);
    for(Tensor tensor:tensors){
        for (int i = 0; i < tensor.size(); ++i) {
            //calculate the indices of tensor
            int temp = i;
            for (size_t d = 0; d < tensor.dims.size(); ++d) {
                indices[d] = temp / tensor.strides[d];
                indices[d] += offset[d];
                temp %= tensor.strides[d];
            }
           (*result)[indices] = (*tensor.data)[i];
        }
        offset[dim]+=tensor.dims[dim];
    }
    return result;
}


