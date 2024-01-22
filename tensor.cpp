//
// Created by Lenovo on 2023/12/2.
//

#include "tensor.h"
#include "tensor_bool.h"
#include <random>
#include <algorithm>
#include <iomanip>

// a helper function to calculate the linear index from a multi-index
template<typename T>
size_t Tensor<T>::linear_index(const std::vector<size_t> &indices) const {
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides[i];
    }
    return index+ offset;
}

// a default constructor that creates an empty tensor
template<typename T>
Tensor<T>::Tensor() = default;

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

// a constructor that creates a tensor with given dimensions  values and offset
template<typename T>
Tensor<T>::Tensor(const std::vector<size_t> &dimensions, std::shared_ptr<std::vector<T>> values, size_t offset){
    dims = dimensions;
    data = values;
    strides.resize(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    this->offset = offset;
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

// a constructor that creates a tensor with all information
template<typename T>
Tensor<T>::Tensor(const std::vector<size_t> &dimensions, const std::vector<T> &values, const std::vector<size_t> &strides, size_t offset) {
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
    this->strides = strides;
    this->offset = offset;
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
    offset = other.offset;
}

template<typename T>
Tensor<T>::Tensor(Tensor<T> &other) {
    // share the data and dimensions with the other tensor
    data = other.data;
    dims = other.dims;
    strides = other.strides;
    offset = other.offset;
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
        offset = other.offset;
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
        offset = other.offset;
    }
    return *this;
}

//set all elements in the tensor to the given value
template<typename T>
Tensor<T>& Tensor<T>::operator=(const T& value) {
    // Set all elements in the tensor to the given value
    std::fill(data->begin()+offset, data->begin()+offset+this->size(), value);
    return *this;
}

//set all elements in the tensor to the given array
template<typename T>
Tensor<T>& Tensor<T>::operator=(const std::vector<T>& values) {
    // Check if the size of the values matches the size of the 0th dimension
    if (values.size() != dims[0]) {
        throw std::invalid_argument("Size of values does not match size of 0th dimension");
    }
    // Set the values at the 0th dimension
    int range = strides[0];
    for (size_t i = 0; i < values.size(); ++i) {
        std::fill(data->begin() + i * range+offset, data->begin() + (i + 1) * range+offset, values[i]);
    }
    return *this;
}

// a destructor that destroys the tensor
template<typename T>
Tensor<T>::~Tensor() = default;

// a function that returns the order (rank) of the tensor
template<typename T>
size_t Tensor<T>::order() const {
    return dims.size();
}

// a function that returns the size (number of elements) of the tensor
template<typename T>
size_t Tensor<T>::size() const {
    return data->size()-offset;
}

// a function that returns the stride of the tensor
template<typename T>
std::vector<size_t> Tensor<T>::getstrides() const {
    return strides;
}

// a function that returns the offset of the tensor
template<typename T>
size_t Tensor<T>::getoffset() const {
    return offset;
}

// a function that returns the const data_ptr of the tensor
template<typename T>
const T *Tensor<T>::data_ptr() const {
    return data->data();
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
    std::random_device rd;
    std::mt19937 gen(rd());
    if (std::is_same<T, int>::value) {
        std::uniform_int_distribution<> dis(0, 100);
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(dis(gen)); // Generate random int between 0 and 100
        }
    } else if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
        std::uniform_real_distribution<> dis(0, 1);
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(dis(gen)); // Generate random float or double between 0 and 1
        }
    } else if (std::is_same<T, char>::value) {
        std::uniform_int_distribution<> dis(0, 255);
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(dis(gen)); // Generate random char between 0 and 255
        }
    } else if (std::is_same<T, bool>::value) {
        std::uniform_int_distribution<> dis(0, 1);
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(dis(gen)); // Generate random bool
        }
    } else {
        throw std::invalid_argument("Unsupported data type");
    }
    auto *tensor = new Tensor<T>(dimensions, data);
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
//full
template<typename T>
Tensor<T>* Tensor<T>::full(const std::vector<size_t> &dimensions, const T &value) {
    auto *tensor = new Tensor<T>(dimensions, value);
    return tensor;
}

//Create an identity matrix with a given shape and data type
template<typename T>
Tensor<T>* Tensor<T>::eye(size_t n) {
    // Create a data vector and fill it with zeros
    std::vector<T> data(n * n, 0);

    // Set the diagonal elements to 1
    for (size_t i = 0; i < n; i++) {
        data[i * n + i] = 1;
    }

    // Create a new Tensor object with the given dimensions and data
    auto* tensor = new Tensor<T>({n, n}, data);

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
        if (indices.size() >= 3)os << endl;
    }
    if (dim == 2 && indices[1] != 0)os << endl;
    if(dim == 2 && indices[1] != 0)os << "  [";
    else if(dim == 1 && indices[0] != 0)os << " [";
    else os << "[";
    for (size_t i = 0; i < dims[dim]; i++) {
        std::vector<size_t> new_indices = indices;
        new_indices[dim] = i;
        if (dim == dims.size() - 1) {
            // print the element
//            os << (*data)[linear_index(new_indices)];
            if (std::is_same<T, bool>::value) {
                os << std::setw(7) << std::right << ((*data)[linear_index(new_indices)] ? "true" : "false");
            } else {
                os << std::setw(7) << std::right << (*data)[linear_index(new_indices)];
            }
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
    if(dims.size()==1&&dims[0] ==1){
        std::cout << data->at(offset) << std::endl;
    }else print(std::cout, std::vector<size_t>(dims.size(), 0), 0);
    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}


// an operator that prints the tensor elements in a formatted way
template<typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    if (!tensor.data || tensor.data->empty()) {
        os << "Empty tensor" << std::endl;
        return os;
    }if(tensor.dims.size()==1&&tensor.dims[0] ==1){
        os << tensor.data->at(tensor.offset) << std::endl;
        return os;
    }else{
    tensor.print(os, std::vector<size_t>(tensor.dims.size(), 0), 0);
    os << std::endl;}
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
    // Calculate the new offset
    size_t new_offset = offset + index * strides[0];
    // Create a new Tensor object with the new dimensions and shared data
    auto* result = new Tensor<T>(new_dims, data, new_offset);
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
    std::vector<size_t> new_dims(dims.begin() + 1, dims.end());
    new_dims[0] = range.second - range.first;
    // Calculate the new offset
    size_t new_offset = offset + index * strides[0] + range.first * strides[1];
    // Create a new Tensor object with the new dimensions and shared data
    auto* result = new Tensor<T>(new_dims, data, new_offset);
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
    auto* result = new Tensor<T>(new_dims,0);
    vector<size_t> offset_vector(result->dims.size(), 0);
    vector<size_t> indices(result->dims.size(), 0);
    for(Tensor tensor:tensors){
        for (int i = 0; i < tensor.size(); ++i) {
            //calculate the indices of tensor
            int temp = i;
            for (size_t d = 0; d < tensor.dims.size(); ++d) {
                indices[d] = temp / tensor.strides[d];
                indices[d] += offset_vector[d];
                temp %= tensor.strides[d];
            }
           (*result)[indices] = (*tensor.data)[i+tensor.offset];
        }
        offset_vector[dim]+=tensor.dims[dim];
    }
    return result;
}

//tile,an operator that repeatedly joins the tensor along a given dimension
template<typename T>
Tensor<T> * Tensor<T>::tile(const Tensor<T>& tensor, int dim, int n) {
    // Check if the dimension is valid
    if (dim < 0 || dim >= tensor.dims.size()) {
        throw std::invalid_argument("Invalid dimension");
    }

    // Create a vector to store the tensors to concatenate
    std::vector<Tensor<T>> tensors(n, tensor);

    // Concatenate the tensors along the given dimension
    return cat(tensors, dim);
}
//transpose,an operator that returns a transposed tensor
template<typename T>
Tensor<T> Tensor<T>::transpose(int dim1, int dim2)  {
    // Check if the dimensions are valid
    if (dim1 < 0 || dim1 >= dims.size() || dim2 < 0 || dim2 >= dims.size()) {
        throw std::invalid_argument("Invalid dimensions");
    }

    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(*this);

    // Swap the strides
    std::swap(result.strides[dim1], result.strides[dim2]);

    // Swap the dimensions
    std::swap(result.dims[dim1], result.dims[dim2]);

    return result;
}

template<typename T>
Tensor<T> * Tensor<T>::transpose( Tensor<T>& tensor, int dim1, int dim2) {
    // Check if the dimensions are valid
    if (dim1 < 0 || dim1 >= tensor.dims.size() || dim2 < 0 || dim2 >= tensor.dims.size()) {
        throw std::invalid_argument("Invalid dimensions");
    }

    // Create a new Tensor object with the same data, dimensions, strides and offset
    auto* result = new Tensor<T>(tensor);

    // Swap the strides
    std::swap(result->strides[dim1], result->strides[dim2]);

    // Swap the dimensions
    std::swap(result->dims[dim1], result->dims[dim2]);

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::permute(const std::vector<int>& dims) {
    // Check if the dimensions are valid
    if (dims.size() != this->dims.size()) {
        throw std::invalid_argument("Invalid dimensions");
    }

    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(*this);

    // Permute the strides and dimensions
    for (size_t i = 0; i < dims.size(); ++i) {
        result.strides[i] = this->strides[dims[i]];
        result.dims[i] = this->dims[dims[i]];
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::permute(const std::vector<int>& dims) const{
    // Check if the dimensions are valid
    if (dims.size() != this->dims.size()) {
        throw std::invalid_argument("Invalid dimensions");
    }

    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(*this);

    // Permute the strides and dimensions
    for (size_t i = 0; i < dims.size(); ++i) {
        result.strides[i] = this->strides[dims[i]];
        result.dims[i] = this->dims[dims[i]];
    }

    return result;
}

template<typename T>
Tensor<T>* Tensor<T>::permute(Tensor<T>& tensor, const std::vector<int>& dims) {
    // Check if the dimensions are valid
    if (dims.size() != tensor.dims.size()) {
        throw std::invalid_argument("Invalid dimensions");
    }

    // Create a new Tensor object with the same data, dimensions, strides and offset
    auto *result= new Tensor<T>(tensor);

    // Permute the strides and dimensions
    for (size_t i = 0; i < dims.size(); ++i) {
        result->strides[i] = tensor.strides[dims[i]];
        result->dims[i] = tensor.dims[dims[i]];
    }

    return result;
}
template<typename T>
Tensor<T> Tensor<T>::view(const std::vector<size_t>& dims) {
    // Check if the dimensions are valid
    size_t size = 1;
    for (size_t dim: dims) {
        if (dim == 0) {
            throw std::invalid_argument("Zero dimension");
        }
        size *= dim;
    }
    if (size != this->size()) {
        throw std::invalid_argument("Dimensions do not match");
    }

    // Create a new Tensor object with the same data, new dimensions, same strides and offset
    Tensor<T> result(*this);
    result.dims = dims;
    std::vector<size_t> new_strides(dims.size());
    new_strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * dims[i + 1];
    }
    result.strides = new_strides;
    return result;
}

template<typename T>
Tensor<T>* Tensor<T>::view(Tensor<T>& tensor, const std::vector<size_t>& dims) {
    // Check if the dimensions are valid
    size_t size = 1;
    for (size_t dim: dims) {
        if (dim == 0) {
            throw std::invalid_argument("Zero dimension");
        }
        size *= dim;
    }
    if (size != tensor.size()) {
        throw std::invalid_argument("Dimensions do not match");
    }

    // Create a new Tensor object with the same data, new dimensions, same strides and offset
    auto *result = new Tensor<T>(tensor);
    result->dims = dims;
    std::vector<size_t> new_strides(dims.size());
    new_strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * dims[i + 1];
    }
    result->strides = new_strides;
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::clone(const Tensor<T> &tensor) {
    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(tensor);
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::add(const T &rhs) const {

    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(*this);

    // Add the scalar to all elements
    for (size_t i = 0; i < result.size(); ++i) {
        (*result.data)[i+result.offset] += rhs;
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::add(const Tensor<T> &rhs) const {
    // Check if the dimensions match
    if (this->dims != rhs.dims) {
        throw std::invalid_argument("Dimensions do not match");
    }

    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(*this);

    // Add the elements of the other tensor
    for (size_t i = 0; i < result.size(); ++i) {
        (*result.data)[i+result.offset] += (*rhs.data)[i+rhs.offset];
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::subtract(const T &rhs) const {

    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(*this);

    // Subtract the scalar from all elements
    for (size_t i = 0; i < result.size(); ++i) {
        (*result.data)[i+result.offset] -= rhs;
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::subtract(const Tensor<T> &rhs) const {
    // Check if the dimensions match
    if (this->dims != rhs.dims) {
        throw std::invalid_argument("Dimensions do not match");
    }

    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(*this);

    // Subtract the elements of the other tensor
    for (size_t i = 0; i < result.size(); ++i) {
        (*result.data)[i+result.offset] -= (*rhs.data)[i+rhs.offset];
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::multiply(const T &rhs) const {

    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(*this);

    // Multiply all elements by the scalar
    for (size_t i = 0; i < result.size(); ++i) {
        (*result.data)[i+result.offset] *= rhs;
    }

    return result;
}
//dot product
template <typename T>
Tensor<T> Tensor<T>::multiply(const Tensor<T> &rhs) const {
    //do matrix multiplication
    if (this->dims.size() == 2 && rhs.dims.size() == 2) {
        // Check if the dimensions match
        if (this->dims[1] != rhs.dims[0]) {
            throw std::invalid_argument("Dimensions do not match");
        }

        // Create a new Tensor object with the same data, dimensions, strides and offset
        Tensor<T> result({this->dims[0], rhs.dims[1]}, 0);

        // Multiply the elements of the other tensor
        for (size_t i = 0; i < this->dims[0]; ++i) {
            for (size_t j = 0; j < rhs.dims[1]; ++j) {
                for (size_t k = 0; k < this->dims[1]; ++k) {
                    (*result.data)[i * result.strides[0] + j * result.strides[1]+result.offset] +=
                            (*this->data)[i * this->strides[0] + k * this->strides[1]+this->offset] *
                            (*rhs.data)[k * rhs.strides[0] + j * rhs.strides[1]+rhs.offset];
                }
            }
        }

        return result;
    } else {
        // Check if the dimensions match
        if (this->dims != rhs.dims) {
            throw std::invalid_argument("Dimensions do not match");
        }

        // Create a new Tensor object with the same data, dimensions, strides and offset
        Tensor<T> result(*this);

        // Multiply the elements of the other tensor
        for (size_t i = 0; i < result.size(); ++i) {
            (*result.data)[i+result.offset] *= (*rhs.data)[i+rhs.offset];
        }

        return result;
    }
}

template <typename T>
Tensor<T> Tensor<T>::divide(const T &rhs) const {

    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(*this);

    // Divide all elements by the scalar
    for (size_t i = 0; i < result.size(); ++i) {
        (*result.data)[i+result.offset] /= rhs;
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::divide(const Tensor<T> &rhs) const {
    // Check if the dimensions match
    if (this->dims != rhs.dims) {
        throw std::invalid_argument("Dimensions do not match");
    }

    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(*this);

    // Divide the elements of the other tensor
    for (size_t i = 0; i < result.size(); ++i) {
        (*result.data)[i+result.offset] /= (*rhs.data)[i+rhs.offset];
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::log() const {

    // Create a new Tensor object with the same data, dimensions, strides and offset
    Tensor<T> result(*this);

    // Take the logarithm of all elements
    for (size_t i = 0; i < result.size(); ++i) {
        (*result.data)[i+result.offset] = std::log((*result.data)[i+result.offset]);
    }

    return result;
}

//sum: sum the elements to an lower order tensor on a given axis
// A function to reduce a tensor along a given dimension and sum the elements in that dimension
template<typename T>
Tensor<T> Tensor<T>::sum(int dim) const {
    // Check if the dimension is valid
    if (dim < 0 || dim >= this->order()) {
        throw std::out_of_range("Dimension out of range for sum operation");
    }

    // Determine the dimensions of the result tensor
    std::vector<size_t> result_dims = this->dims;
    result_dims.erase(result_dims.begin() + dim); // Remove the dimension being summed

    // Create a tensor to store the result
    Tensor<T> result(result_dims, T(0));

    // Iterate over the tensor and sum along the specified dimension
    std::vector<size_t> indices(this->order(), 0);
    do {
        // Calculate the linear index for the current element
        size_t linearIdx = this->linear_index(indices);

        // Calculate the linear index for the result tensor
        std::vector<size_t> result_indices = indices;
        result_indices.erase(result_indices.begin() + dim);
        size_t resultLinearIdx = result.linear_index(result_indices);

        // Sum the elements
        result.data->at(resultLinearIdx) += this->data->at(linearIdx + this->offset);

    } while (increment_indices(indices, this->dims));

    return result;
}
//max: maximum of the elements on a given axis
// A function to reduce a tensor along a given dimension and find the maximum element in that dimension
template<typename T>
Tensor<T> Tensor<T>::max(int dim) const {
    // Check if the dimension is valid
    if (dim < 0 || dim >= this->order()) {
        throw std::out_of_range("Dimension out of range for max operation");
    }

    // Determine the dimensions of the result tensor
    std::vector<size_t> result_dims = this->dims;
    result_dims.erase(result_dims.begin() + dim); // Remove the dimension being reduced

    // Create a tensor to store the result
    Tensor<T> result(result_dims, T(-2147483647));

    // Iterate over the tensor and find the maximum element along the specified dimension
    std::vector<size_t> indices(this->order(), 0);
    do {
        // Calculate the linear index for the current element
        size_t linearIdx = this->linear_index(indices);

        // Calculate the linear index for the result tensor
        std::vector<size_t> result_indices = indices;
        result_indices.erase(result_indices.begin() + dim);
        size_t resultLinearIdx = result.linear_index(result_indices);

        // Find the maximum element
        result.data->at(resultLinearIdx) = std::max(result.data->at(resultLinearIdx), this->data->at(linearIdx + this->offset));

    } while (increment_indices(indices, this->dims));

    return result;
}
//min: minimum of the elements on a given axis
// A function to reduce a tensor along a given dimension and find the minimum element in that dimension
template<typename T>
Tensor<T> Tensor<T>::min(int dim) const {
    // Check if the dimension is valid
    if (dim < 0 || dim >= this->order()) {
        throw std::out_of_range("Dimension out of range for min operation");
    }

    // Determine the dimensions of the result tensor
    std::vector<size_t> result_dims = this->dims;
    result_dims.erase(result_dims.begin() + dim); // Remove the dimension being reduced

    // Create a tensor to store the result
    Tensor<T> result(result_dims, T(2147483647));

    // Iterate over the tensor and find the minimum element along the specified dimension
    std::vector<size_t> indices(this->order(), 0);
    do {
        // Calculate the linear index for the current element
        size_t linearIdx = this->linear_index(indices);

        // Calculate the linear index for the result tensor
        std::vector<size_t> result_indices = indices;
        result_indices.erase(result_indices.begin() + dim);
        size_t resultLinearIdx = result.linear_index(result_indices);

        // Find the minimum element
        result.data->at(resultLinearIdx) = std::min(result.data->at(resultLinearIdx), this->data->at(linearIdx + this->offset));

    } while (increment_indices(indices, this->dims));

    return result;
}
//mean: mean value of the elements on a given axis
// A function to reduce a tensor along a given dimension and find the mean of the elements in that dimension
template<typename T>
Tensor<T> Tensor<T>::mean(int dim) const {
    // Check if the dimension is valid
    if (dim < 0 || dim >= this->order()) {
        throw std::out_of_range("Dimension out of range for mean operation");
    }

    // Determine the dimensions of the result tensor
    std::vector<size_t> result_dims = this->dims;
    result_dims.erase(result_dims.begin() + dim); // Remove the dimension being reduced

    // Create a tensor to store the result
    Tensor<T> result(result_dims, T(0));

    // Iterate over the tensor and find the mean of the elements along the specified dimension
    std::vector<size_t> indices(this->order(), 0);
    do {
        // Calculate the linear index for the current element
        size_t linearIdx = this->linear_index(indices);

        // Calculate the linear index for the result tensor
        std::vector<size_t> result_indices = indices;
        result_indices.erase(result_indices.begin() + dim);
        size_t resultLinearIdx = result.linear_index(result_indices);

        // Calculate the mean
        result.data->at(resultLinearIdx) += this->data->at(linearIdx + this->offset) / this->dims[dim];

    } while (increment_indices(indices, this->dims));

    return result;
}

//eq: compare two tensors
template<typename T>
Tensor<bool> Tensor<T>::eq(const Tensor<T> &rhs) const {
    // check if the tensors have the same dimensions
    if (this->dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(this->size());
    //compare the tensors element-wise
    for (size_t i = 0; i < this->size(); ++i) {
        result[i] = (bool)(this->values()[i] == rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(this->dimensions(), result);
    return result_tensor;
}

template<typename T>
Tensor<bool> Tensor<T>::ne(const Tensor<T> &rhs) const{
    // check if the tensors have the same dimensions
    if (this->dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(this->size());
    //compare the tensors element-wise
    for (size_t i = 0; i < this->size(); ++i) {
        result[i] = (bool)(this->values()[i] != rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(this->dimensions(), result);
    return result_tensor;
}

template<typename T>
Tensor<bool> Tensor<T>::gt(const Tensor<T> &rhs) const{
    // check if the tensors have the same dimensions
    if (this->dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(this->size());
    //compare the tensors element-wise
    for (size_t i = 0; i < this->size(); ++i) {
        result[i] = (bool)(this->values()[i] > rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(this->dimensions(), result);
    return result_tensor;
}

template<typename T>
Tensor<bool> Tensor<T>::lt(const Tensor<T> &rhs) const{
    // check if the tensors have the same dimensions
    if (this->dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(this->size());
    //compare the tensors element-wise
    for (size_t i = 0; i < this->size(); ++i) {
        result[i] = (bool)(this->values()[i] < rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(this->dimensions(), result);
    return result_tensor;
}

template<typename T>
Tensor<bool> Tensor<T>::ge(const Tensor<T> &rhs) const{
    // check if the tensors have the same dimensions
    if (this->dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(this->size());
    //compare the tensors element-wise
    for (size_t i = 0; i < this->size(); ++i) {
        result[i] = (bool)(this->values()[i] >= rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(this->dimensions(), result);
    return result_tensor;
}

template<typename T>
Tensor<bool> Tensor<T>::le(const Tensor<T> &rhs) const{
    // check if the tensors have the same dimensions
    if (this->dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(this->size());
    //compare the tensors element-wise
    for (size_t i = 0; i < this->size(); ++i) {
        result[i] = (bool)(this->values()[i] <= rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(this->dimensions(), result);
    return result_tensor;
}


// A helper function to increment a multi-index for iterating over a tensor
template<typename T>
bool Tensor<T>::increment_indices(std::vector<size_t> &indices, const std::vector<size_t> &dims) const {
    for (int i = indices.size() - 1; i >= 0; --i) {
        if (++indices[i] < dims[i]) {
            return true;
        }
        indices[i] = 0;
    }
    return false;
}