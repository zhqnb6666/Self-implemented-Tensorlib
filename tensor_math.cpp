#include "tensor_math.h"
#include "tensor.h"
#include "tensor_bool.h"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sched.h>
#include <vector>
#include <stdexcept>
using namespace std;
// mutiplication: currently only support matrix multiplication;
// reduction: currently only support opreation on the axis 0;
// comparison: support element-wise comparison;

// add two tensors
// overload operator +
template<typename T>
Tensor<T> operator+(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }

    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());
    vector<T> lhs_values = lhs.values();
    vector<T> rhs_values = rhs.values();

    // add the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = lhs_values[i] + rhs_values[i];
    }

    return result;
}
// add a tensor and a scalar
// overload operator +
template<typename T>
Tensor<T> operator+(const Tensor<T> &lhs, const T &rhs) {
    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());

    vector<T> lhs_values = lhs.values();
    // add the tensor and the scalar element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = lhs_values[i] + rhs;
    }

    return result;
}
// add as a class function
template<typename T>
Tensor<T> add(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }

    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());

    // add the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = lhs.values()[i] + rhs.values()[i];
    }

    return result;
}

// subtract two tensors
// overload operator -
template<typename T>
Tensor<T> operator-(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }

    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());

    // subtract the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = lhs.values()[i] - rhs.values()[i];
    }

    return result;
}

// subtract a tensor and a scalar
// overload operator -
template<typename T>
Tensor<T> operator-(const Tensor<T> &lhs, const T &rhs) {
    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());

    // subtract the tensor and the scalar element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = lhs.values()[i] - rhs;
    }

    return result;
}

// subtract as a class function
template<typename T>
Tensor<T> subtract(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }

    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());

    // subtract the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = lhs.values()[i] - rhs.values()[i];
    }

    return result;
}

// multiply two tensors
// overload operator *
template<typename T>
Tensor<T> operator*(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // do dot product if the tensors are matrices
    if (lhs.order() == 2 && rhs.order() == 2) {
        // check if the tensors have the same dimensions
        if (lhs.dimensions()[1] != rhs.dimensions()[0]) {
            throw std::invalid_argument("Dimensions mismatch!");
        }

        // create a tensor to store the result
        Tensor<T> result({lhs.dimensions()[0], rhs.dimensions()[1]});

        // multiply the tensors element-wise
        for (size_t i = 0; i < lhs.dimensions()[0]; ++i) {
            for (size_t j = 0; j < rhs.dimensions()[1]; ++j) {
                for (size_t k = 0; k < lhs.dimensions()[1]; ++k) {
                    result[{i, j}] += lhs[{i, k}] * rhs[{k, j}];
                }
            }
        }
        return result;
    }else{
        return Tensor<T>();
    }
}
// multiply as a class function
// currently only support matrix multiplication
template<typename T>
Tensor<T> multiply(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // do dot product if the tensors are matrices
    if (lhs.order() == 2 && rhs.order() == 2) {
        // check if the tensors have the same dimensions
        if (lhs.dimensions()[1] != rhs.dimensions()[0]) {
            throw std::invalid_argument("Dimensions mismatch!");
        }

        // create a tensor to store the result
        Tensor<T> result({lhs.dimensions()[0], rhs.dimensions()[1]});

        // multiply the tensors element-wise
        for (size_t i = 0; i < lhs.dimensions()[0]; ++i) {
            for (size_t j = 0; j < rhs.dimensions()[1]; ++j) {
                for (size_t k = 0; k < lhs.dimensions()[1]; ++k) {
                    result[{i, j}] += lhs[{i, k}] * rhs[{k, j}];
                }
            }
        }
        return result;
    }else{
        return Tensor<T>();
    }
}
// multiply a tensor and a scalar
// overload operator *
template<typename T>
Tensor<T> operator*(const Tensor<T> &lhs, const T &rhs) {
    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());

    // multiply the tensor and the scalar element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = lhs.values()[i] * rhs;
    }

    return result;
}

// divide two tensors
// overload operator /
template<typename T>
Tensor<T> operator/(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }

    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());

    // divide the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = lhs.values()[i] / rhs.values()[i];
    }

    return result;
}

// divide a tensor and a scalar
// overload operator /
template<typename T>
Tensor<T> operator/(const Tensor<T> &lhs, const T &rhs) {
    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());

    // divide the tensor and the scalar element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = lhs.values()[i] / rhs;
    }

    return result;
}

// divide as a class function
template<typename T>
Tensor<T> divide(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }

    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());

    // divide the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = lhs.values()[i] / rhs.values()[i];
    }

    return result;
}
template<typename T>
Tensor<T> divide(const Tensor<T> &lhs, const T &rhs) {
    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());

    // divide the tensor and the scalar element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = lhs.values()[i] / rhs;
    }

    return result;
}

//log as a class function
template<typename T>
Tensor<T> log(const Tensor<T> &lhs) {
    // create a tensor to store the result
    Tensor<T> result(lhs.dimensions());

    // log the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result.data_ptr()[i] = log10(lhs.values()[i]);
    }

    return result;
}
//sun: sum of the elements on a given axis
template<typename T>
Tensor<T> sum(const Tensor<T> &lhs, const size_t &axis) {
    return lhs.sum(axis);
}
//mean: mean of the elements on a given axis
template<typename T>
Tensor<T> mean(const Tensor<T> &lhs, const size_t &axis) {
    return lhs.mean(axis);
}
//max: maximum of the elements on a given axis
template<typename T>
Tensor<T> max(const Tensor<T> &lhs, const size_t &axis) {
    return Tensor<T>::lhs.max(axis);
}
//min: minimum of the elements on a given axis
template<typename T>
Tensor<T> min(const Tensor<T> &lhs, const size_t &axis) {
    return Tensor<T>::lhs.min(axis);
}
//eq: compare two tensors
template<typename T>
Tensor<bool> eq(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(lhs.size());
    //compare the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = (bool)(lhs.values()[i] == rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(lhs.dimensions(), result);
    return result_tensor;
}
//ne: compare two tensors
template<typename T>
Tensor<bool> ne(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(lhs.size());
    //compare the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = (bool)(lhs.values()[i] != rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(lhs.dimensions(), result);
    return result_tensor;
}
//gt: compare two tensors
template<typename T>
Tensor<bool> gt(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(lhs.size());
    //compare the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = (bool)(lhs.values()[i] > rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(lhs.dimensions(), result);
    return result_tensor;
}
//lt: compare two tensors
template<typename T>
Tensor<bool> lt(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(lhs.size());
    //compare the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = (bool)(lhs.values()[i] < rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(lhs.dimensions(), result);
    return result_tensor;
}
//ge: compare two tensors
template<typename T>
Tensor<bool> ge(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(lhs.size());
    //compare the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = (bool)(lhs.values()[i] >= rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(lhs.dimensions(), result);
    return result_tensor;
}
//le: compare two tensors
template<typename T>
Tensor<bool> le(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    // check if the tensors have the same dimensions
    if (lhs.dimensions() != rhs.dimensions()) {
        throw std::invalid_argument("Dimensions mismatch!");
    }
    //first create a data vector to store the result
    vector<bool> result(lhs.size());
    //compare the tensors element-wise
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = (bool)(lhs.values()[i] <= rhs.values()[i]);
    }
    //then create a tensor to store the result
    Tensor<bool> result_tensor(lhs.dimensions(), result);
    return result_tensor;
}