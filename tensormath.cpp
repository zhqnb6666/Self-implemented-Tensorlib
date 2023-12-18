#include "tensormath.h"
#include "tensor.h"
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

//sum: sum the elements to an lower order tensor on a given axis
template<typename T>
Tensor<T> sum(const Tensor<T> &lhs, size_t axis) {
        if (axis >= lhs.order()) {
        throw std::invalid_argument("Invalid axis!");
    }
//find the shape of the result
    vector<size_t> result_shape;
    for (size_t i = 0; i < lhs.order(); ++i) {
        if (i == axis) {
            result_shape.push_back(1);
        }else{
            result_shape.push_back(lhs.dimensions()[i]);
        }
    }
    //create a tensor to store the result
    Tensor<T> result(result_shape);
    //sum the tensor's elements along the given axis
    for (size_t i = 0; i < lhs.size(); ++i) {
        //sum the elements along the given axis
        result.data_ptr()[i % result.size()] += lhs.values()[i];
    }

    return result;
}

//mean: mean of the elements on a given axis


//max: maximum of the elements on a given axis


//min: minimum of the elements on a given axis


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

/*
void test_equal(){
    Tensor<int> t1({2, 2}, {1, 2, 3, 4});
    Tensor<int> t2({2, 2}, {1, 2, 3, 4});
    Tensor<int> t3({2, 2}, {1, 2, 3, 5});
    Tensor<bool> t4 = eq(t1,t2);
    Tensor<bool> t5 = ne(t1,t3);
    Tensor<bool> t6 = gt(t1,t3);
    Tensor<bool> t7 = lt(t1,t3);
    Tensor<bool> t8 = ge(t1,t3);
    Tensor<bool> t9 = le(t1,t3);
    t4.print();
    t5.print();
    t6.print();
    t7.print();
    t8.print();
    t9.print();
}

*/