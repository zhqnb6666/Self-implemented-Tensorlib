// Purpose: Header file for tensormath.cpp
#ifndef TENSORMATH_H
#define TENSORMATH_H
#include <functional>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "tensor.h"
#include "tensor_bool.h"
#include <cmath>
using namespace std;

// add two tensors
template <typename T>
// overload operator +
Tensor<T> operator+(const Tensor<T> &lhs, const Tensor<T> &rhs);
// add a tensor and a scalar
template <typename T>
// overload operator +
Tensor<T> operator+(const Tensor<T> &lhs, const T &rhs);
template <typename T>
// add as a class function
Tensor<T> add(const Tensor<T> &lhs, const Tensor<T> &rhs);
template<typename T>
Tensor<T> add(const Tensor<T> &lhs, const T &rhs);
template<typename T>
Tensor<T> add(const T &lhs, const Tensor<T> &rhs);
// subtract two tensors
template <typename T>
// overload operator -
Tensor<T> operator-(const Tensor<T> &lhs, const Tensor<T> &rhs);
//subtract a tensor and a scalar
template <typename T>
// overload operator -
Tensor<T> operator-(const Tensor<T> &lhs, const T &rhs);
// subtract as a class function
template <typename T>
Tensor<T> subtract(const Tensor<T> &lhs, const Tensor<T> &rhs);
template <typename T>
Tensor<T> subtract(const Tensor<T> &lhs, const T &rhs);
template <typename T>
Tensor<T> subtract(const T &lhs, const Tensor<T> &rhs);


// multiply two tensors
template <typename T>
// overload operator *
Tensor<T> operator*(const Tensor<T> &lhs, const Tensor<T> &rhs);
// multiply a tensor and a scalar
template <typename T>
// overload operator *
Tensor<T> operator*(const Tensor<T> &lhs, const T &rhs);
// multiply as a member function
template <typename T>
Tensor<T> multiply(const Tensor<T> &lhs, const Tensor<T> &rhs);

template <typename T>
Tensor<T> multiply(const Tensor<T> &lhs, const T &rhs);

template <typename T>
Tensor<T> multiply(const T &lhs, const Tensor<T> &rhs);

// divide two tensors
template <typename T>
// overload operator /
Tensor<T> operator/(const Tensor<T> &lhs , const Tensor<T> &rhs);
// divide a tensor and a scalar
template <typename T>
// overload operator /
Tensor<T> operator/(const Tensor<T> &lhs , const T &rhs);
// divide as a member function
template <typename T>
Tensor<T> divide(const Tensor<T> &lhs, const Tensor<T> &rhs);
// divide as a class function
template <typename T>
Tensor<T> divide(const Tensor<T> &lhs, const T &rhs);
template <typename T>
Tensor<T> divide(const T &lhs, const Tensor<T> &rhs);

//logarithm of a tensor as a member function
template <typename T>
Tensor<T> log(const Tensor<T> &lhs);

//sum: sum of the elements on a given axis
template <typename T>
Tensor<T> sum(const Tensor<T> &lhs,const size_t &axis);

//max: maximum of the elements on a given axis
template <typename T>
Tensor<T> max(const Tensor<T> &lhs,const size_t &axis);

//min: minimum of the elements on a given axis
template <typename T>
Tensor<T> min(const Tensor<T> &lhs,const size_t &axis);

//mean: mean of the elements on a given axis
template <typename T>
Tensor<T> mean(const Tensor<T> &lhs,const size_t &axis);

//compare two tensors
template <typename T>
Tensor<bool> eq(const Tensor<T> &lhs, const Tensor<T> &rhs);
template <typename T>
Tensor<bool> operator==(const Tensor<T> &lhs, const Tensor<T> &rhs);

template <typename T>
Tensor<bool> ne(const Tensor<T> &lhs, const Tensor<T> &rhs);
template <typename T>
Tensor<bool> operator!=(const Tensor<T> &lhs, const Tensor<T> &rhs);

template <typename T>
Tensor<bool> lt(const Tensor<T> &lhs, const Tensor<T> &rhs);
template <typename T>
Tensor<bool> operator<(const Tensor<T> &lhs, const Tensor<T> &rhs);

template <typename T>
Tensor<bool> gt(const Tensor<T> &lhs, const Tensor<T> &rhs);
template <typename T>
Tensor<bool> operator>(const Tensor<T> &lhs, const Tensor<T> &rhs);

template <typename T>
Tensor<bool> le(const Tensor<T> &lhs, const Tensor<T> &rhs);
template <typename T>
Tensor<bool> operator<=(const Tensor<T> &lhs, const Tensor<T> &rhs);

template <typename T>
Tensor<bool> ge(const Tensor<T> &lhs, const Tensor<T> &rhs);
template <typename T>
Tensor<bool> operator>=(const Tensor<T> &lhs, const Tensor<T> &rhs);

#endif //TENSORMATH_H