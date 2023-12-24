#ifndef TENSOR_EINSUM_H
#define TENSOR_EINSUM_H

#include "tensor.h"
#include "tensormath.h"
#include "tensor_bool.h"
#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>

using namespace std;

//目前残留问题。没有处理“ii->i”的特殊输入
//没有调用递归,einsum_help.cpp
//不确定输入tensor的个数限制

//用来存储一个vector的free indices等
struct IndicesStorage
{
    unordered_map<char, size_t> dimensionMap;
    vector<char> freeIndices;//free indices
    vector<char> sumIndices;  //sum indices
    vector<size_t> freeIndicesNum; //对应的 dimension数字
    vector<size_t> sumIndicesNum;
    vector<char> outputIndices; // output indices
    vector<size_t> outputIndicesNum; //
};

//用来存储两个vector
struct IndicesStorage2
{
    unordered_map<char, size_t> dimensionMap;//用来存储对应的char所对应的数字
    vector<char> freeIndices1;//第一个tensor的free indices
    vector<char> sumIndices1;//第一个tensor的summation indices
    vector<char> freeIndices2;//第二个tensor的free indices
    vector<char> sumIndices2;//第二个tensor的summation indices
    vector<char> outputIndices;
    vector<size_t> freeIndicesNum1;//第一个tensor free indices对应的dimension 大小
    vector<size_t> sumIndicesNum1;
    vector<size_t> freeIndicesNum2;
    vector<size_t> sumIndicesNum2;
    vector<size_t> outputIndicesNum;

};

//输入规范（"ijk,kmn->ijmn"）没有空格，一个tensor
template <typename T>
IndicesStorage analyzeEinsum(const std::string& einsumExpression, const Tensor<T>& tensor);

//输入规范（"ijk,kmn->ijmn"）没有空格,两个tensor
template <typename T>
IndicesStorage2 analyzeEinsum(const std::string& einsumExpression, const Tensor<T>& tensor1, const Tensor<T>& tensor2);

// Einsum function declaration with one tensor
template <typename T>
Tensor<T> einsum(const std::string& equation, const Tensor<T>& tensor);

// Einsum function declaration with one tensor
template <typename T>
Tensor<T> einsum(const std::string& equation, const Tensor<T>& tensor1, const Tensor<T>& tensor2);

#endif // TENSOR_EINSUM_H