//
// Created by HUAWEI on 2024/1/15.
//
#ifndef TENSOR_GRADIENT_H
#define TENSOR_GRADIENT_H
#include "tensor.h"
bool isOperator(char c);
int getPrecedence(char op);
string infixToPostfix(const string& infix);
double getFunctionValue(const string& postfix, double x, double y, double z);

Tensor<double> gradient(const Tensor<double>& t,const string& func);
#endif //TENSOR_GRADIENT_H
