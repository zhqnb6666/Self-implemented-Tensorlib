//
// Created by HUAWEI on 2024/1/15.
//
#ifndef TENSOR_GRADIENT_H
#define TENSOR_GRADIENT_H
#include "tensor.h"
#include "string"
#include <stack>
#include <queue>
#include <string>
#include <unordered_map>
#include <iostream>
using namespace std;

//infix to postfix
bool isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/';
}
int getPrecedence(char op) {
    std::unordered_map<char, int> precedence;
    precedence['+'] = 1;
    precedence['-'] = 1;
    precedence['*'] = 2;
    precedence['/'] = 2;
    return precedence[op];
}
string infixToPostfix(const string& infix) {
    string postfix;
    stack<char> operators;

    for (char c : infix) {
        if (isalnum(c)) {
            postfix += c;
        } else if (c == '(') {
            operators.push(c);
        } else if (c == ')') {
            while (!operators.empty() && operators.top() != '(') {
                postfix += operators.top();
                operators.pop();
            }
            operators.pop(); // Pop '('
        } else if (isOperator(c)) {
            while (!operators.empty() && getPrecedence(c) <= getPrecedence(operators.top())) {
                postfix += operators.top();
                operators.pop();
            }
            operators.push(c);
        }
    }

    while (!operators.empty()) {
        postfix += operators.top();
        operators.pop();
    }

    return postfix;
}
//postfix to value
double getFunctionValue(const string& postfix, double x, double y, double z) {
    stack<double> values;
    for (char c : postfix) {
        if (isalnum(c)) {
            if (c == 'x') {
                values.push(x);
            } else if (c == 'y') {
                values.push(y);
            } else if (c == 'z') {
                values.push(z);
            } else {
                values.push(c - '0');
            }
        } else if (isOperator(c)) {
            double val1 = values.top();
            values.pop();
            double val2 = values.top();
            values.pop();
            if (c == '+') {
                values.push(val2 + val1);
            } else if (c == '-') {
                values.push(val2 - val1);
            } else if (c == '*') {
                values.push(val2 * val1);
            } else if (c == '/') {
                values.push(val2 / val1);
            }
        }
    }
    return values.top();
}

template <typename T>
Tensor<double> gradient(const Tensor<T>& t,const string& func) {
    //the tensor is a vector, so the gradient is a vector
    //the vector maybe (x), (x,y), (x,y,z),only consider this three situation
    //the func is a string, like "x*2", "x+y", "x*y", "x+y+z"
    //calculate the gradient at the point t, return a vector
    //use difference quotient to calculate the gradient
    //the difference quotient is (f(x+dx)-f(x))/dx

    //copy the value of t to a vector
    double dx = 0.01;
    double dy = 0.01;
    double dz = 0.01;
    vector<T> value;
    int size = t.size();
    for (int i = 0; i < t.size(); i++) {
        T temp = t.values()[i]; //原本是T temp = t.[i]; data or dimension???
        value.push_back(temp);
    }
    while (value.size() < 3) {
        value.push_back(0.0);
    }
    string postfix = infixToPostfix(func);
    double x = value[0];
    double y = value[1];
    double z = value[2];
    double f = getFunctionValue(postfix, x, y, z);
    double dfx = getFunctionValue(postfix, x + dx, y, z);
    double dfy = getFunctionValue(postfix, x, y + dy, z);
    double dfz = getFunctionValue(postfix, x, y, z + dz);

    vector<double> result;
    if(size==1){
        result.push_back((dfx-f)/dx);
        return Tensor<double>({1,1},result);
    }else if(size==2){
        result.push_back((dfx-f)/dx);
        result.push_back((dfy-f)/dy);
        return Tensor<double>({1,2},result);
    }else if(size==3){
        result.push_back((dfx-f)/dx);
        result.push_back((dfy-f)/dy);
        result.push_back((dfz-f)/dz);
        return Tensor<double>({1,3},result);
    }
    std::cout << "size out of boundary" << std::endl;
    return Tensor<double>();  //won't reach here, but in consideration of compilation
}
#endif //TENSOR_GRADIENT_H
