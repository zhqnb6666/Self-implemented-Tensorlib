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

template <typename T>
Tensor<T> gradient(const Tensor<T>& t, string func) {
    return Tensor<T>();
}

unordered_map<char, int> op_priority = {
    {'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}, {'(', 0}
};

bool isOperator(char c) {
    return op_priority.count(c) > 0;
}

string infixToPostfix(string infix) {
    stack<char> op_stack;
    queue<char> postfix_queue;

    for (char c : infix) {
        if (isOperator(c)) {
            while (!op_stack.empty() && op_priority[op_stack.top()] >= op_priority[c] && op_stack.top() != '(') {
                postfix_queue.push(op_stack.top());
                op_stack.pop();
            }
            op_stack.push(c);
        } else if (c == '(') {
            op_stack.push(c);
        } else if (c == ')') {
            while (op_stack.top() != '(') {
                postfix_queue.push(op_stack.top());
                op_stack.pop();
            }
            op_stack.pop();
        } else {
            postfix_queue.push(c);
        }
    }

    while (!op_stack.empty()) {
        postfix_queue.push(op_stack.top());
        op_stack.pop();
    }

    string postfix;
    while (!postfix_queue.empty()) {
        postfix += postfix_queue.front();
        postfix_queue.pop();
    }

    return postfix;
}
#endif //TENSOR_GRADIENT_H
