#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "tensor_einsum.h"
#include "tensor_einsum.cpp"

int main() {
    std::string einsumExpr = "ij->i";
    Tensor<int> tensor1({1,3,3}, {1, 2, 3, 4,5,6,7,8,9});
    Tensor<int> tensor2({2, 3}, {1, 2, 3, 4,5,6});
    IndicesStorage indicesStorage = analyzeEinsum(einsumExpr, tensor1);
    IndicesStorage2 indicesStorage2 = analyzeEinsum("iaj,kj->kia",tensor1,tensor2);
    return 0;
}
