#include "tensor_einsum.h"
#include "tensor.h"
#include "tensor.cpp"
#include "iostream"
// perfect explanation in yutube: https://www.youtube.com/watch?v=pkVwUVEHmfI&list=WL&index=1
// explanation: free indices, summation indices

//form the indicesStorage for one tensor
template <typename T>
IndicesStorage analyzeEinsum(const std::string& einsumExpression, const Tensor<T>& tensor) {
    IndicesStorage indicesStorage;
    size_t arrowPos = einsumExpression.find("->");
    std::string inputIndices = einsumExpression.substr(0, arrowPos);
    std::string outputIndices = einsumExpression.substr(arrowPos + 2);
    
    // 存储每个索引对应的维度大小
    for (size_t i = 0; i < inputIndices.length(); ++i) {
        //检查是否已经出现过该字母，如果出现，检查对应dimension大小是否相等
        if (indicesStorage.dimensionMap.find(inputIndices[i]) != indicesStorage.dimensionMap.end()
            && (indicesStorage.dimensionMap[inputIndices[i]] != tensor.dimensions()[i]))
        {
            throw std::invalid_argument("same char but different value");
        }
        //赋值该map
        indicesStorage.dimensionMap[inputIndices[i]] = tensor.dimensions()[i];
    }

    //输入 输出的tensor 对应的vector<char>
    //其实output和free是一样的，但是为了两个tensor方法的类似性，命名output
    for (char c : outputIndices) {
        if (inputIndices.find(c) == std::string::npos) {
            indicesStorage.outputIndices.push_back(c);
            indicesStorage.outputIndicesNum.push_back(indicesStorage.dimensionMap[c]);
    }
    }

    // 确定自由索引和求和索引
    for (char c : inputIndices) {
        if (outputIndices.find(c) == std::string::npos) {
            indicesStorage.sumIndices.push_back(c);
            indicesStorage.sumIndicesNum.push_back(indicesStorage.dimensionMap[c]);
        } else {
            indicesStorage.freeIndices.push_back(c);
            indicesStorage.freeIndicesNum.push_back(indicesStorage.dimensionMap[c]);
        }
    }


    // 打印结果，用于验证
    cout << "!!!one tensor indicesStorage:\n";
    std::cout << "Map(char,size_t): ";
    for (auto& pair : indicesStorage.dimensionMap) {
        std::cout << pair.first << ": " << pair.second << ", ";
    }
    std::cout << "\nFree indices: ";
    for (int i = 0; i <indicesStorage.freeIndices.size(); i ++) {
        std::cout << indicesStorage.freeIndices[i] << ": " << indicesStorage.freeIndicesNum[i] << " ";
    }
    std::cout << "\nSummation indices: ";
    for (int i = 0; i <indicesStorage.sumIndices.size(); i ++) {
        std::cout << indicesStorage.sumIndices[i] << ":  " << indicesStorage.sumIndicesNum[i] << " ";
    }
    std::cout << "\noutput indices: ";
    for (int i = 0; i <indicesStorage.outputIndices.size(); i ++) {
        std::cout << indicesStorage.outputIndices[i] << ":  " << indicesStorage.outputIndicesNum[i] << " ";
    }
    std::cout << std::endl;

    return indicesStorage;
}

//form the indicesStorage2 for two tensors
template <typename T>
IndicesStorage2 analyzeEinsum(const std::string& einsumExpression, const Tensor<T>& tensor1, const Tensor<T>& tensor2 ) {
    IndicesStorage2 indicesStorage2;
    size_t arrowPos = einsumExpression.find("->");
    size_t commaPose = einsumExpression.find(",");
    string inputIndices1 = einsumExpression.substr(0,commaPose);
    string inputIndices2 = einsumExpression.substr(commaPose + 1,arrowPos - commaPose - 1);
    string outputIndices = einsumExpression.substr(arrowPos + 2);
    cout << "arrowPos: " << arrowPos << endl;
    cout << "commaPose: " << commaPose << endl;
    cout << "inputIndices1: " << inputIndices1 << endl;
    cout << "inputIndices2: " << inputIndices2 << endl;
    cout << "outputIndices: " << outputIndices<< endl;


    //处理第1个tensor的维度，将对应的char和dimension大小存储在map里面
    for (size_t i = 0; i < inputIndices1.length(); ++i) {
        if ((indicesStorage2.dimensionMap.find(inputIndices1[i]) != indicesStorage2.dimensionMap.end())
            && (indicesStorage2.dimensionMap[inputIndices1[i]] != tensor1.dimensions()[i]) )
            {
               throw std::invalid_argument("unmatched dimension1"); 
            }  
        indicesStorage2.dimensionMap[inputIndices1[i]] = tensor1.dimensions()[i];
    }   
    
    //处理第2个tensor的维度，将对应的char和dimension大小存储在map里面
    for (size_t i = 0; i < inputIndices2.length(); ++i) {
        if ((indicesStorage2.dimensionMap.find(inputIndices2[i]) == indicesStorage2.dimensionMap.end())
            && (indicesStorage2.dimensionMap[inputIndices2[i]] == tensor2.dimensions()[i]) )
            {
               throw std::invalid_argument("unmatched dimension2"); 
            }  
        indicesStorage2.dimensionMap[inputIndices2[i]] = tensor2.dimensions()[i];
    }

    //处理output
    for (char c : outputIndices)
    {
        if ((inputIndices1.find(c) == string::npos) &&(inputIndices2.find(c) == string::npos))
        {
            throw std::invalid_argument("no corresponding dimenison found in input");
        }
        indicesStorage2.outputIndices.push_back(c);
        indicesStorage2.outputIndicesNum.push_back(indicesStorage2.dimensionMap[c]);
    }

    //处理input indices 1 生成 tensor1的 freeindice1 num1
    for (char c : inputIndices1)
    {
        if (outputIndices.find(c) == std::string::npos) {
            indicesStorage2.sumIndices1.push_back(c);
            indicesStorage2.sumIndicesNum1.push_back(indicesStorage2.dimensionMap[c]);
        } else {
            indicesStorage2.freeIndices1.push_back(c);
            indicesStorage2.freeIndicesNum1.push_back(indicesStorage2.dimensionMap[c]);
        }
    }

    //处理input indices 2 生成 tensor2的 freeindice2 num2
    for (char c : inputIndices2)
    {
        if (outputIndices.find(c) == std::string::npos) {
            indicesStorage2.sumIndices2.push_back(c);
            indicesStorage2.sumIndicesNum2.push_back(indicesStorage2.dimensionMap[c]);
        } else {
            indicesStorage2.freeIndices2.push_back(c);
            indicesStorage2.freeIndicesNum2.push_back(indicesStorage2.dimensionMap[c]);
        }
    }

    //打印出来
    cout << "!!!two tensor indicesStorage2:\n";
    std::cout << "Map(char,size_t): ";
    for (auto& pair : indicesStorage2.dimensionMap) {
        std::cout << pair.first << ": " << pair.second << ", ";
    }
    std::cout << "\nFree indices 1: ";
    for (int i = 0; i <indicesStorage2.freeIndices1.size(); i ++) {
        std::cout << indicesStorage2.freeIndices1[i] << ": " << indicesStorage2.freeIndicesNum1[i] << " ";
    }
    std::cout << "\nSummation indices 1: ";
    for (int i = 0; i <indicesStorage2.sumIndices1.size(); i ++) {
        std::cout << indicesStorage2.sumIndices1[i] << ":  " << indicesStorage2.sumIndicesNum1[i] << " ";
    }
    std::cout << "\nFree indices 2: ";
    for (int i = 0; i <indicesStorage2.freeIndices2.size(); i ++) {
        std::cout << indicesStorage2.freeIndices2[i] << ": " << indicesStorage2.freeIndicesNum2[i] << " ";
    }
    std::cout << "\nSummation indices 2: ";
    for (int i = 0; i <indicesStorage2.sumIndices2.size(); i ++) {
        std::cout << indicesStorage2.sumIndices2[i] << ":  " << indicesStorage2.sumIndicesNum2[i] << " ";
    }
    std::cout << "\noutput indices: ";
    for (int i = 0; i <indicesStorage2.outputIndices.size(); i ++) {
        std::cout << indicesStorage2.outputIndices[i] << ":  " << indicesStorage2.outputIndicesNum[i] << " ";
    }
    std::cout << std::endl;
    
    return indicesStorage2;

}

// Einsum function implementation for one tensor
template <typename T>
Tensor<T> einsum(const std::string& equation, const Tensor<T>& tensor) {//dealing with one input
    IndicesStorage indicesStorage = analyzeEinsum(equation,tensor);
    Tensor result(indicesStorage.outputIndicesNum);

    //调用递归，进行计算

    return result;
}

// Einsum function implementation for two tensor
template <typename T>
Tensor<T> einsum(const std::string& equation, const Tensor<T>& tensor1, const Tensor<T>& tensor2) {//dealing with one input
    IndicesStorage indicesStorage2 = analyzeEinsum(equation,tensor1,tensor2);
    Tensor result(indicesStorage2.outputIndicesNum);

    //调用递归，进行计算
    
    return result;
}





