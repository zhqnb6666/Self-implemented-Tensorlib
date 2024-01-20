#ifndef TENSOR_EINSUM_H
#define TENSOR_EINSUM_H

#include "tensor.h"
#include "tensor_math.h"

template<typename T>
std::vector<size_t> DetermineOutputShape(const std::vector<int>& free_indices_1,
                                         const std::vector<int>& free_indices_2,
                                         const Tensor<T>& tensor1,
                                         const Tensor<T>& tensor2);

bool increment_indices(std::vector<size_t>& indices, const std::vector<int>& sum_indices, const std::vector<size_t>& dims);

bool increment_sum_indices(std::vector<size_t>& indices1, const std::vector<int>& sum_indices1,
                           std::vector<size_t>& indices2, const std::vector<int>& sum_indices2,
                           const std::vector<size_t>& dims1, const std::vector<size_t>& dims2);

std::vector<size_t> MapToOutputIndices(const std::vector<size_t>& indices1,
                                       const std::vector<size_t>& indices2,
                                       const std::vector<int>& free_indices_1,
                                       const std::vector<int>& free_indices_2);

template<typename T>
Tensor<T> Inner_multiplication(const std::vector<int>& free_indices_1,
                        const std::vector<int>& free_indices_2,
                        const std::vector<int>& sum_indices_1,
                        const std::vector<int>& sum_indices_2,
                        const Tensor<T>& tensor1,
                        const Tensor<T>& tensor2);

template<typename T>
Tensor<T> batch_matrix_multiplication(const Tensor<T>& tensor1, const Tensor<T>& tensor2);

template<typename T>
Tensor<T> element_wise_multiplication(const Tensor<T>& tensor1, const Tensor<T>& tensor2);

template<typename T>
Tensor<T> sum_over_dimension(const Tensor<T>& tensor);

template<typename T>
Tensor<T> permute(const Tensor<T>& tensor, const std::vector<size_t>& permutation);

template<typename T>
Tensor<T> outer_product(const Tensor<T>& vec1, const Tensor<T>& vec2);

template<typename T>
Tensor<T> trace(const Tensor<T>& matrix);

template<typename T>
Tensor<T> dot_product(const Tensor<T>& vec1, const Tensor<T>& vec2);

template<typename T>
Tensor<T> diagonal(const Tensor<T>& matrix);

std::pair<std::string, std::string> parse_and_normalize_indices(const std::string& expression);

template<typename T>
Tensor<T> einsum_analyze(const std::string& normalized_input,
                         const std::string& normalized_output,
                         const Tensor<T>& tensor1,
                         const Tensor<T>& tensor2 = Tensor<T>());



template<typename T>
std::vector<size_t> DetermineOutputShape(const std::vector<int>& free_indices_1,
                                         const std::vector<int>& free_indices_2,
                                         const Tensor<T>& tensor1,
                                         const Tensor<T>& tensor2) {
    std::vector<size_t> output_shape;

    // Add dimensions from tensor1 corresponding to its free indices
    for (int index : free_indices_1) {
//        assert(index < tensor1.dimensions().size()); // Ensure valid index
        output_shape.push_back(tensor1.dimensions()[index]);
    }

    // Add dimensions from tensor2 corresponding to its free indices
    for (int index : free_indices_2) {
//        assert(index < tensor2.dimensions().size()); // Ensure valid index
        output_shape.push_back(tensor2.dimensions()[index]);
    }

    return output_shape;
}

bool increment_indices(std::vector<size_t>& indices, const std::vector<int>& sum_indices, const std::vector<size_t>& dims) {
    for (int i = sum_indices.size() - 1; i >= 0; --i) {
        int dim = sum_indices[i];
        if (dim < 0 || dim >= dims.size()) {
            throw std::out_of_range("Summation index out of range");
        }

        if (++indices[dim] < dims[dim]) {
            return true;
        }

        indices[dim] = 0;
    }
    return false;
}

bool increment_sum_indices(std::vector<size_t>& indices1, const std::vector<int>& sum_indices1,
                           std::vector<size_t>& indices2, const std::vector<int>& sum_indices2,
                           const std::vector<size_t>& dims1, const std::vector<size_t>& dims2) {
    for (int i = 0; i < sum_indices1.size(); ++i) {
        int idx1 = sum_indices1[i];
        int idx2 = sum_indices2[sum_indices2.size() - 1 - i]; // Counterpart index in the second tensor

        indices1[idx1]++;
        indices2[idx2]++;

        // Check if both indices are within their respective dimension ranges
        if (indices1[idx1] < dims1[idx1] && indices2[idx2] < dims2[idx2]) {
            return true; // Continue if within bounds
        }

        // Reset indices if they exceed dimensions and move to the next set of indices
        indices1[idx1] = (indices1[idx1] >= dims1[idx1]) ? 0 : indices1[idx1];
        indices2[idx2] = (indices2[idx2] >= dims2[idx2]) ? 0 : indices2[idx2];

        if (indices1[idx1] == 0 && indices2[idx2] == 0 && i == sum_indices1.size() - 1) {
            return false; // Completed a full cycle through all summation indices
        }
    }
    return false;
}



std::vector<size_t> MapToOutputIndices(const std::vector<size_t>& indices1,
                                       const std::vector<size_t>& indices2,
                                       const std::vector<int>& free_indices_1,
                                       const std::vector<int>& free_indices_2) {
    std::vector<size_t> output_indices;

    // Add indices from tensor1 corresponding to its free indices
    for (int index : free_indices_1) {
        output_indices.push_back(indices1[index]);
    }

    // Add indices from tensor2 corresponding to its free indices
    for (int index : free_indices_2) {
        output_indices.push_back(indices2[index]);
    }

    return output_indices;
}


template<typename T>
Tensor<T> Inner_multiplication(const std::vector<int>& free_indices_1,
                        const std::vector<int>& free_indices_2,
                        const std::vector<int>& sum_indices_1,
                        const std::vector<int>& sum_indices_2,
                        const Tensor<T>& tensor1,
                        const Tensor<T>& tensor2) {

    // Determine the shape of the output tensor
    std::vector<size_t> output_shape = DetermineOutputShape(free_indices_1, free_indices_2, tensor1, tensor2);
    Tensor<T> output_tensor(output_shape, T(0));

    // Initialize indices for tensor iteration
    std::vector<size_t> indices1(tensor1.order(), 0);
    std::vector<size_t> indices2(tensor2.order(), 0);

    do {
        // Compute indices for output tensor
        std::vector<size_t> output_indices = MapToOutputIndices(indices1, indices2, free_indices_1, free_indices_2);

        T sum = 0;
        do {
            sum += tensor1[indices1] * tensor2[indices2];
        } while (increment_sum_indices(indices1, sum_indices_1, indices2, sum_indices_2, tensor1.dimensions(), tensor2.dimensions()));

        // Set the computed sum in the output tensor
        output_tensor[output_indices] = sum;

    } while (increment_indices(indices1, free_indices_1,tensor1.dimensions()) || increment_indices(indices2,free_indices_2, tensor2.dimensions()));

    return output_tensor;
}

template<typename T>
Tensor<T> batch_matrix_multiplication(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
    // Assuming tensor1 is of shape [batch, M, N] and tensor2 is of shape [batch, N, K]
    // The result will be of shape [batch, M, K]

    if (tensor1.dimensions()[2] != tensor2.dimensions()[1]) {
        throw std::invalid_argument("Inner dimensions must match for batch matrix multiplication.");
    }

    size_t batch_size = tensor1.dimensions()[0];
    size_t M = tensor1.dimensions()[1];
    size_t N = tensor1.dimensions()[2];
    size_t K = tensor2.dimensions()[2];

    Tensor<T> result({batch_size, M, K}, T(0));

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t m = 0; m < M; ++m) {
            for (size_t k = 0; k < K; ++k) {
                T sum = T(0);
                for (size_t n = 0; n < N; ++n) {
                    sum += tensor1[{b, m, n}] * tensor2[{b, n, k}];
                }
                result[{b, m, k}] = sum;
            }
        }
    }

    return result;
}

template<typename T>
Tensor<T> element_wise_multiplication(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
    if (tensor1.dimensions() != tensor2.dimensions()) {
        throw std::invalid_argument("Dimensions must match for element-wise multiplication.");
    }

    Tensor<T> result(tensor1.dimensions(), T(0));

    for (size_t i = 0; i < tensor1.size(); ++i) {
        result[{i}] = tensor1[{i}] * tensor2[{i}];
    }

    return result;
}

template<typename T>
Tensor<T> sum_over_dimension(const Tensor<T>& tensor) {
    if (tensor.order() != 1) {
        throw std::invalid_argument("Tensor must be a vector (1D) for this operation.");
    }

    T totalSum = T(0);
    for (size_t i = 0; i < tensor.size(); ++i) {
        totalSum += tensor.data_ptr()[i];
    }

    return Tensor<T>({1}, totalSum);
}

template<typename T>
Tensor<T> permute(const Tensor<T>& tensor, const std::vector<int>& permutation) {
    if (tensor.order() != permutation.size()) {
        throw std::invalid_argument("Permutation must have the same size as the tensor order.");
    }
    return tensor.permute(permutation);
}

template<typename T>
Tensor<T> outer_product(const Tensor<T>& vec1, const Tensor<T>& vec2) {
    if (vec1.order() != 1 || vec2.order() != 1) {
        throw std::invalid_argument("Both tensors must be vectors (1D) for outer product.");
    }

    size_t N = vec1.size();
    size_t M = vec2.size();
    Tensor<T> result({N, M}, T(0));

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[{i, j}] = vec1.data_ptr()[i] * vec2.data_ptr()[j];
        }
    }

    return result;
}

template<typename T>
Tensor<T> trace(const Tensor<T>& matrix) {
    if (matrix.order() != 2 || matrix.dimensions()[0] != matrix.dimensions()[1]) {
        throw std::invalid_argument("Tensor must be a square matrix (2D with equal dimensions) for trace calculation.");
    }

    T traceSum = T(0);
    size_t N = matrix.dimensions()[0]; // Assuming square matrix

    for (size_t i = 0; i < N; ++i) {
        traceSum += matrix[{i, i}];
    }

    return Tensor<T>({1}, traceSum);
}

template<typename T>
Tensor<T> dot_product(const Tensor<T>& vec1, const Tensor<T>& vec2) {
    if (vec1.order() != 1 || vec2.order() != 1 || vec1.size() != vec2.size()) {
        throw std::invalid_argument("Both tensors must be vectors (1D) of the same size for dot product.");
    }

    T productSum = T(0);
    for (size_t i = 0; i < vec1.size(); ++i) {
        productSum += vec1.data_ptr()[i] * vec2.data_ptr()[i];
    }

    return Tensor<T>({1}, productSum);
}

template<typename T>
Tensor<T> diagonal(const Tensor<T>& matrix) {
    if (matrix.order() != 2 || matrix.dimensions()[0] != matrix.dimensions()[1]) {
        throw std::invalid_argument("Tensor must be a square matrix (2D with equal dimensions) for diagonal extraction.");
    }

    size_t N = matrix.dimensions()[0]; // Assuming square matrix
    Tensor<T> diag({N}, T(0));

    for (size_t i = 0; i < N; ++i) {
        diag[{i}] = matrix[{i, i}];
    }

    return diag;
}

std::pair<std::string, std::string> parse_and_normalize_indices(const std::string& expression) {
    size_t arrowPos = expression.find("->");
    std::string inputIndices = expression.substr(0, arrowPos);
    std::string outputIndices = arrowPos != std::string::npos ? expression.substr(arrowPos + 2) : "";

    std::unordered_map<char, int> indexMap;
    int currentIndex = 0;
    auto normalize = [&indexMap, &currentIndex](const std::string& indices) -> std::string {
        std::string normalized;
        for (char ch : indices) {
            if (ch != ',') {
                if (indexMap.find(ch) == indexMap.end()) {
                    indexMap[ch] = currentIndex++;
                }
                normalized += std::to_string(indexMap[ch]);
            }else {
                normalized += ",";
            }
        }
        return normalized.substr(0, normalized.size()); // Remove trailing comma
    };

    return {normalize(inputIndices), normalize(outputIndices)};
}

template<typename T>
Tensor<T> einsum_analyze(const std::string& normalized_input,
                         const std::string& normalized_output,
                         const Tensor<T>& tensor1,
                         const Tensor<T>& tensor2) {
    // Summation
    if (normalized_input == "0" && normalized_output.empty()) {
        return sum_over_dimension(tensor1);
    }

    // Element-wise Multiplication
    if (normalized_input == "0,0" && normalized_output == "0") {
        return element_wise_multiplication(tensor1, tensor2);
    }


    // Dot Product
    if (normalized_input == "0,0" && normalized_output.empty() ) {
        return dot_product(tensor1, tensor2);
    }

    // Outer Product
    if (normalized_input == "0,1" && normalized_output == "01") {
        return outer_product(tensor1, tensor2);
    }


    // Diagonal
    if (normalized_input == "00" && normalized_output == "0") {
        return diagonal(tensor1);
    }

    // Batch Matrix Multiplication
    if (normalized_input == "012,023" && normalized_output == "013") {
        return batch_matrix_multiplication(tensor1, tensor2);
    }

    // Trace
    if (normalized_input == "00" && normalized_output.empty()) {
        return trace(tensor1);
    }

    // Permutation (generalized case)
    if (normalized_input.length() == normalized_output.length()) {
        std::vector<int> permutation(normalized_input.length(), 0);
        for (size_t i = 0; i < normalized_input.length(); ++i) {
            size_t pos = normalized_output.find(normalized_input[i]);
            if (pos != std::string::npos) {
                permutation[pos] = i;
            } else {
                throw std::invalid_argument("Invalid permutation in einsum pattern.");
            }
        }
        return permute(tensor1, permutation);
    }
    // Find the position of the comma in the normalized input
    size_t commaPos = normalized_input.find(',');

    // Split the normalized input into two parts for tensor1 and tensor2
    std::string input1 = normalized_input.substr(0, commaPos);
    std::string input2 = normalized_input.substr(commaPos + 1);

    // Define free and summation indices for tensor1 and tensor2
    std::vector<int> free_indices_1, sum_indices_1, free_indices_2, sum_indices_2;

    // Process input1 (for tensor1)
    for (size_t i = 0; i < input1.length(); ++i) {
        if (normalized_output.find(input1[i]) != std::string::npos) {
            free_indices_1.push_back(i);
        } else {
            sum_indices_1.push_back(i);
        }
    }

    // Process input2 (for tensor2)
    for (size_t i = 0; i < input2.length(); ++i) {
        if (normalized_output.find(input2[i]) != std::string::npos) {
            free_indices_2.push_back(i);
        } else {
            sum_indices_2.push_back(i);
        }
    }

    // Call Inner_multiplication with the determined indices
    return Inner_multiplication(free_indices_1, free_indices_2, sum_indices_1, sum_indices_2, tensor1, tensor2);
}

template<typename T>
Tensor<T> einsum(const std::string& expression, const Tensor<T>& tensor1, const Tensor<T>& tensor2 = Tensor<T>()) {
    auto [normalized_input, normalized_output] = parse_and_normalize_indices(expression);
    return einsum_analyze(normalized_input, normalized_output, tensor1, tensor2);
}






#endif // TENSOR_EINSUM_H
//Summation: 'i->' sums over the i dimension, equivalent to a reduction or total sum if applied to a vector.
//
//Element-wise Multiplication: 'i,i->i' multiplies elements of two vectors element-wise.
//
//Matrix Multiplication: 'ij,jk->ik' represents matrix multiplication.
//
//Dot Product: 'i,i->' or 'i,i' calculates the dot product of two vectors.
//
//Outer Product: 'i,j->ij' computes the outer product of two vectors.
//
//Transpose: 'ij->ji' transposes a matrix.
//
//Diagonal: 'ii->i' extracts the diagonal of a matrix.
//
//Batch Matrix Multiplication: 'bij,bjk->bik' performs batched matrix multiplication over 3D tensors.
//
//Trace: 'ii' computes the trace of a matrix.
