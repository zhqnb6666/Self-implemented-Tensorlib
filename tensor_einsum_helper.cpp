#include "tensor.h"
#include "tensor.cpp"





template<typename T>
std::vector<size_t> DetermineOutputShape(const std::vector<int>& free_indices_1,
                                         const std::vector<int>& free_indices_2,
                                         const Tensor<T>& tensor1,
                                         const Tensor<T>& tensor2) {
    std::vector<size_t> output_shape;

    // Add dimensions from tensor1 corresponding to its free indices
    for (int index : free_indices_1) {
        assert(index < tensor1.dimensions().size()); // Ensure valid index
        output_shape.push_back(tensor1.dimensions()[index]);
    }

    // Add dimensions from tensor2 corresponding to its free indices
    for (int index : free_indices_2) {
        assert(index < tensor2.dimensions().size()); // Ensure valid index
        output_shape.push_back(tensor2.dimensions()[index]);
    }

    return output_shape;
}

template<typename T>
bool increment_sum_indices(std::vector<size_t>& indices, const std::vector<int>& sum_indices, const std::vector<size_t>& dims) {
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



template<typename T>
Tensor<T> einsum_helper(const std::vector<int>& free_indices_1,
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
            sum += tensor1.at(indices1) * tensor2.at(indices2);
        } while (increment_sum_indices(indices1, sum_indices_1, tensor1.dims) && increment_sum_indices(indices2, sum_indices_2, tensor2.dims));

        // Set the computed sum in the output tensor
        output_tensor.set(output_indices, sum);

    } while (increment_indices(indices1, tensor1.dims) && increment_indices(indices2, tensor2.dims));

    return output_tensor;
}

