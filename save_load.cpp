//
// Created by Lenovo on 2023/12/21.
//
#include <fstream>
#include "tensor.h"

    template<typename T>
    Tensor<T> load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::in);

        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file for reading");
        }

        // Read dimensions
        size_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));
        std::vector<size_t> dimensions(num_dims);
        file.read(reinterpret_cast<char*>(dimensions.data()), sizeof(size_t) * num_dims);

        // Read strides
        std::vector<size_t> strides(num_dims);
        file.read(reinterpret_cast<char*>(strides.data()), sizeof(size_t) * num_dims);

        // Read offset
        size_t offset;
        file.read(reinterpret_cast<char*>(&offset), sizeof(offset));

        // Read data
        //calculate the size of data
        size_t data_size = 1;
        for (size_t i = 0; i < num_dims; i++) {
            data_size *= dimensions[i];
        }
        std::vector<T> data(data_size);
        file.read(reinterpret_cast<char*>(data.data()), sizeof(T) * data_size);

        file.close();

        return Tensor<T>(dimensions, data, strides, offset);
    }




    template<typename T>
    void save(const Tensor<T>& tensor, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary | std::ios::out);

        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file for writing");
        }

        // Write dimensions
        size_t num_dims = tensor.dimensions().size();
        file.write(reinterpret_cast<const char*>(&num_dims), sizeof(num_dims));
        file.write(reinterpret_cast<const char*>(tensor.dimensions().data()), sizeof(size_t) * num_dims);

        // Write strides
        file.write(reinterpret_cast<const char*>(tensor.getstrides().data()), sizeof(size_t) * num_dims);

        // Write offset
        size_t offset = tensor.getoffset();
        file.write(reinterpret_cast<const char*>(&offset), sizeof(offset));

        // Write data
        size_t data_size = tensor.size();
        file.write(reinterpret_cast<const char*>(tensor.data_ptr()), sizeof(T) * data_size);

        file.close();
    }



