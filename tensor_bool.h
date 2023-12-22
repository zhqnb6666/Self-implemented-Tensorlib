#pragma once

#include "tensor.h"
template<>
class Tensor<bool> {
private:
    std::shared_ptr<std::vector<char>> data;
    std::vector<size_t> dims;
    std::vector<size_t> strides;
    size_t offset = 0;

    size_t linear_index(const std::vector<size_t> &indices) const;

public:
    Tensor();
    Tensor(const std::vector<size_t> &dimensions, const std::vector<bool> &values);
    explicit Tensor(const std::vector<size_t> &dimensions, bool value = false);
    Tensor(const Tensor<bool> &other);
    Tensor(const std::vector<size_t> &dimensions, std::shared_ptr<std::vector<char>> values);
    Tensor(const std::vector<size_t> &dimensions, std::shared_ptr<std::vector<char>> values, size_t offset);
    Tensor(Tensor<bool> &other);

    Tensor<bool> &operator=(const Tensor<bool> &other);
    Tensor<bool> &operator=(Tensor<bool> &other);
    Tensor<bool> &operator=(bool value);
    Tensor<bool> &operator=(const std::vector<bool> &values);

    ~Tensor();

    static Tensor<bool> *rand(const std::vector<size_t> &dimensions);
    static Tensor<bool> *Zeros(const std::vector<size_t> &dimensions);
    static Tensor<bool> *Ones(const std::vector<size_t> &dimensions);
    static Tensor<bool> *eye(size_t n);

    size_t order() const;
    size_t size() const;
    std::vector<size_t> dimensions() const;
    char *data_ptr();
    string type() const;
    std::vector<char>& values() const;

    void print(std::ostream &os, const std::vector<size_t> &indices, size_t dim) const;
    void print() const;

    friend std::ostream &operator<<(std::ostream &os, const Tensor<bool> &tensor);

    bool operator[](const std::vector<size_t> &indices) const;
    void set(const std::vector<size_t> &indices, bool value);

    Tensor<bool>* operator()(size_t index);
    Tensor<bool>* operator()(size_t index, const std::pair<size_t, size_t> &range);

    static Tensor<bool> *cat(const std::vector<Tensor<bool>>& tensors, int dim);
    static Tensor<bool> *tile(const Tensor<bool>& tensor, int dim, int n);

    static Tensor<bool> *transpose(Tensor<bool>& tensor, int dim1, int dim2);
    Tensor<bool> transpose(int dim1, int dim2);

    static Tensor<bool> *permute(Tensor<bool>& tensor, const std::vector<int>& dims);
    Tensor<bool> permute(const std::vector<int>& dims);

    static Tensor<bool> *view(Tensor<bool>& tensor, const std::vector<size_t>& dims);
    Tensor<bool> view(const std::vector<size_t>& dims);

//    vector<size_t> indices(size_t linear_index) const;
};