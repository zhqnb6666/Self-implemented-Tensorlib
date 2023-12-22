#include "tensor_bool.h"
#include <random>
#include <algorithm>
#include <iomanip>
#include <utility>

// 默认构造函数
Tensor<bool>::Tensor() : data(std::make_shared<std::vector<char>>()), offset(0) {}

// 用给定维度和值创建张量
Tensor<bool>::Tensor(const std::vector<size_t> &dimensions, const std::vector<bool> &values) {
    // 检查维度和值的合法性...
    dims = dimensions;
    data = std::make_shared<std::vector<char>>();
    data->reserve(values.size());
    for (bool val : values) {
        data->push_back(static_cast<char>(val));
    }
    // 初始化 strides...
    strides = std::vector<size_t>(dims.size(), 0);
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}

Tensor<bool>::Tensor(const std::vector<size_t> &dimensions, bool value) {
    dims = dimensions;
    data = std::make_shared<std::vector<char>>(dims.size(), static_cast<char>(value));
    // 初始化 strides...
    strides = std::vector<size_t>(dims.size(), 0);
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}
Tensor<bool>::Tensor(const Tensor<bool> &other)
        : data(std::make_shared<std::vector<char>>(*other.data)),
          dims(other.dims),
          strides(other.strides),
          offset(other.offset) {
}
Tensor<bool>::Tensor(const std::vector<size_t> &dimensions, std::shared_ptr<std::vector<char>> values)
        : data(std::move(values)),
          dims(dimensions),
          offset(0) {
    // 计算 strides
    strides.resize(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}
Tensor<bool>::Tensor(const std::vector<size_t> &dimensions, std::shared_ptr<std::vector<char>> values, size_t offset)
        : data(std::move(values)),
          dims(dimensions),
          offset(offset) {
    // 计算 strides
    strides.resize(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}
Tensor<bool>::Tensor(Tensor<bool> &other)
        : data(other.data),
          dims(other.dims),
          strides(other.strides),
          offset(other.offset) {
}


// 析构函数
Tensor<bool>::~Tensor() = default;
// 赋值运算符
Tensor<bool>& Tensor<bool>::operator=(const Tensor<bool> &other) {
    if (this != &other) {
        data = std::make_shared<std::vector<char>>(*other.data);
        dims = other.dims;
        strides = other.strides;
        offset = other.offset;
    }
    return *this;
}

Tensor<bool>& Tensor<bool>::operator=(Tensor<bool> &other) {
    if (this != &other) {
        data = std::make_shared<std::vector<char>>(*other.data);
        dims = other.dims;
        strides = other.strides;
        offset = other.offset;
    }
    return *this;
}
Tensor<bool>& Tensor<bool>::operator=(bool value) {
    std::fill(data->begin() + offset, data->begin() + offset + this->size(), static_cast<char>(value));
    return *this;
}
Tensor<bool>& Tensor<bool>::operator=(const std::vector<bool> &values) {
    if (dims.size() != 1 || dims[0] != values.size()) {
        throw std::invalid_argument("Dimension mismatch");
    }

    if (data->size() != values.size()) {
        data->resize(values.size());
    }

    std::transform(values.begin(), values.end(), data->begin(), [](bool b) { return static_cast<char>(b); });

    return *this;
}

// 下标运算符
bool Tensor<bool>::operator[](const std::vector<size_t> &indices) const {
    return static_cast<bool>((*data)[linear_index(indices)]);
}

void Tensor<bool>::set(const std::vector<size_t> &indices, bool value) {
    (*data)[linear_index(indices)] = static_cast<char>(value);
}

std::vector<char>& Tensor<bool>::values() const {
    return *data;
}



Tensor<bool>* Tensor<bool>::rand(const std::vector<size_t> &dimensions) {
    size_t total_size = 1;
    for (auto &dim : dimensions) {
        if (dim == 0) throw std::invalid_argument("Zero dimension not allowed");
        total_size *= dim;
    }

    auto values = std::make_shared<std::vector<char>>(total_size);
    for (auto &val : *values) {
        val = static_cast<char>(std::rand() % 2); // 随机生成 0 或 1
    }

    return new Tensor<bool>(dimensions, values);
}
Tensor<bool>* Tensor<bool>::Zeros(const std::vector<size_t> &dimensions) {
    size_t total_size = 1;
    for (auto &dim : dimensions) {
        if (dim == 0) throw std::invalid_argument("Zero dimension not allowed");
        total_size *= dim;
    }

    auto values = std::make_shared<std::vector<char>>(total_size, static_cast<char>(0));
    return new Tensor<bool>(dimensions, values);
}
Tensor<bool>* Tensor<bool>::Ones(const std::vector<size_t> &dimensions) {
    size_t total_size = 1;
    for (auto &dim : dimensions) {
        if (dim == 0) throw std::invalid_argument("Zero dimension not allowed");
        total_size *= dim;
    }

    auto values = std::make_shared<std::vector<char>>(total_size, static_cast<char>(1));
    return new Tensor<bool>(dimensions, values);
}
Tensor<bool>* Tensor<bool>::eye(size_t n) {
    std::vector<bool> data(n * n, false);
    for (size_t i = 0; i < n; i++) {
        data[i * n + i] = true;
    }

    return new Tensor<bool>({n, n}, data);
}
size_t Tensor<bool>::order() const {
    return dims.size();
}
size_t Tensor<bool>::size() const {
    size_t total_size = 1;
    for (auto &dim : dims) {
        total_size *= dim;
    }
    return total_size;
}
std::vector<size_t> Tensor<bool>::dimensions() const {
    return dims;
}

char *Tensor<bool>::data_ptr() {
    return data->data();
}
string Tensor<bool>::type() const {
    return "bool";
}
void Tensor<bool>::print(std::ostream &os, const std::vector<size_t> &indices, size_t dim) const {
    if (dim == 1 && indices[0] != 0) {
        os << std::endl;
        if (indices.size() >= 3) os << std::endl;
    }
    if (dim == 2 && indices[1] != 0) os << std::endl;

    if (dim == 2 && indices[1] != 0) os << "  [";
    else if (dim == 1 && indices[0] != 0) os << " [";
    else os << "[";

    for (size_t i = 0; i < dims[dim]; i++) {
        std::vector<size_t> new_indices = indices;
        new_indices[dim] = i;
        if (dim == dims.size() - 1) {
            os << std::setw(7) << std::right << ((*data)[linear_index(new_indices)] ? "true" : "false");
        } else {
            print(os, new_indices, dim + 1);
        }
        if (i < dims[dim] - 1) {
            os << ", ";
        }
    }
    os << "]";
}
void Tensor<bool>::print() const {
    if (data->empty()) {
        std::cout << "Empty tensor" << std::endl;
        return;
    }
    print(std::cout, std::vector<size_t>(dims.size(), 0), 0);
    std::cout << std::endl << "----------------------------------------" << std::endl;
}
std::ostream &operator<<(std::ostream &os, const Tensor<bool> &tensor) {
    if (tensor.data->empty()) {
        os << "Empty tensor" << std::endl;
        return os;
    }
    tensor.print(os, std::vector<size_t>(tensor.dims.size(), 0), 0);
    os << std::endl;
    return os;
}
size_t Tensor<bool>::linear_index(const std::vector<size_t> &indices) const {
    if (indices.size() != dims.size()) {
        throw std::invalid_argument("Dimension mismatch");
    }

    size_t linear_index = offset;
    for (int i = 0; i < dims.size(); i++) {
        if (indices[i] >= dims[i]) {
            throw std::invalid_argument("Index out of range");
        }
        linear_index += indices[i] * strides[i];
    }
    return linear_index;
}
Tensor<bool>* Tensor<bool>::operator()(size_t index) {
    // 检查索引是否有效
    if (index >= this->dims[0]) {
        throw std::out_of_range("Index out of range");
    }

    // 计算新的维度
    std::vector<size_t> new_dims(this->dims.begin() + 1, this->dims.end());

    // 计算新的偏移量
    size_t new_offset = this->offset + index * this->strides[0];

    // 创建并返回新的Tensor对象
    return new Tensor<bool>(new_dims, this->data, new_offset);
}
Tensor<bool>* Tensor<bool>::operator()(size_t index, const std::pair<size_t, size_t> &range) {
    // 检查索引和范围是否有效
    if (index >= this->dims[0] || range.first > range.second || range.second > this->dims[1]) {
        throw std::out_of_range("Index or range out of range");
    }

    // 计算新的维度
    std::vector<size_t> new_dims(this->dims.begin() + 1, this->dims.end());
    new_dims[0] = range.second - range.first;

    // 计算新的偏移量
    size_t new_offset = this->offset + index * this->strides[0] + range.first * this->strides[1];

    // 创建并返回新的Tensor对象
    return new Tensor<bool>(new_dims, this->data, new_offset);
}
Tensor<bool>* Tensor<bool>::transpose(Tensor<bool>& tensor, int dim1, int dim2) {
    // 检查维度是否有效
    if (dim1 < 0 || dim1 >= tensor.dims.size() || dim2 < 0 || dim2 >= tensor.dims.size()) {
        throw std::invalid_argument("Invalid dimensions");
    }

    // 创建新的Tensor对象
    auto* result = new Tensor<bool>(tensor);

    // 交换步幅
    std::swap(result->strides[dim1], result->strides[dim2]);

    // 交换维度
    std::swap(result->dims[dim1], result->dims[dim2]);

    return result;
}
Tensor<bool> Tensor<bool>::transpose(int dim1, int dim2) {
    // 检查维度是否有效
    if (dim1 < 0 || dim1 >= this->dims.size() || dim2 < 0 || dim2 >= this->dims.size()) {
        throw std::invalid_argument("Invalid dimensions");
    }

    // 创建新的Tensor对象
    Tensor<bool> result(*this);

    // 交换步幅
    std::swap(result.strides[dim1], result.strides[dim2]);

    // 交换维度
    std::swap(result.dims[dim1], result.dims[dim2]);

    return result;
}
Tensor<bool>* Tensor<bool>::permute(Tensor<bool>& tensor, const std::vector<int>& dims) {
    // 检查维度是否有效
    if (dims.size() != tensor.dims.size()) {
        throw std::invalid_argument("Invalid dimensions");
    }

    // 创建新的Tensor对象
    auto* result = new Tensor<bool>(tensor);

    // 调整步幅和维度
    for (size_t i = 0; i < dims.size(); ++i) {
        result->strides[i] = tensor.strides[dims[i]];
        result->dims[i] = tensor.dims[dims[i]];
    }

    return result;
}
Tensor<bool> Tensor<bool>::permute(const std::vector<int>& dims) {
    // 检查维度是否有效
    if (dims.size() != this->dims.size()) {
        throw std::invalid_argument("Invalid dimensions");
    }

    // 创建新的Tensor对象
    Tensor<bool> result(*this);

    // 调整步幅和维度
    for (size_t i = 0; i < dims.size(); ++i) {
        result.strides[i] = this->strides[dims[i]];
        result.dims[i] = this->dims[dims[i]];
    }

    return result;
}
Tensor<bool>* Tensor<bool>::view(Tensor<bool>& tensor, const std::vector<size_t>& new_dims) {
    // 检查维度是否有效
    size_t new_size = 1;
    for (size_t dim : new_dims) {
        if (dim == 0) {
            throw std::invalid_argument("Zero dimension");
        }
        new_size *= dim;
    }
    if (new_size != tensor.size()) {
        throw std::invalid_argument("Dimensions do not match tensor size");
    }

    // 创建新的Tensor对象
    auto* result = new Tensor<bool>(tensor);
    result->dims = new_dims;

    // 计算新的步幅
    std::vector<size_t> new_strides(new_dims.size());
    new_strides[new_dims.size() - 1] = 1;
    for (int i = new_dims.size() - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * new_dims[i + 1];
    }
    result->strides = new_strides;

    return result;
}
Tensor<bool> Tensor<bool>::view(const std::vector<size_t>& new_dims) {
    // 检查维度是否有效
    size_t new_size = 1;
    for (size_t dim : new_dims) {
        if (dim == 0) {
            throw std::invalid_argument("Zero dimension");
        }
        new_size *= dim;
    }
    if (new_size != this->size()) {
        throw std::invalid_argument("Dimensions do not match tensor size");
    }

    // 创建新的Tensor对象
    Tensor<bool> result(*this);
    result.dims = new_dims;

    // 计算新的步幅
    std::vector<size_t> new_strides(new_dims.size());
    new_strides[new_dims.size() - 1] = 1;
    for (int i = new_dims.size() - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * new_dims[i + 1];
    }
    result.strides = new_strides;

    return result;
}