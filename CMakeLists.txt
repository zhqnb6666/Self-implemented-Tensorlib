cmake_minimum_required(VERSION 3.16)
project(Tensor)

set(CMAKE_CXX_STANDARD 17)

# add_executable(Tensor main.cpp
#         tensor.cpp
#         tensor.h
#         tensor_bool.cpp
#         tensor_bool.h
#         tensor_bool.cpp
#         tensor_bool.h
#         save_load.cpp
#         tensor_einsum.cpp
#         tensor_einsum.h
#         tensor_math.cpp
#         tensor_math.h
#         gradient.h
#         )

add_executable(
math_test
math_test.cpp
tensor.cpp
tensor.h
tensor_bool.cpp
tensor_bool.h
tensor_bool.cpp
tensor_bool.h
tensor_math.cpp
tensor_math.h
)