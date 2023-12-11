cmake_minimum_required(VERSION 3.26)
project(Tensor)

set(CMAKE_CXX_STANDARD 17)

add_executable(Tensor main.cpp
        tensor.cpp
        tensor.h)
