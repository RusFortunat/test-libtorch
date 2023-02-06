cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(linear_regression)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(linear_regression linear_regression.cpp)
target_link_libraries(linear_regression "${TORCH_LIBRARIES}")
set_property(TARGET linear_regression PROPERTY CXX_STANDARD 14)
