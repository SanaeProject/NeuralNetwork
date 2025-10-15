cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenBLAS
find_package(BLAS REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Library source files
set(LIBRARY_SOURCES
    src/Matrix.cpp
    src/ActivationFunctions.cpp
    src/Layer.cpp
    src/NeuralNetwork.cpp
)

# Create static library
add_library(NeuralNetwork STATIC ${LIBRARY_SOURCES})
target_link_libraries(NeuralNetwork ${BLAS_LIBRARIES})

# Examples
add_executable(example_xor examples/example_xor.cpp)
target_link_libraries(example_xor NeuralNetwork)

# Installation
install(TARGETS NeuralNetwork
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/NeuralNetwork
    DESTINATION include
)
