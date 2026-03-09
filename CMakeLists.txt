cmake_minimum_required(VERSION 3.15)
project(NeuralNetwork)

option(USE_OPENBLAS "Enable OpenBLAS acceleration if headers/libs are available" OFF)
option(USE_CUBLAS "Enable cuBLAS acceleration if CUDA Toolkit is available" OFF)
set(OPENBLAS_PATH "C:/vcpkg/installed/x64-windows/include/openblas/" CACHE STRING "Blas library include path")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB DEBUG_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
message(STATUS "Files in src/: ${DEBUG_SRC_FILES}")

# ソースが存在するか検証
if(PROJECT_SOURCES)
    add_executable(NeuralNetwork ${PROJECT_SOURCES} "include/matrix/matrix.h" "include/matrix/ops.hpp" "include/matrix/ctor.hpp" "include/matrix/util.hpp" "include/matrix/matrix" "include/view/view.h" "include/matrix/blasgemm.h" "src/performance.h")
else()
    message(FATAL_ERROR "No source files found in src/*.cpp. Create at least one .cpp file (e.g. src/main.cpp) or adjust CMakeLists.txt.")
endif()

# CuBLASライブラリを使用する
if(USE_CUBLAS)
    find_package(CUDAToolkit REQUIRED)

    # CUDA 言語を有効化（既に有効ならスキップされる）
    enable_language(CUDA)

    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            CUDA::cublas
            CUDA::cudart
    )

    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUBLAS)

    message(STATUS "✅ cuBLAS acceleration enabled.")
    message(STATUS "    CUDA Toolkit: ${CUDAToolkit_VERSION}")
else()
    message(STATUS "cuBLAS acceleration not enabled.")
endif()

# OPENBLASライブラリを使用する
if(USE_OPENBLAS)
    # OpenBLAS のヘッダ（cblas.h）を探す
    find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h PATHS "${OPENBLAS_PATH}")

    if(NOT OPENBLAS_INCLUDE_DIR)
        message(WARNING "OpenBLAS request but headers not found.")
        message(NOTICE "💡Hint: Place it in the following path:${OPENBLAS_PATH}/cblas.h")
        message(NOTICE "💡Hint: Install OpenBLAS with vcpkg.")
    endif()

    # OpenBLAS ライブラリを探す
    find_library(OPENBLAS_LIB NAMES openblas PATHS "${OPENBLAS_PATH}/../../lib")

    if(NOT OPENBLAS_LIB)
        message(WARNING "OpenBLAS request but libs not found.")
        message(STATUS "💡Hint: If you installed OpenBLAS via vcpkg, try adding:")
        message(STATUS "    -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake")
        message(STATUS "    -DVCPKG_TARGET_TRIPLET=x64-windows")
    endif()

    if(OPENBLAS_INCLUDE_DIR AND OPENBLAS_LIB)
        target_include_directories(${PROJECT_NAME} PRIVATE ${OPENBLAS_INCLUDE_DIR})
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPENBLAS)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENBLAS_LIB})
        message(NOTICE "✅ Use OpenBLAS SUCCESS!!")
        message(NOTICE "INCLUDE DIR:${OPENBLAS_INCLUDE_DIR}")
        message(NOTICE "LIB PATH:${OPENBLAS_LIB}")
    else()
        message(WARNING "OpenBLAS requested but headers/libs not found. Falling back to pure C++ implementation.")
    endif()
else()
    message(STATUS "OpenBLAS acceleration not enabled. Using pure C++ implementation.")
endif()