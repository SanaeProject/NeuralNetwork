cmake_minimum_required(VERSION 3.15)
project(NeuralNetwork)

# オプションの定義
option(USE_OPENBLAS "Enable OpenBLAS acceleration if headers/libs are available" OFF)
option(USE_CUBLAS "Enable cuBLAS acceleration if CUDA Toolkit is available" OFF)
option(USE_CLBLAST "Enable CLBlast acceleration if OpenCL and CLBlast are available" OFF)

# C++20を使用
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# src以下のcppを全部取得
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# 実行ファイル
add_executable(NeuralNetwork ${PROJECT_SOURCES})

# includeパス
target_include_directories(NeuralNetwork PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

# CUBLASとCLBlastの両方が有効になっている場合CUBLASを優先する
if(USE_CLBLAST AND USE_CUBLAS)
    set(USE_CLBLAST OFF)    
    message(WARNING "Both cuBLAS and CLBlast acceleration requested. This may lead to conflicts. Consider enabling only one.")
endif()

# OpenBLASとCUBLAS/CLBLASTの両方が有効になっている場合CUBLAS/CLBLASTを優先する
if((USE_CLBLAST OR USE_CUBLAS) AND USE_OPENBLAS)
    set(USE_OPENBLAS OFF)    
    message("WARNING: Both OpenBLAS and cuBLAS/CLBlast acceleration requested. This may lead to conflicts. Consider enabling only one.")
endif()

# CuBLASライブラリを使用する
if(USE_CUBLAS)
    find_package(CUDAToolkit REQUIRED)

    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            CUDA::cublas
            CUDA::cudart
    )

    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUBLAS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_BLAS)

    message(STATUS "✅ cuBLAS acceleration enabled.")
    message(STATUS "    CUDA Toolkit: ${CUDAToolkit_VERSION}")

endif()

# CLBlastライブラリを使用する
if(USE_CLBLAST)
    find_package(OpenCL REQUIRED)
    find_package(CLBlast REQUIRED)

    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            clblast
            OpenCL::OpenCL
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CLBLAST)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_BLAS)

    message(STATUS "✅ CLBlast acceleration enabled.")
    message(STATUS "    OpenCL: ${OpenCL_VERSION}")
    message(STATUS "    CLBlast: ${CLBlast_VERSION}")
endif()

# OpenBLASライブラリを使用する
if(USE_OPENBLAS)
    find_package(OpenBLAS REQUIRED)

    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            OpenBLAS::OpenBLAS
    )

    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPENBLAS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_BLAS)
    message(STATUS "✅ OpenBLAS acceleration enabled.")
endif()