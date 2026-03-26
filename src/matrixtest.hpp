#ifndef MATRIXTEST_HPP
#define MATRIXTEST_HPP

#include "include/matrix/matrix"
#include <iostream>

void run_matrix_tests() {
#ifdef USE_OPENBLAS
    std::cout << "\033[32mUsing OpenBLAS for matrix operations.\033[0m" << std::endl;
#elif defined(USE_CUBLAS)
    std::cout << "\033[32mUsing cuBLAS for matrix operations.\033[0m" << std::endl;
#elif defined(USE_CLBLAST)
    std::cout << "\033[32mUsing clBLAST for matrix operations.\033[0m" << std::endl;
#endif

    using MatrixType = Matrix<float,true,std::array<float,9>>;
    MatrixType::Container2D data1 = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    MatrixType::Container2D data2 = { {9, 8, 7}, {6, 5, 4}, {3, 2, 1} };

    // コンストラクタのテスト
    {
        std::cout << "Testing constructors..." << std::endl;
        MatrixType mat1;
        MatrixType mat2(3, 3);
        MatrixType mat3(3, 3, []() { return 1.0f; });
        MatrixType mat4(data1);
        MatrixType mat5({ {1, 2}, {3, 4} });

        std::cout << "mat2:" << mat2 << std::endl
            << "mat3:" << mat3 << std::endl
            << "mat4:" << mat4 << std::endl
            << "mat5:" << mat5 << std::endl;
        std::cout << "Constructors tested.\n" << std::endl;
    }

    // 行数と列数の取得
    {
        std::cout << "Testing rows and cols...\n";
        MatrixType mat(data1);
        std::cout << mat << std::endl
            << "Rows: " << mat.rows()
            << ", Cols: " << mat.cols()
            << "\n" << std::endl;
    }

    // データ取得
    {
        std::cout << "Testing data retrieval...\n";
        MatrixType mat(data1);
        const auto& data = mat.data();

        std::cout << mat << std::endl;
        for (const auto& val : data) {
            std::cout << val << " ";
        }

        std::cout << "\nData retrieved.\n" << std::endl;
    }

    // レイアウト変換
    {
        std::cout << "Testing layout conversion...\n";
        MatrixType mat(data1);
        auto transposed = mat.convertLayout();

        std::cout << "Original Matrix:\n" << mat << std::endl;
        std::cout << "Transposed Matrix:\n" << transposed << std::endl;

        std::cout << "Original Layout:";
        for (const auto& val : mat.data()) {
            std::cout << val << " ";
        }

        std::cout << "\nConverted Layout:";
        for (const auto& val : transposed.data()) {
            std::cout << val << " ";
        }

        std::cout << "\nLayout converted.\n" << std::endl;
    }

    // 要素アクセス
    {
        std::cout << "Testing element access..." << std::endl;
        MatrixType mat(data1);
        std::cout << mat << std::endl;
        std::cout << "Element (1,1): " << mat(1, 1) << "\n";
        mat(1, 1) = 10;
        std::cout << "Modified Element (1,1): " << mat(1, 1) << "\n" << std::endl;
    }

    // 比較演算子
    {
        std::cout << "Testing comparison operators...\n";
        MatrixType mat1(data1);
        MatrixType mat2(data1);
        MatrixType mat3(data2);
        std::cout << mat1 << std::endl
            << mat2 << std::endl
            << mat3 << std::endl;
        std::cout << "mat1 == mat2: " << (mat1 == mat2) << "\n";
        std::cout << "mat1 != mat3: " << (mat1 != mat3) << "\n" << std::endl;
    }

    // 加算
    {
        std::cout << "Testing addition...\n";
        MatrixType mat1(data1);
        MatrixType mat2(data2);
        std::cout << mat1 << "+" << mat2 << "=" << mat1.add(mat2) << std::endl;
        std::cout << "Addition performed.\n" << std::endl;
    }

    // 減算
    {
        std::cout << "Testing subtraction...\n";
        MatrixType mat1(data1);
        MatrixType mat2(data2);
        std::cout << mat1 << "-" << mat2 << "=" << mat1.sub(mat2) << std::endl;
        std::cout << "Subtraction performed.\n" << std::endl;
    }

    // アダマール積
    {
        std::cout << "Testing Hadamard multiplication...\n";
        MatrixType mat1(data1);
        MatrixType mat2(data2);
        std::cout << mat1 << " Hadamard* " << mat2 << " = " << mat1.hadamard_mul(mat2) << std::endl;
        std::cout << "Hadamard multiplication performed.\n" << std::endl;
    }

    // スカラー乗算
    {
        std::cout << "Testing scalar multiplication...\n";
        MatrixType mat(data1);
        std::cout << mat << " * 2 = " << mat.scalar_mul(2) << std::endl;
        std::cout << "Scalar multiplication performed.\n" << std::endl;
    }

    // アダマール除算
    {
        std::cout << "Testing Hadamard division...\n";
        MatrixType mat1(data1);
        MatrixType mat2(data2);
        std::cout << mat1 << " Hadamard/ " << mat2 << " = " << mat1.hadamard_div(mat2) << std::endl;
        std::cout << "Hadamard division performed.\n" << std::endl;
    }

    // スカラー除算
    {
        std::cout << "Testing scalar division...\n";
        MatrixType mat(data1);
        std::cout << mat << " / 2" << "=" << mat.scalar_div(2) << std::endl;
        std::cout << "Scalar division performed.\n" << std::endl;
    }

    // 行列積
    {
        std::cout << "Testing matrix multiplication...\n";
        MatrixType mat1(data1);
        MatrixType mat2(data2);

        std::cout << mat1 << " * " << mat2 << " = " << mat1.matrix_mul(mat2) << std::endl;
        std::cout << "Matrix multiplication performed.\n" << std::endl;
    }

    // 転置
    {
        std::cout << "Testing transpose methods...\n";
        MatrixType mat(data1);
        auto transposed_copy = mat.transpose_copy();

        std::cout << "Original Matrix:\n" << mat << std::endl;
        std::cout << "transpose_copy result:\n" << transposed_copy << std::endl;
        std::cout << "transpose in-place result:\n" << mat.transpose() << std::endl;
        std::cout << "Transpose methods tested.\n" << std::endl;
    }

    // 要素への関数適用
    {
        std::cout << "Testing apply and apply_copy...\n";
        MatrixType mat(data1);
        auto applied_copy = mat.apply_copy([](float x) { return x + 0.5f; });

        std::cout << "apply_copy result (+0.5):\n" << applied_copy << std::endl;
        std::cout << "apply in-place result (*2):\n" << mat.apply([](float x) { return x * 2.0f; }) << std::endl;
        std::cout << "Apply methods tested.\n" << std::endl;
    }

    // 行ごとの演算適用
    {
        std::cout << "Testing apply_row and apply_row_copy...\n";
        MatrixType mat(data1);
        std::array<float, 9> row_delta = { 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        auto applied_row_copy = mat.apply_row_copy(row_delta, [](float a, float b) { return a + b; });
        std::cout << "apply_row_copy result (add [1,0,-1]):\n" << applied_row_copy << std::endl;

        std::cout << "apply_row in-place result (sub [1,0,-1]):\n"
            << mat.apply_row(row_delta, [](float a, float b) { return a - b; }) << std::endl;
        std::cout << "apply_row methods tested.\n" << std::endl;
    }

    // 行方向の合計
    {
        std::cout << "Testing sum_rows...\n";
        MatrixType mat(data1);
        auto summed = mat.sum_rows();
        std::cout << "Input Matrix:\n" << mat << std::endl;
        std::cout << "sum_rows result:\n" << summed << std::endl;
        std::cout << "sum_rows tested.\n" << std::endl;
    }
}

#endif // MATRIXTEST_HPP