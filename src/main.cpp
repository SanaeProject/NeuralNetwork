#include "matrix/matrix"
#include "performance.cpp"
#include <iostream>

int main() {
	exec();
	return 0;

#ifdef USE_OPENBLAS
	std::cout << "Using OpenBLAS for matrix operations." << std::endl;
#endif

    using MatrixType = Matrix<float>;
    MatrixType::Container2D data1 = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    MatrixType::Container2D data2 = { {9, 8, 7}, {6, 5, 4}, {3, 2, 1} };

    // コンストラクタのテスト
    {
        std::cout << "Testing constructors..." << std::endl;
        MatrixType mat1;
        MatrixType mat2(3, 3);
        MatrixType mat3(3, 3, 5);
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
        std::cout << mat1 << " * " << mat2 << " = " << mat1.matrix_mul<true>(mat2) << std::endl;
        std::cout << "Matrix multiplication performed.\n" << std::endl;
    }

    return 0;
}
