# NeuralNetwork

## 目次

- [概要](#概要)
- [目標](#目標)
- [特徴](#特徴)
- [前提・要件](#前提要件)
- [OpenBLAS の利用ポリシー](#openblas-の利用ポリシー)
- [ビルド手順](#ビルド手順)
- [ビルドコマンド例](#ビルドコマンド例)
- [Matrix 実装方針](#matrix-実装方針)
- [API](#api)
- [ライセンス](#ライセンス)

## 概要

このプロジェクトはまず行列型（Matrix）を実装し、そこから層（Layer）・ニューラルネットワーク（NeuralNetwork）を構築していきます。行列演算では高速化のためにOpenBLASを利用するオプションを提供しますが、OpenBLASの利用条件については「OpenBLAS の利用ポリシー」を参照してください。

この方針により、ビルド環境に依存せず安全にフォールバック実装へ切り替えられるようにします。

## 目標

- 教育・研究用途向けの分かりやすいC++実装。
- 行列演算はOpenBLASを利用可能（オプション）で、無い場合は簡易実装でフォールバック。
- レイヤーや損失関数、最適化（簡易SGD等）を段階的に実装。

## 特徴

- Matrix 型（行列演算、転置、要素ごとの演算、行列積など）
- OpenBLAS を利用するオプション（条件を満たす場合のみ有効）
- OpenBLAS がない環境向けの可読なフォールバック実装

## 前提・要件

下記はこのREADMEが想定する一般的な開発環境です。プロジェクトに既にビルドシステム（CMake 等）があることを前提としています。

- C++20 以上
- CMake 3.15 以上（CMake を使う例を示します）
- （任意）OpenBLAS（性能を出したい場合）
  CMakeListsでは初期値でwindows環境かつvcpkgフォルダがCドライブ直下にあり、OPENBLASがインストール済みであることを想定しています。

## OpenBLAS の利用ポリシー

OpenBLAS を利用する場合、本リポジトリでは次の条件を満たす必要があります:

1. OpenBLAS のヘッダ（例: `cblas.h`）がプロジェクトの `include` ディレクトリ、または CMake 等で指定したインクルードパスに存在する。
2. ビルド時に `USE_OPENBLAS` マクロが定義されている。
3. OpenBLAS のライブラリ（例: openblas）が検出・リンク可能であること。

上記3つがすべて満たされる場合に限り、Matrix クラス内部で BLAS 呼び出しを行い、行列積や他の重い演算を OpenBLAS に委譲します。いずれかが満たされない場合は、ライブラリは自動的に純粋な C++ 実装（フォールバック）を使用します。

この設計により、開発中や CI 環境で OpenBLAS が無くても安全にビルド・テストが可能です。

## ビルド手順

- 以下は CMake を使った一例です。`USE_OPENBLAS` オプションを用意し、OpenBLAS のヘッダとライブラリが見つかった場合にのみ `USE_OPENBLAS` を定義してリンクします。
- Windows環境を想定しているため、Cドライブ直下のvcpkgでOpenBLASをインストールした場合のパスを例示しています。
- Windows環境でない場合や別ディレクトリでOpenBLASライブラリをインストールしている場合、適宜[プリセット](CMakePresets.json)に変更を加えてください

```cmake
# CMakeLists.txt 抜粋
option(USE_OPENBLAS "Enable OpenBLAS acceleration if headers/libs are available" OFF)
set(OPENBLAS_PATH "C:/vcpkg/installed/x64-windows/include/openblas/" CACHE STRING "Blas library include path")

if(USE_OPENBLAS)
    # OpenBLAS のヘッダ（cblas.h）を探す
    find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h PATHS "${OPENBLAS_PATH}")
    if(NOT OPENBLAS_INCLUDE_DIR)
        message(WARNING "OpenBLAS request but headers not found.")
    endif()

    # OpenBLAS ライブラリを探す
    find_library(OPENBLAS_LIB NAMES openblas PATHS "${OPENBLAS_PATH}/../../lib")
    if(NOT OPENBLAS_LIB)
        message(WARNING "OpenBLAS request but libs not found.")
    endif()

    if(OPENBLAS_INCLUDE_DIR AND OPENBLAS_LIB)
        target_include_directories(${PROJECT_NAME} PRIVATE ${OPENBLAS_INCLUDE_DIR})
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPENBLAS)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${OPENBLAS_LIB})

        message(NOTICE "Use OpenBLAS SUCCESS!!")
    else()
        message(WARNING "OpenBLAS requested (USE_OPENBLAS=ON) but headers/libs not found. Falling back to pure C++ implementation.")
    endif()
endif()
```

上記のように、ヘッダ（include）とライブラリが正しく見つかった場合のみ `USE_OPENBLAS` がコンパイル定義されます。Windows では vcpkg 等で事前に OpenBLAS を取得し、CMake の toolchain を通すのが簡単です。

### ビルドコマンド例

- ビルドする際は、OpenBLAS を有効化・無効化したい場合に応じて CMake プリセットを使い分けます。
- 最適化を有効にする場合は`release-openblas-gcc`などのプリセットを使用してください。
- Ninja ビルドでは、`debug-openblas-gcc` と `debug-no-openblas-gcc` のプリセットを用意しています。
- Visual Studioでのビルドでは、`debug-openblas-vs` と `debug-no-openblas-vs` のプリセットを用意しています。

```shell
# Windows 例（PowerShell）
# プロジェクトルートで実行(OpenBLAS 有効化)
cmake --preset=debug-openblas-gcc
cmake --build --preset=debug-openblas-gcc
# OpenBLAS 無効化
cmake --preset=debug-no-openblas-gcc
cmake --build --preset=debug-no-openblas-gcc
```

## Matrix 実装方針

- デフォルト：フォールバックの純粋なC++実装（可読性・デバッグ重視）。
- オプション：`USE_OPENBLAS` が定義され、かつヘッダが見つかった場合に OpenBLAS に委譲。
- 抽象レイヤー：Matrix クラスの API は同じに保ち、内部で条件分岐して最適な実装を呼びます。

設計（簡潔）:

```cpp
template<typename T, bool RowMajor = true>
class Matrix {
private:
  size_t _rows, _cols;
public:
  std::vector<T> data;

  Matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols), data(rows * cols) {}

  T& operator()(size_t row, size_t col) {
    return data[row * _cols + col];
  }

  const T& operator()(size_t row, size_t col) const {
    return data[row * _cols + col];
  }

  Matrix<T, RowMajor> multiply(const Matrix<T, RowMajor>& other) const {
    // 行列積の実装
  }

  // その他のメソッド...
};
```

## API

- Matrix
  - コンストラクタ（サイズ指定・初期値指定・2次元コンテナ/イニシャライザリストから初期化など）
  - `operator()`, `operator[]` で要素アクセス
  - `get_row()`, `get_col()` で行・列ビューの取得
  - `transpose()` で転置
  - `add()`, `sub()`, `scalar_mul()`, `scalar_div()` でスカラー/行列演算
  - `hadamard_mul()`, `hadamard_div()` でアダマール積・除算
  - `matrix_mul()` で行列積
  - `convertLayout()` で行優先/列優先変換
  - `is_blas_enabled()` でBLAS使用可否の確認

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。
[LICENSE](LICENSE) ファイルを参照してください。
