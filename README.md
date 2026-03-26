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
- CMake 3.21 以上（`CMakePresets.json` の要件）
- （任意）OpenBLAS（性能を出したい場合）
  CMakeListsでは初期値でwindows環境かつvcpkgフォルダがCドライブ直下にあり、OPENBLASがインストール済みであることを想定しています。

## OpenBLAS の利用ポリシー

OpenBLAS を利用する場合、本リポジトリでは次の条件を満たす必要があります:

1. CMake オプション `USE_OPENBLAS` が有効化されていること。
2. `find_package(OpenBLAS REQUIRED)` が成功すること。
3. ビルド時に `USE_OPENBLAS` マクロが定義されること。

上記3つがすべて満たされる場合に限り、Matrix クラス内部で BLAS 呼び出しを行い、行列積や他の重い演算を OpenBLAS に委譲します。`USE_OPENBLAS=ON` かつ OpenBLAS が見つからない場合は CMake 構成エラーになります。フォールバック実装を使う場合は `USE_OPENBLAS=OFF`（`no-blas` プリセット）でビルドしてください。

この設計により、開発中や CI 環境で OpenBLAS が無くても安全にビルド・テストが可能です。

## ビルド手順

- BLAS バックエンドは CMake オプション `USE_OPENBLAS` / `USE_CUBLAS` / `USE_CLBLAST` で切り替えます。
- `USE_OPENBLAS=ON` の場合は `find_package(OpenBLAS REQUIRED)`、`USE_CUBLAS=ON` の場合は `find_package(CUDAToolkit REQUIRED)`、`USE_CLBLAST=ON` の場合は `find_package(OpenCL REQUIRED)` と `find_package(CLBlast REQUIRED)` が通る必要があります。
- `USE_CUBLAS` と `USE_CLBLAST` を同時に有効化した場合は `USE_CUBLAS` が優先され、`USE_OPENBLAS` と GPU 系（`USE_CUBLAS` / `USE_CLBLAST`）を同時に有効化した場合は GPU 系が優先されます（詳細は [CMakeLists.txt](CMakeLists.txt) を参照）。
- Windows で OpenBLAS / CLBlast を使う場合、既定プリセットは `C:/vcpkg/scripts/buildsystems/vcpkg.cmake` を参照します。環境が異なる場合は [CMakePresets.json](CMakePresets.json) の `CMAKE_TOOLCHAIN_FILE` を変更してください。

### ビルドコマンド例

- ビルドする際は OpenBLAS、CuBLAS、CLBlast を有効化・無効化したい場合に応じて CMake プリセットを使い分けます。
- 最適化を有効にする場合は`release-openblas-gcc`などのプリセットを使用してください。
- Visual Studio でのビルドでは、{`debug-openblas-vs`, `debug-cublas-vs` , `debug-clblast-vs`} と `debug-no-blas-vs` のプリセットを用意しています。
- clang を使用する場合は、{`debug-openblas-clang`, `debug-cublas-clang`, `debug-clblast-clang`} と `debug-no-blas-clang` のプリセットを用意しています。
- gcc を使用する場合は、{`debug-openblas-gcc`, `debug-cublas-gcc`} と `debug-no-blas-gcc` のプリセットを用意しています。

```shell
# Windows 例（PowerShell）
# OpenBLAS 有効化
cmake --preset=debug-openblas-clang
cmake --build --preset=debug-openblas-clang

# CuBLAS 有効化
cmake --preset=debug-cublas-clang
cmake --build --preset=debug-cublas-clang

# CLBlast 有効化
cmake --preset=debug-clblast-clang
cmake --build --preset=debug-clblast-clang

# BLAS 無効化
cmake --preset=debug-no-blas-clang
cmake --build --preset=debug-no-blas-clang


```

## API

- Matrix
  - コンストラクタ（サイズ指定・初期値コンテナ指定・初期化関数指定・2次元コンテナ/イニシャライザリスト指定）
  - 要素アクセス: `operator()`, `operator[]`
  - ビュー取得: `get_row()`, `get_col()` (参照のみ、const 版対応済み)
  - ポインタ取得: `get_row_ptr()`, `get_col_ptr()`（レイアウト制約あり）
  - レイアウト/転置: `convertLayout()`, `transpose()`, `transpose_copy()`
  - 関数適用: `apply()`, `apply_copy()`, `apply_row()`, `apply_row_copy()`
  - 四則/要素演算: `add()`, `sub()`, `scalar_mul()`, `scalar_div()`, `hadamard_mul()`, `hadamard_div()`
  - 行列積: `matrix_mul()`
  - 集計: `sum_rows()`
  - 補助: `rows()`, `cols()`, `data()`, `is_blas_enabled()`

- Layers
  - Affine
    - 初期化スケール戦略: `StandardDeviation`（抽象基底）, `Xavier`, `He`
    - クラステンプレート: `Affine<ty, use_blas, ExecType, DeviationType, OptimizerType>`
      - `ty`: 要素型
      - `use_blas`: 行列積でBLASを利用するかどうか
      - `ExecType`: 実行ポリシー（既定: `std::execution::sequenced_policy`）
      - `DeviationType`: 重み初期化の標準偏差戦略（既定: `Xavier`）
      - `OptimizerType`: 最適化器（既定: `SGD<ty, ExecType, use_blas>`）
    - コンストラクタ: `Affine(size_t input_size, size_t output_size, ty lr = 0.01f, uint32_t seed = std::random_device{}(), DeviationType dev = DeviationType{})`
      - `_w` (in_dim, out_dim), `_b` (1, out_dim) を正規分布で初期化
      - `optimizer(_w, _b, lr)` を内部で保持
    - `forward(const Matrix<ty>& in) -> Matrix<ty>`
      - `out = in * W + b` を計算（`matrix_mul` + `apply_row`）
    - `backward(const Matrix<ty>& dout) -> Matrix<ty>`
      - `dx = dout * W^T`
      - `dW = X^T * dout`
      - `db = sum_rows(dout)`
      - `optimizer.optimize(dW, db)` を実行

  - ReLU
    - クラステンプレート: `ReLU<ty, ExecPolicy>`
      - `ty`: 要素型
      - `ExecPolicy`: 実行ポリシー（既定: `std::execution::sequenced_policy`）
    - `forward(const Matrix<ty>& in) -> Matrix<ty>`
      - 要素ごとに `max(0, x)` を適用
      - 逆伝播用に出力を内部保持
    - `backward(const Matrix<ty>& dout) -> Matrix<ty>`
      - 保持した出力からマスク `(x > 0 ? 1 : 0)` を生成
      - `dout` と要素積（Hadamard積）を取り、入力勾配 `dx` を返す

  - Sigmoid
    - クラステンプレート: `Sigmoid<ty, ExecPolicy>`
      - `ty`: 要素型
      - `ExecPolicy`: 実行ポリシー（既定: `std::execution::sequenced_policy`）
    - `forward(const Matrix<ty>& in) -> Matrix<ty>`
      - 要素ごとに `1 / (1 + exp(-x))` を適用
      - 逆伝播用に出力を内部保持
    - `backward(const Matrix<ty>& dout) -> Matrix<ty>`
      - 保持した出力から勾配を計算: `dx = dout * (out * (1 - out))`

  - Tanh
    - クラステンプレート: `Tanh<ty, ExecPolicy>`
      - `ty`: 要素型
      - `ExecPolicy`: 実行ポリシー（既定: `std::execution::sequenced_policy`）
    - `forward(const Matrix<ty>& in) -> Matrix<ty>`
      - 要素ごとに `tanh(x)` を適用
      - 逆伝播用に出力を内部保持
    - `backward(const Matrix<ty>& dout) -> Matrix<ty>`
      - 保持した出力から勾配を計算: `dx = dout * (1 - out^2)`

  - IdentityWithLoss
    - クラステンプレート: `IdentityWithLoss<ty, ExecPolicy>`
      - `ty`: 要素型
      - `ExecPolicy`: 実行ポリシー（既定: `std::execution::sequenced_policy`）
    - 静的フラグ: `has_loss = true`
    - `forward(const Matrix<ty>& in) -> Matrix<ty>`
      - 恒等写像として入力をそのまま出力
      - 損失計算/逆伝播用に出力を内部保持
    - `backward(const Matrix<ty>& t) -> Matrix<ty>`
      - `dx = (out - t) / batch_size` を計算
      - `batch_size == 0` の場合は例外を送出
    - `loss(const Matrix<ty>& t) -> double`
      - 二乗誤差: `Σ(out_i - t_i)^2 / (2 * batch_size)`
      - `batch_size == 0` の場合は例外を送出

  - SoftmaxWithLoss
    - クラステンプレート: `SoftmaxWithLoss<ty, ExecPolicy>`
      - `ty`: 要素型
      - `ExecPolicy`: 実行ポリシー（既定: `std::execution::sequenced_policy`）
    - 静的フラグ: `has_loss = true`
    - `forward(const Matrix<ty>& in) -> Matrix<ty>`
      - 行ごとに softmax を適用
      - 数値安定化のため各行の最大値を減算してから `exp` を計算
      - 入力が空、または正規化和が非正の場合は例外を送出
      - 損失計算/逆伝播用に出力を内部保持
    - `backward(const Matrix<ty>& t) -> Matrix<ty>`
      - `dx = (out - t) / batch_size` を計算
      - `batch_size == 0` の場合は例外を送出
    - `loss(const Matrix<ty>& t) -> double`
      - 交差エントロピー: `-Σ(t_i * log(out_i + ε)) / batch_size`
      - `batch_size == 0` の場合は例外を送出

- NeuralNetwork
  - レイヤ構成型: `LayerPack<Layers...>`
    - 制約: レイヤ数は偶数かつ4以上
    - 制約: 最終レイヤは `has_loss == true` を持つ型
  - クラステンプレート: `NeuralNetwork<ty, LayerPackT>`
    - 実体化: `NeuralNetwork<ty, LayerPack<Layers...>>`
  - コンストラクタ: `NeuralNetwork(size_t in_size, size_t hidden_size, size_t out_size, ty learning_rate = 0.01f, uint32_t seed = std::random_device{}())`
    - 先頭/末尾は `Affine` 前提でサイズを設定
    - 中間層は `Affine` の場合 `hidden_size -> hidden_size`、それ以外はデフォルト構築
  - `learn<use_loss = true>(const Matrix<ty>& in, const Matrix<ty>& t) -> double`
    - 全レイヤで順伝播を実行後、逆順で逆伝播を実行
    - `use_loss == true` の場合は最終レイヤの `loss(t)` を返す
    - `use_loss == false` の場合は `0` を返す
  - `predict(const Matrix<ty>& in) -> Matrix<ty>`
    - 学習なしの順伝播のみを実行して推論結果を返す


## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。
[LICENSE](LICENSE) ファイルを参照してください。
