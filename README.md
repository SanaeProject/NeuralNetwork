# NeuralNetwork
C++でニューラルネットワークを扱うためのライブラリです。OpenBlasを使用した高速な行列演算により、効率的なニューラルネットワークの実装を提供します。

## 特徴

- **OpenBLASによる高速行列演算**: BLAS (Basic Linear Algebra Subprograms) を使用した効率的な行列計算
- **シンプルなAPI**: C++で簡単にニューラルネットワークを構築・訓練できるインターフェース
- **複数の活性化関数**: Sigmoid, ReLU, Tanh, Softmax, Linear, Leaky ReLUをサポート
- **柔軟なネットワーク構造**: レイヤーを追加することで任意のネットワーク構造を構築可能
- **重み初期化戦略**: Xavier初期化、He初期化などをサポート

## 必要な環境

- C++17 対応コンパイラ (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10+
- OpenBLAS ライブラリ

### OpenBLASのインストール

#### Ubuntu/Debian
```bash
sudo apt-get install libopenblas-dev
```

#### macOS
```bash
brew install openblas
```

#### Windows
OpenBLASのバイナリをダウンロードして、適切なパスに配置してください。

## ビルド方法

```bash
# リポジトリをクローン
git clone https://github.com/SanaeProject/NeuralNetwork.git
cd NeuralNetwork

# ビルドディレクトリを作成
mkdir build
cd build

# CMakeでビルドシステムを生成
cmake ..

# ビルド
make

# 実行例を試す
./example_xor
```

## 使用例

### シンプルなインクルード

すべての機能を使用するには、単一のヘッダーファイルをインクルードできます：

```cpp
#include "NeuralNetwork.hpp"
```

個別のコンポーネントを使用する場合は、個別にインクルードすることもできます：

```cpp
#include "NeuralNetwork/Matrix.h"
#include "NeuralNetwork/ActivationFunctions.h"
#include "NeuralNetwork/Layer.h"
#include "NeuralNetwork/NeuralNetwork.h"
```

### XOR問題を解く

```cpp
#include "NeuralNetwork.hpp"

int main() {
    // ニューラルネットワークの作成 (学習率 = 0.5)
    NeuralNetwork::NeuralNetworkModel nn(0.5);
    
    // レイヤーの追加: 2入力 -> 4隠れ層 -> 1出力
    nn.addLayer(2, 4, NeuralNetwork::ActivationType::SIGMOID);
    nn.addLayer(4, 1, NeuralNetwork::ActivationType::SIGMOID);
    
    // 訓練データの準備 (XOR真理値表)
    std::vector<NeuralNetwork::Matrix> inputs;
    std::vector<NeuralNetwork::Matrix> targets;
    
    inputs.push_back(NeuralNetwork::Matrix(2, 1, {0.0, 0.0}));
    targets.push_back(NeuralNetwork::Matrix(1, 1, {0.0}));
    
    inputs.push_back(NeuralNetwork::Matrix(2, 1, {0.0, 1.0}));
    targets.push_back(NeuralNetwork::Matrix(1, 1, {1.0}));
    
    inputs.push_back(NeuralNetwork::Matrix(2, 1, {1.0, 0.0}));
    targets.push_back(NeuralNetwork::Matrix(1, 1, {1.0}));
    
    inputs.push_back(NeuralNetwork::Matrix(2, 1, {1.0, 1.0}));
    targets.push_back(NeuralNetwork::Matrix(1, 1, {0.0}));
    
    // 訓練
    nn.trainBatch(inputs, targets, 10000);
    
    // 予測
    for (const auto& input : inputs) {
        NeuralNetwork::Matrix prediction = nn.predict(input);
        std::cout << "予測: " << prediction(0, 0) << std::endl;
    }
    
    return 0;
}
```

### 行列演算

```cpp
#include "NeuralNetwork.hpp"

int main() {
    // 行列の作成
    NeuralNetwork::Matrix A(2, 2, {1.0, 2.0, 3.0, 4.0});
    NeuralNetwork::Matrix B(2, 2, {5.0, 6.0, 7.0, 8.0});
    
    // 行列演算
    auto C = A + B;           // 行列の加算
    auto D = A * B;           // 行列の乗算
    auto E = A.transpose();   // 転置
    auto F = A * 2.5;         // スカラー倍
    
    // 要素ごとの演算
    auto G = A.elementWiseMultiply(B);
    
    // 統計関数
    double sum = A.sum();
    double mean = A.mean();
    
    // 表示
    A.print();
    
    return 0;
}
```

## API リファレンス

### Matrix クラス

行列演算を提供するクラスです。OpenBLASを使用した効率的な計算を行います。

**主なメソッド:**
- `Matrix(rows, cols)` - 指定サイズの行列を作成
- `operator+, operator-, operator*` - 行列演算
- `transpose()` - 転置行列
- `elementWiseMultiply()` - 要素ごとの乗算
- `randomize()` - ランダム値で初期化
- `print()` - 行列を表示

### NeuralNetworkModel クラス

ニューラルネットワークを管理するクラスです。

**主なメソッド:**
- `addLayer(inputSize, outputSize, activation)` - レイヤーを追加
- `predict(input)` - 予測を実行
- `train(input, target)` - 単一サンプルで訓練
- `trainBatch(inputs, targets, epochs)` - バッチ訓練

### 活性化関数

- `SIGMOID` - シグモイド関数
- `RELU` - ReLU関数
- `TANH` - 双曲線正接関数
- `SOFTMAX` - ソフトマックス関数
- `LINEAR` - 線形関数（恒等関数）
- `LEAKY_RELU` - Leaky ReLU関数

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。
