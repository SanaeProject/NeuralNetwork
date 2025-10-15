# NeuralNetwork
このプロジェクトはまず行列型（Matrix）を実装し、そこから層（Layer）・ニューラルネットワーク（NeuralNetwork）を構築していきます。行列演算では高速化のためにOpenBLASを利用するオプションを提供しますが、OpenBLASの利用条件については「OpenBLAS の利用ポリシー」を参照してください。


この方針により、ビルド環境に依存せず安全にフォールバック実装へ切り替えられるようにします。

## 目標

- 教育・研究用途向けの分かりやすいC++実装。
- 行列演算はOpenBLASを利用可能（オプション）で、無い場合は簡易実装でフォールバック。
- レイヤーや損失関数、最適化（簡易SGD等）を段階的に実装。

## 特徴

- Matrix 型（行列演算、転置、要素ごとの演算、行列積など）
- Layer 抽象化（Linear, Activation 等）
- NeuralNetwork 組み立て、順伝播・逆伝播の基礎実装
- OpenBLAS を利用するオプション（条件を満たす場合のみ有効）
- OpenBLAS がない環境向けの可読なフォールバック実装

## 前提・要件（想定）

下記はこのREADMEが想定する一般的な開発環境です。プロジェクトに既にビルドシステム（CMake 等）があることを前提としています。

- C++17 以上（C++20 推奨）
- CMake 3.15 以上（CMake を使う例を示します）
- （任意）OpenBLAS（性能を出したい場合）

## OpenBLAS の利用ポリシー（重要）

OpenBLAS を利用する場合、本リポジトリでは次の条件を満たす必要があります:

1. OpenBLAS のヘッダ（例: `cblas.h`）がプロジェクトの `include` ディレクトリ、または CMake 等で指定したインクルードパスに存在する。
2. ビルド時に `USE_OPENBLAS` マクロが定義されている。
3. OpenBLAS のライブラリ（例: openblas）が検出・リンク可能であること。

上記3つがすべて満たされる場合に限り、Matrix クラス内部で BLAS 呼び出しを行い、行列積や他の重い演算を OpenBLAS に委譲します。いずれかが満たされない場合は、ライブラリは自動的に純粋な C++ 実装（フォールバック）を使用します。

この設計により、開発中や CI 環境で OpenBLAS が無くても安全にビルド・テストが可能です。

## ビルド（CMake 例）

以下は CMake を使った一例です。`USE_OPENBLAS` オプションを用意し、OpenBLAS のヘッダとライブラリが見つかった場合にのみ `USE_OPENBLAS` を定義してリンクします。

```cmake
# CMakeLists.txt 抜粋
option(USE_OPENBLAS "Enable OpenBLAS acceleration if headers/libs are available" OFF)

if(USE_OPENBLAS)
	# まず include ディレクトリ内に cblas.h があるかどうかをチェック（プロジェクト内 include を優先）
	find_path(OPENBLAS_INCLUDE_DIR cblas.h PATHS ${PROJECT_SOURCE_DIR}/include)

	if(OPENBLAS_INCLUDE_DIR)
		target_include_directories(${PROJECT_NAME} PRIVATE ${OPENBLAS_INCLUDE_DIR})
		target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPENBLAS)
	else()
		message(WARNING "OpenBLAS requested (USE_OPENBLAS=ON) but headers/libs not found in expected locations. Falling back to pure C++ implementation.")
	endif()
endif()
```

上記のように、ヘッダ（include）とライブラリが正しく見つかった場合のみ `USE_OPENBLAS` がコンパイル定義されます。Windows では vcpkg 等で事前に OpenBLAS を取得し、CMake の toolchain を通すのが簡単です。

## Matrix 実装方針（OpenBLAS とフォールバック）

- デフォルト：フォールバックの純粋なC++実装（可読性・デバッグ重視）。
- オプション：`USE_OPENBLAS` が定義され、かつヘッダが見つかった場合に OpenBLAS に委譲。
- 抽象レイヤー：Matrix クラスの API は同じに保ち、内部で条件分岐して最適な実装を呼びます。

設計（簡潔）:

- class Matrix {
  - rows, cols
  - std::vector<double> data
  - static Matrix multiply(const Matrix &A, const Matrix &B) // 内部でOpenBLAS or フォールバック
  - elementwise ops, transpose, reshape など
- }

## API（想定インターフェース）

- Matrix
  - コンストラクタ、zeros/ones/random ヘルパー
  - operator() で要素アクセス
  - multiply / add / subtract / transpose / apply (要素ごとの関数適用)
- Layer (抽象クラス)
  - forward, backward
- Linear : Layer
  - 重み行列、バイアス、順伝播・逆伝播
- Activation : Layer
  - Sigmoid / ReLU 等
- NeuralNetwork
  - addLayer, predict, train

## 使い方（簡単な例）

以下は擬似コードの例です。

```cpp
#include "matrix.h"
#include "network.h"

int main() {
	// データ準備
	Matrix X = Matrix::random(100, 10);
	Matrix y = Matrix::random(100, 1);

	NeuralNetwork net;
	net.addLayer(std::make_unique<Linear>(10, 32));
	net.addLayer(std::make_unique<ReLU>());
	net.addLayer(std::make_unique<Linear>(32, 1));

	net.train(X, y, /*epochs=*/100, /*lr=*/0.01);
	auto pred = net.predict(X);
}
```

## テスト

簡単なユニットテストやベンチマークを追加しておくと、フォールバックと OpenBLAS 利用時の結果差や精度・速度を確認できます。GoogleTest 等を使うのが便利です。

## 貢献

- Issue で提案してください。
- Pull Request は小さな単位でお願いします（機能追加 / バグ修正 / ドキュメント）。
- コーディング規約：可読性重視、単体テストを付けること。

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は LICENSE ファイルを参照してください。

## 補足・今後の予定

- より高速な行列ライブラリとの統合（MKL など）
- GPU サポート（CUDA / ROCm）
- 追加の最適化アルゴリズム（Adam 等）

---

もしこのREADMEの内容で追加・修正したい点があれば教えてください。ビルド設定が別のシステム（Makefile, Bazel 等）なら、そのビルド手順に合わせて文面を調整します。
