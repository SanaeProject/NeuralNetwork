# SanaeProject/NeuralNetwork

NeuralNetworkをゼロから構築するプロジェクトです。

- [ゼロから作るDeep Learning](https://www.oreilly.co.jp/books/9784873117584/)を参考にしています。  
- 作成にはRustを使用し、行列型の実装から始めています。

## 行列型

- 行列型の実装は、`matrix`クレートにまとめています。
- `matrix`クレートは、行列の基本的な演算（加算、減算、乗算、転置など）をサポートしています。
- あくまで、NeuralNetworkの構築に必要な機能のみを実装しています。(逆行列や固有値分解などは未実装)
- 行列の演算には、rayonを使用して並列化を行っています。若しくはCLBLASTを使用して高速化することも検討しています。
