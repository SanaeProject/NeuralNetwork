二次元配列から行列を作成します。
* TはDefaultとCopyトレイトを実装している必要があります。
# Examples
use NeuralNetwork::matrix::Matrix;
assert_eq!(m.get(0, 0), Some(&1));
assert_eq!(m.get(0, 1), Some(&2));
assert_eq!(m.get(1, 0), Some(&3));
assert_eq!(m.get(1, 1), Some(&4));
```
