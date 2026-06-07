同じサイズの行列同士の要素ごとの減算を行います。
* サイズが異なる行列同士の減算はパニックを引き起こします。

# Examples

```rust
use NeuralNetwork::matrix::Matrix;
let mut m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
m1.sub(&m2).unwrap();
assert_eq!(m1.get(0, 0), Some(&-4));
assert_eq!(m1.get(0, 1), Some(&-4));
assert_eq!(m1.get(1, 0), Some(&-4));
assert_eq!(m1.get(1, 1), Some(&-4));
```
