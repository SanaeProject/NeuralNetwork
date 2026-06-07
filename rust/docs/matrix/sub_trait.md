同じサイズの行列同士の要素ごとの減算を行います。
* サイズが異なる行列同士の減算はパニックを引き起こします。

# Examples

```rust
use NeuralNetwork::matrix::Matrix;
let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
let m3 = m1 - m2;
assert_eq!(m3[(0, 0)], -4);
assert_eq!(m3[(0, 1)], -4);
assert_eq!(m3[(1, 0)], -4);
assert_eq!(m3[(1, 1)], -4);
```
