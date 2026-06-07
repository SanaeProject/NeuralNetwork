並列イテレータを返します。
このメソッドは、行の要素を並列に処理したい場合に使用します。

# Examples

```rust
use NeuralNetwork::matrix::Matrix;
let m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let sum: i32 = m.row_par_iter(0).unwrap().cloned().sum();
assert_eq!(sum, 3);
```