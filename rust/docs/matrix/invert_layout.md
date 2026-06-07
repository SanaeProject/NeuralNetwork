行列のレイアウトを反転させた新しい行列を返す

# Example
```rust
use NeuralNetwork::matrix::Matrix;
use NeuralNetwork::matrix_layout::{RowMajor, ColumnMajor};
let m: Matrix<i32, RowMajor> = Matrix::new([[1, 2], [3, 4]]);
let inverted = m.invert_layout();

// 反転後の行列はColumnMajorレイアウトになる。
// アクセス時のインデックスは同じだが、内部的なデータの配置が異なる。
assert_eq!(inverted[(0, 0)], 1);
assert_eq!(inverted[(0, 1)], 2);
assert_eq!(inverted[(1, 0)], 3);
assert_eq!(inverted[(1, 1)], 4);
```
