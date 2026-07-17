転置を計算するメソッドです。

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let transposed = m.transpose();
assert_eq!(transposed[(0, 0)], 1);
assert_eq!(transposed[(0, 1)], 3);
assert_eq!(transposed[(1, 0)], 2);
assert_eq!(transposed[(1, 1)], 4);
```
