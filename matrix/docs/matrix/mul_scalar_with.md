スカラー倍を行います。
このメソッドは行列の全要素に対してスカラー値を掛ける操作を行います。

# Example
```rust
use matrix::Matrix;
use matrix::algorithm::*;

let mut mat = Matrix::<i32>::new([[1, 2], [3, 4]]);
mat.mul_scalar_with::<ParallelAlgorithm>(2);
assert_eq!(mat[(0, 0)], 2);
assert_eq!(mat[(0, 1)], 4);
assert_eq!(mat[(1, 0)], 6);
assert_eq!(mat[(1, 1)], 8);
```
