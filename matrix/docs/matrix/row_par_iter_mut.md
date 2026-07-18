並列イテレータを返します。
このメソッドは、行の要素を並列に処理したい場合に使用します。

# Examples

```rust
use matrix::Matrix;
use rayon::prelude::*;

let mut m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
m.row_par_iter_mut(0).unwrap().for_each(|x| *x *= 2);
assert_eq!(m[(0, 0)], 2);
```
