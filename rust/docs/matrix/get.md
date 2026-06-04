指定した位置の要素への参照を返します。
# Examples
```
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(2, 2);
assert_eq!(m.get(0, 0), Some(&0));
```
