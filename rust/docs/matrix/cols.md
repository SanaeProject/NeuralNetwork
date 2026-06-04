列数を返します。
# Examples
```
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(3, 4); // [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
assert_eq!(m.cols(), 4);
```
