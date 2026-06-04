指定した位置の要素への可変参照を返します。
# Examples
```
use matrix::Matrix;
let mut m: Matrix<i32> = Matrix::with_size(2, 2);
if let Some(val) = m.get_mut(0, 0) {
    *val = 42;
}
assert_eq!(m.get(0, 0), Some(&42));
```
