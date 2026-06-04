指定した列の要素への可変参照を返します。
# Examples
```
use matrix::Matrix;
let mut m: Matrix<i32> = Matrix::with_size(2, 2);
if let Some(col_iter) = m.col_mut_iter(0) {
   col_iter.for_each(|val| *val = 42);
}
assert_eq!(m.col_iter(0).unwrap().collect::<Vec<&i32>>(), vec![&42, &42]);
```
