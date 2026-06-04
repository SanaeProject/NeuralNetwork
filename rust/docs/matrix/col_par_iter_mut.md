指定した列の要素への並列可変参照を返します。
このイテレータは、列の要素への可変参照を返します。イテレータは、列の要素を順番に返しますが、同時に複数の要素にアクセスすることができます。

# Examples

```rust
use matrix::Matrix;
let mut m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let sum = m.col_par_iter_mut(0).unwrap().cloned().sum::<i32>();
assert_eq!(sum, 2);
```