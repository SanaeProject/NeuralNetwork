指定した列の要素への並列可変参照を返します。
このイテレータは、列の要素への可変参照を返します。イテレータは、列の要素を順番に返しますが、同時に複数の要素にアクセスすることができます。

# Examples

```rust
use matrix::Matrix;
use rayon::prelude::*;
let mut m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
m.col_par_iter_mut(0).unwrap().for_each(|x| *x *= 2);
assert_eq!(m[(0, 0)], 2);
assert_eq!(m[(1, 0)], 6);
```
