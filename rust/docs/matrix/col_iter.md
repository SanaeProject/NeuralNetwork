指定した列の要素への参照を返します。

# Examples

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(2, 2);
assert_eq!(m.col_iter(0).unwrap().collect::<Vec<&i32>>(), vec![&0, &0]);
```
