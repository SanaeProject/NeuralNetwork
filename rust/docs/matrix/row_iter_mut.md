指定した行の要素への可変参照を返します。

# Examples

```rust
use matrix::Matrix;
let mut m: Matrix<i32> = Matrix::with_size(2, 2);
if let Some(row_iter) = m.row_mut_iter(0) {
    row_iter.for_each(|val| *val = 42);
}
assert_eq!(m.row_mut_iter(0).unwrap().collect::<Vec<&i32>>(), vec![&42, &42]);
```
