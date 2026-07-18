指定した行の要素への可変参照を返します。

# Examples

```rust
use matrix::Matrix;

let mut m: Matrix<i32> = Matrix::with_size(2, 2);
if let Some(row_iter) = m.row_iter_mut(0) {
    row_iter.for_each(|val| *val = 42);
}

assert_eq!(m[(0, 0)], 42);
assert_eq!(m[(0, 1)], 42);
```
