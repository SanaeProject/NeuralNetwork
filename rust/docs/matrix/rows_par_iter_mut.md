各行の可変スライスを処理する並列イテレータを返します。

# Examples
```rust
let mut matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);

matrix.rows_par_iter_mut().for_each(|row| {
    row.iter_mut().for_each(|elem| *elem += 1);
});

assert_eq!(matrix, Matrix::new([[2, 3, 4], [5, 6, 7]]));
```