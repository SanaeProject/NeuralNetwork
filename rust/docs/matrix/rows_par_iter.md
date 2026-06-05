各行の不変スライスを順不同で処理する並列イテレータを返します。
# Examples

```rust
let matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);

let row_sums: Vec<i32> = matrix.rows_par_iter()
    .map(|row| row.iter().sum())
    .collect();
assert_eq!(row_sums, vec![6, 15]);
```