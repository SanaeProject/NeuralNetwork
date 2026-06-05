すべての行の不変スライスを返します。
# Examples

```rust
let matrix = Matrix::new([[1, 2, 3], [4, 5, 6]]);
for row in matrix.rows_iter() {
    println!("{:?}", row);
}
```