指定した行の要素への参照を返します。

# Examples

```rust
use NeuralNetwork::matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(2, 2);
assert_eq!(
    m.row_iter(0).unwrap().collect::<Vec<&i32>>(), 
    vec![&0, &0]
);
```
