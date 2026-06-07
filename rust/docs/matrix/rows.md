行数を返します。

# Examples

```rust
use NeuralNetwork::matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(3, 4); // [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
assert_eq!(m.rows(), 3);
```
