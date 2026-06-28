行列をクローンします。

# Example
```rust
use NeuralNetwork::matrix::Matrix;
let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let m2 = m1.clone();

assert_eq!(m2.get(0, 0), Some(&1));
assert_eq!(m2.get(0, 1), Some(&2));
assert_eq!(m2.get(1, 0), Some(&3));
assert_eq!(m2.get(1, 1), Some(&4));
```