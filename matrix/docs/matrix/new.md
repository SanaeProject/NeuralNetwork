二次元配列から行列を作成します。
* TはDefaultとCopyトレイトを実装している必要があります。

# Examples

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
assert_eq!(m.get(0, 0), Some(&1));
assert_eq!(m.get(0, 1), Some(&2));
assert_eq!(m.get(1, 0), Some(&3));
assert_eq!(m.get(1, 1), Some(&4));
```
