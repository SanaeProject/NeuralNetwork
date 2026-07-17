同じサイズの行列同士の要素ごとの除算を行います。
* サイズが異なる行列同士の除算はパニックを引き起こします。
* 0で割る要素がある場合もパニックを引き起こします。

# Examples

```rust
use matrix::Matrix;
use matrix::algorithm::*;
let mut m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
m1.div_with::<NaiveAlgorithm, _>(&m2).unwrap();
assert_eq!(m1[(0, 0)], 0);
assert_eq!(m1[(0, 1)], 0);
assert_eq!(m1[(1, 0)], 0);
assert_eq!(m1[(1, 1)], 0);
```
