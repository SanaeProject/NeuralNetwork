同じサイズの行列同士の要素ごとの乗算を行います。
* サイズが異なる行列同士の乗算はNoneを返します。
# Examples
```
use matrix::Matrix;
let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
let m3 = m1 * m2;
assert_eq!(m3.unwrap().get(0, 0), Some(&19));
assert_eq!(m3.unwrap().get(0, 1), Some(&22));
assert_eq!(m3.unwrap().get(1, 0), Some(&43));
assert_eq!(m3.unwrap().get(1, 1), Some(&50));
```
