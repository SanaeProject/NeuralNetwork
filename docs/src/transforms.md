# 変換処理 (Transformations)

## `transpose()` / `transpose_par()`

行列の転置を行います（行と列の入れ替え）。`_par` 版は並列イテレータを用いて高速に転置処理を実行します。

$$
\begin{aligned}
A_{r,c} => A^T  = A_{c,r} \\
A^T_{j, k} = A_{k, j}\\ 
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\Rightarrow
\begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}
\end{aligned}
$$

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let transposed = m.transpose();
assert_eq!(transposed[(0, 0)], 1);
assert_eq!(transposed[(0, 1)], 3);
assert_eq!(transposed[(1, 0)], 2);
assert_eq!(transposed[(1, 1)], 4);
```

## `invert_layout()` / `invert_layout_par()`

内部のデータ配列を転置させつつ、型レベルでのメモリレイアウト（例: 行優先 ⇄ 列優先）を反転させた新しい行列を生成します。

```rust
use matrix::Matrix;
use matrix::layout::{RowMajor, ColumnMajor};
let m: Matrix<i32, RowMajor> = Matrix::new([[1, 2], [3, 4]]);
let inverted = m.invert_layout();

// 反転後の行列はColumnMajorレイアウトになる。
// アクセス時のインデックスは同じだが、内部的なデータの配置が異なる。
assert_eq!(inverted[(0, 0)], 1);
assert_eq!(inverted[(0, 1)], 2);
assert_eq!(inverted[(1, 0)], 3);
assert_eq!(inverted[(1, 1)], 4);
```
