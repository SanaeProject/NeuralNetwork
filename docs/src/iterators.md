# イテレータ (Iterators)

単一スレッド用の通常のイテレータに加え、`rayon` を利用した並列処理用イテレータを標準でサポートしています。

## 全体イテレータ `iter()` / `iter_mut()` (通常) 、 `par_iter()` / `par_iter_mut()` (並列)

$$
\begin{aligned}
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\Rightarrow
^{行優先 = \begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix}}
_{列優先 = \begin{bmatrix} 1 & 3 & 2 & 4 \end{bmatrix}}
\end{aligned}
$$

```rust
use matrix::Matrix;
let mtx = Matrix::<i32>::new([[1, 2], [3, 4]]);
let mut iter = mtx.iter();
assert_eq!(iter.next(), Some(&1));
assert_eq!(iter.next(), Some(&2));
assert_eq!(iter.next(), Some(&3));
assert_eq!(iter.next(), Some(&4));
```

## 行方向イテレータ `row_iter(row)` / `row_iter_mut(row)` (通 常) 、`row_par_iter(row)` / `row_par_iter_mut(row)` (並列)

* レイアウトには影響されません。

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
=
\begin{bmatrix}
\begin{bmatrix}
1 & 2
\end{bmatrix}
\begin{bmatrix}
3 & 4
\end{bmatrix}
\end{bmatrix}
$$

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(2, 2);
assert_eq!(
    m.row_iter(0).unwrap().collect::<Vec<&i32>>(), 
    vec![&0, &0]
);
```

```rust
use matrix::Matrix;

let mut m: Matrix<i32> = Matrix::with_size(2, 2);
if let Some(row_iter) = m.row_iter_mut(0) {
    row_iter.for_each(|val| *val = 42);
}

assert_eq!(m[(0, 0)], 42);
assert_eq!(m[(0, 1)], 42);
```

## 列方向イテレータ `col_iter(col)` / `col_iter_mut(col)` (通 常) 、 `col_par_iter(col)` / `col_par_iter_mut(col)` (並列)

* レイアウトには影響されません。

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
=
\begin{bmatrix}
\begin{bmatrix}
1 & 3
\end{bmatrix}
\begin{bmatrix}
2 & 4
\end{bmatrix}
\end{bmatrix}
$$

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(2, 2);
assert_eq!(m.col_iter(0).unwrap().collect::<Vec<&i32>>(), vec![&0, & 0]);
```

```rust
use matrix::Matrix;
let mut m: Matrix<i32> = Matrix::with_size(2, 2);
if let Some(col_iter) = m.col_iter_mut(0) {
    col_iter.for_each(|val| *val = 42);
}
assert_eq!(m.col_iter(0).unwrap().collect::<Vec<&i32>>(), vec![&42, &42]);
```
