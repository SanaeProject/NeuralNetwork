# コンストラクタ (Constructors)

## Matrix::new(data)

固定長2次元配列 `[[T; COLS]; ROWS]` から行列を生成します。指定された  レイアウト（`MatrixLayout`）に基づき、内部の1次元ベクタへ自動的に配置されます。

- 行優先時

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\Rightarrow
\begin{bmatrix}
1 & 2 & 3 & 4
\end{bmatrix}
$$

- 列優先時

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\Rightarrow
\begin{bmatrix}
1 & 3 & 2 & 4
\end{bmatrix}
$$

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
assert_eq!(m.get(0, 0), Some(&1));
assert_eq!(m.get(0, 1), Some(&2));
assert_eq!(m.get(1, 0), Some(&3));
assert_eq!(m.get(1, 1), Some(&4));
```

## Matrix::with_size(row, col)
  
指定した行数・列数で、すべての要素が初期値（`T::default()`）の行列を生成します。

$$
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
$$

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(2, 2);
assert_eq!(m.get(0, 0), Some(&0));
assert_eq!(m.get(0, 1), Some(&0));
assert_eq!(m.get(1, 0), Some(&0));
assert_eq!(m.get(1, 1), Some(&0));
```

## Matrix::with_data(data, row, col)
  
既存の1次元ベクタ `Vec<T>` をもとに、指定したサイズでデータ長が一致しているかアサートした上で初期化します。

- 二行三列にパース(行優先)

$$
\begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6 \end{bmatrix}
\Rightarrow
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

- 二行三列にパース(列優先)

$$
\begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6 \end{bmatrix}
\Rightarrow
\begin{bmatrix}
1 & 3 & 5 \\
2 & 4 & 6
\end{bmatrix}
$$

```rust
use matrix::Matrix;
let data = vec![1, 2, 3, 4, 5, 6];
let matrix = Matrix::<i32>::with_data(data, 2, 3);
assert_eq!(matrix.get(0, 0), Some(&1));
```
