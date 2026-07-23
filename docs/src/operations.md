# 行列演算 (Matrix Operations)

通常のメソッド呼び出しのほか、Rustの標準的な演算子（`+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`）に対応しています。また、アルゴリズム戦略（`MatrixAlgorithm`）が用意されています。

## 加算・減算 (`+`, `-`, `+=`, `-=`)

同じサイズの行列同士の要素ごとの加減算を行います。サイズ不一致の場合はエラーを返します。

$$
\begin{aligned}
A_{r,c} + B_{r,c} = C_{r,c} \\
C_{j,k} = A_{j,k} + B_{j,k} \\
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
+
\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
=
\begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}
\end{aligned}
$$

```rust
use matrix::Matrix;
let mut m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
m1.add(&m2).unwrap();
assert_eq!(m1.get(0, 0), Some(&6));
assert_eq!(m1.get(0, 1), Some(&8));
assert_eq!(m1.get(1, 0), Some(&10));
assert_eq!(m1.get(1, 1), Some(&12));
```

- `MatrixAlgorithm` トレイトを実装することで、加算・減算のアルゴリズムを差し替えることができます。

```rust
use matrix::Matrix;
use matrix::algorithm::*;
let mut m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
m1.add_with::<NaiveAlgorithm, _>(&m2).unwrap();
assert_eq!(m1.get(0, 0), Some(&6));
assert_eq!(m1.get(0, 1), Some(&8));
assert_eq!(m1.get(1, 0), Some(&10));
assert_eq!(m1.get(1, 1), Some(&12));
```



## 行列積 (`*`, `*=`)
通常の行列掛け算（線形代数的な積）を行います（左側の列数と右側の行数が一致している必要があります）。

$$
\begin{aligned}
A_{a, b} * B_{b, c} = C_{a, c} \\
C_{j,k} = \sum_{i=1}^b{A_{j, i} \cdot B_{i, k}}
\end{aligned}
$$
$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\cdot
\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
=
\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

```rust
use matrix::Matrix;
let mut m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
m1.mul(&m2).unwrap();
assert_eq!(m1.get(0, 0), Some(&19));
assert_eq!(m1.get(0, 1), Some(&22));
assert_eq!(m1.get(1, 0), Some(&43));
assert_eq!(m1.get(1, 1), Some(&50));
```



## 要素ごとの積・商 (`hadamard_mul`, `/`, `/=`)
アダマール積（要素ごとの掛け算）および要素ごとの割り算を行います。ゼロ除算は厳しくチェックされ、エラーを返します。

$$
\begin{aligned}
A_{r,c} \odot B_{r,c} = C_{r,c} \\
C_{j,k} = A_{j,k} \cdot B_{j,k} \\
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\odot
\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
=
\begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}
\end{aligned}
$$

```rust
use matrix::Matrix;
let mut m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
m1.hadamard_mul(&m2).unwrap();
assert_eq!(m1.get(0, 0), Some(&5));
assert_eq!(m1.get(0, 1), Some(&12));
assert_eq!(m1.get(1, 0), Some(&21));
assert_eq!(m1.get(1, 1), Some(&32));
```



## スカラー倍 (`mul_scalar`)
行列の全要素に共通の値を掛け合わせます。

$$
\begin{aligned}
A_{r,c} \odot B = C_{r,c} \\
C_{j,k} = A_{j,k} \cdot B \\
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\odot
2
=
\begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix}
\end{aligned}
$$

```rust
use matrix::Matrix;
let mut mat = Matrix::<i32>::new([[1, 2], [3, 4]]);
mat.mul_scalar(2);
assert_eq!(mat[(0, 0)], 2);
assert_eq!(mat[(0, 1)], 4);
assert_eq!(mat[(1, 0)], 6);
assert_eq!(mat[(1, 1)], 8);
```
