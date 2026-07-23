# ユーティリティ

## rows() / cols()

行列の行数、および列数を取得します。

$$ \begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ \end{bmatrix}
=>
rows = 3, cols = 4
$$

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(3, 4); // [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
assert_eq!(m.rows(), 3);
assert_eq!(m.cols(), 4);
```

## get(row, col) / get_mut(row, col)

指定したインデックスの要素への参照（または可変参照）を `Option` 型で安全に取得します。範囲外アクセスの場合は `None` を返します。

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(2, 2);
assert_eq!(m.get(0, 0), Some(&0));
```

```rust
use matrix::Matrix;
let mut m: Matrix<i32> = Matrix::with_size(2, 2);
if let Some(val) = m.get_mut(0, 0) {
    *val = 42;
}
assert_eq!(m.get(0, 0), Some(&42));
```

## インデックスアクセス (`matrix[(r, c)]`)
  
`std::ops::Index` トレイトの実装により、`matrix[(row, col)]` 形式の直感的な記述で要素へのアクセスが可能です（範囲外の場合はパニックします）。

## `std::fmt::Display` トレイト

`println!("{}", matrix);` で呼び出した際、各行をタブ区切り（`\t`）の人間が読みやすい2次元グリッド形式で標準出力へフォーマットします。
