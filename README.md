# SanaeProject/NeuralNetwork

NeuralNetworkをゼロから構築するプロジェクトです。

- [ゼロから作るDeep Learning](https://www.oreilly.co.jp/books/9784873117584/)を参考にしています。  
- 作成にはRustを使用し、行列型の実装から始めています。

## 行列型

- 行列型の実装は、`matrix`クレートにまとめています。
- `matrix`クレートは、行列の基本的な演算（加算、減算、乗算、転置など）をサポートしています。
- あくまで、NeuralNetworkの構築に必要な機能のみを実装しています。(逆行列や固有値分解などは未実装)
- 行列の演算には、rayonを使用して並列化を行っています。若しくはCLBLASTを使用して高速化することも検討しています。

### 主なAPI・メソッド仕様

`Matrix<T, L>` 構造体で提供されている主なメソッドの一覧です。

#### 1. コンストラクタ (Constructors)

<details>
<summary>Matrix::new(data)</summary>

固定長2次元配列 `[[T; COLS]; ROWS]` から行列を生成しま  す。指定されたレイアウト（`MatrixLayout`）に基づき、内部の1次元ベクタへ自動的に配置されます。

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
assert_eq!(m.get(0, 0), Some(&1));
assert_eq!(m.get(0, 1), Some(&2));
assert_eq!(m.get(1, 0), Some(&3));
assert_eq!(m.get(1, 1), Some(&4));
```
</details>

<details>
<summary>Matrix::with_size(row, col)</summary>

指定した行数・列数で、すべての要素が初期値（`T::default()`）の行列を生成します。

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(2, 2);
assert_eq!(m.get(0, 0), Some(&0));
assert_eq!(m.get(0, 1), Some(&0));
assert_eq!(m.get(1, 0), Some(&0));
assert_eq!(m.get(1, 1), Some(&0));
```
</details>

<details>
<summary>Matrix::with_data(data, row, col)</summary>

既存の1次元ベクタ `Vec<T>` をもとに、指定したサイズでデータ長が一致しているかアサートした上で初期化します。

```rust
use matrix::Matrix;
let data = vec![1, 2, 3, 4, 5, 6];
let matrix = Matrix::<i32>::with_data(data, 2, 3);
assert_eq!(matrix.get(0, 0), Some(&1));
```
</details>


#### 2. ユーティリティ & 要素アクセス (Utility & Accessors)

<details>
<summary>rows() / cols()</summary>

行列の行数、および列数を取得します。

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(3, 4); // [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
assert_eq!(m.rows(), 3);
```

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(3, 4); // [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
assert_eq!(m.cols(), 4);
```

</details>

<details>
<summary>get(row, col) / get_mut(row, col)</summary>

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
</details>

<details>
<summary>インデックスアクセス (`matrix[(r, c)]`)</summary>

`std::ops::Index` トレイトの実装により、`matrix[(row, col)]` 形式の直感的な記述で要素へのアクセスが可能です（範囲外の場合はパニックします）。
</details>

#### 3. イテレータ (Iterators)

単一スレッド用の通常のイテレータに加え、`rayon` を利用した並列処理用イテレータを標準でサポートしています。
<details>
<summary>全体イテレータ `iter()` / `iter_mut()` (通常) 、 `par_iter()` / `par_iter_mut()` (並列)</summary>

```rust
use matrix::Matrix;
let mtx = Matrix::<i32>::new([[1, 2], [3, 4]]);
let mut iter = mtx.iter();
assert_eq!(iter.next(), Some(&1));
assert_eq!(iter.next(), Some(&2));
assert_eq!(iter.next(), Some(&3));
assert_eq!(iter.next(), Some(&4));
```

</details>

<details>
<summary>行方向イテレータ `row_iter(row)` / `row_iter_mut(row)` (通常) 、 `row_par_iter(row)` / `row_par_iter_mut(row)` (並列)</summary>

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

</details>

<details>
<summary>列方向イテレータ `col_iter(col)` / `col_iter_mut(col)` (通常) 、 `col_par_iter(col)` / `col_par_iter_mut(col)` (並列)</summary>

*(※メモリレイアウトのストライド計算ロジックを挟むことで、効率的なスキップ・ステップ走査を行います)*

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(2, 2);
assert_eq!(m.col_iter(0).unwrap().collect::<Vec<&i32>>(), vec![&0, &0]);
```
```rust
use matrix::Matrix;
let mut m: Matrix<i32> = Matrix::with_size(2, 2);
if let Some(col_iter) = m.col_iter_mut(0) {
   col_iter.for_each(|val| *val = 42);
}
assert_eq!(m.col_iter(0).unwrap().collect::<Vec<&i32>>(), vec![&42, &42]);
```

</details>

#### 4. 行列演算 (Matrix Operations)

通常のメソッド呼び出しのほか、Rustの標準的な演算子（`+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`）に対応しています。また、アルゴリズム戦略（`MatrixAlgorithm`）が用意されています。

<details>
<summary>加算・減算 (`+`, `-`, `+=`, `-=`)</summary>

同じサイズの行列同士の要素ごとの加減算を行います。サイズ不一致の場合はエラーを返します。

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

</details>

<details>
<summary>行列積 (`*`, `*=`)</summary>
通常の行列掛け算（線形代数的な積）を行います（左側の列数と右側の行数が一致している必要があります）。

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
</details>

<details>
<summary>要素ごとの積・商 (`hadamard_mul`, `/`, `/=`)</summary>
アダマール積（要素ごとの掛け算）および要素ごとの割り算を行います。ゼロ除算は厳しくチェックされ、エラーを返します。

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
</details>

<details>
<summary>スカラー倍 (`mul_scalar`)</summary>
行列の全要素に共通の値を掛け合わせます。

```rust
use matrix::Matrix;
let mut mat = Matrix::<i32>::new([[1, 2], [3, 4]]);
mat.mul_scalar(2);
assert_eq!(mat[(0, 0)], 2);
assert_eq!(mat[(0, 1)], 4);
assert_eq!(mat[(1, 0)], 6);
assert_eq!(mat[(1, 1)], 8);
```
</details>

#### 5. 変換処理 (Transformations)

<details>
<summary>`transpose()` / `transpose_par()`</summary>
行列の転置を行います（行と列の入れ替え）。`_par` 版は並列イテレータを用いて高速に転置処理を実行します。

```rust
use matrix::Matrix;
let m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
let transposed = m.transpose();
assert_eq!(transposed[(0, 0)], 1);
assert_eq!(transposed[(0, 1)], 3);
assert_eq!(transposed[(1, 0)], 2);
assert_eq!(transposed[(1, 1)], 4);
```
</details>

<details>
<summary>`invert_layout()` / `invert_layout_par()`</summary>

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
</details>

#### 6. 表示 (Display)

<details>
<summary>`std::fmt::Display` トレイト</summary>

`println!("{}", matrix);` で呼び出した際、各行をタブ区切り（`\t`）の人間が読みやすい2次元グリッド形式で標準出力へフォーマットします。

</details>
