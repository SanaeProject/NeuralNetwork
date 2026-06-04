サイズを指定して行列を作成します。
* 要素は全てTのデフォルト値で初期化されます。
* TはDefaultとCopyトレイトを実装している必要があります。
# Examples
```
use matrix::Matrix;
let m: Matrix<i32> = Matrix::with_size(2, 2);
assert_eq!(m.get(0, 0), Some(&0));
assert_eq!(m.get(0, 1), Some(&0));
assert_eq!(m.get(1, 0), Some(&0));
assert_eq!(m.get(1, 1), Some(&0));
```
