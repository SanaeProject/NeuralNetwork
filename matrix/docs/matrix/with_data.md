データを指定して行列を作成します。データの長さは行列のサイズと一致する必要があります。

# Panics
データの長さが行列のサイズと一致しない場合にパニックが発生します。

# Arguments
* `data` - 行列の要素を格納するベクタ
* `row` - 行列の行数
* `col` - 行列の列数

# Example
```rust
use matrix::Matrix;
let data = vec![1, 2, 3, 4, 5, 6];
let matrix = Matrix::<i32>::with_data(data, 2, 3);
assert_eq!(matrix.get(0, 0), Some(&1));
```
