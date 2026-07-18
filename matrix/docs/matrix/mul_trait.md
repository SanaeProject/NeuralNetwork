同じサイズの行列同士の行列積を行います。
* 列と行の数が一致しない場合はエラーを返します。

# Examples
```rust
// 1. 参照同士の乗算 (&Matrix * &Matrix)
{
    use matrix::Matrix;
    let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
    let m3 = (&m1 * &m2).unwrap();

    assert_eq!(m3[(0, 0)], 19);
    assert_eq!(m3[(0, 1)], 22);
    assert_eq!(m3[(1, 0)], 43);
    assert_eq!(m3[(1, 1)], 50);
}
// 2. 所有権の移動を伴う乗算 (Matrix * &Matrix)
{
    use matrix::Matrix;
    let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
    let m4 = (m1 * &m2).unwrap(); // m1は所有権を失いますが、m2は所有権を保持します。

    assert_eq!(m4[(0, 0)], 19);
    assert_eq!(m4[(0, 1)], 22);
    assert_eq!(m4[(1, 0)], 43);
    assert_eq!(m4[(1, 1)], 50);
}
```
