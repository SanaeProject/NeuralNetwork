同じサイズの行列同士の要素ごとの減算を行います。
* サイズが異なる行列同士の減算はパニックを引き起こします。

# Examples
```rust
// 1. 参照同士の減算 (&Matrix - &Matrix)
{
    use NeuralNetwork::matrix::Matrix;
    let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
    let m3 = &m1 - &m2;

    assert_eq!(m3[(0, 0)], -4);
    assert_eq!(m3[(0, 1)], -4);
    assert_eq!(m3[(1, 0)], -4);
    assert_eq!(m3[(1, 1)], -4);
}
// 2. 所有権の移動を伴う減算 (Matrix - &Matrix)
{
    use NeuralNetwork::matrix::Matrix;
    let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
    let m4 = m1 - &m2; // m1は所有権を失いますが、m2は所有権を保持します。

    assert_eq!(m4[(0, 0)], -4);
    assert_eq!(m4[(0, 1)], -4);
    assert_eq!(m4[(1, 0)], -4);
    assert_eq!(m4[(1, 1)], -4);
}
```
