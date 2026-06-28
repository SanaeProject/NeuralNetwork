fn main() {
    use NeuralNetwork::matrix::Matrix;
    let m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    assert_eq!(m[(0, 0)], 1);
    assert_eq!(m[(0, 1)], 2);
    assert_eq!(m[(1, 0)], 3);
    assert_eq!(m[(1, 1)], 4);
}