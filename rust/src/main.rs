use NeuralNetwork::matrix_array;
use NeuralNetwork::matrix_layout::{ColumnMajor, RowMajor};

fn main() {
    let mtx = matrix_array::MatrixArray::<i32, RowMajor>::with_size(3, 4);
    println!("{}", mtx);
}