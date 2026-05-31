use NeuralNetwork::matrix;
use NeuralNetwork::matrix_layout::{ColumnMajor, RowMajor};

fn main() {
    let mtx = matrix::Matrix::<i32, RowMajor>::with_size(3, 4);
    println!("{}", mtx);
}