use NeuralNetwork::matrix_array;

fn main() {
    let mtx = matrix_array::MatrixArray::<i32, false>::with_size(3, 4);
    println!("{}", mtx);
}