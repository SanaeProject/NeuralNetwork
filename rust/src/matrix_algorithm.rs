use crate::matrix_layout::MatrixLayout;
use crate::matrix::Matrix;

pub trait MatrixAlgorithm<T, L: MatrixLayout, LO: MatrixLayout> {
    fn add(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> where T: std::ops::AddAssign + Copy;
    fn sub(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> where T: std::ops::SubAssign + Copy;
    fn hadamard_mul(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> where T: std::ops::MulAssign + Copy;
    fn div(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> where T: std::ops::DivAssign + Copy;
    fn mtx_mul(mtx: &Matrix<T, L>, other: &Matrix<T, LO>) -> Result<Matrix<T, L>, String> where T: std::ops::Mul<Output = T> + std::ops::AddAssign + Default + Copy;
}

pub struct NaiveAlgorithm;
impl<T, L: MatrixLayout, LO: MatrixLayout> MatrixAlgorithm<T, L, LO> for NaiveAlgorithm {
    fn add(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String>
    where T: std::ops::AddAssign + Copy
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_mut_iter(i).unwrap().zip(other.row_iter(i).unwrap()).for_each(|(a, b)| *a += *b);
        }
        Ok(())
    }

    fn sub(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> 
    where T: std::ops::SubAssign + Copy
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_mut_iter(i).unwrap().zip(other.row_iter(i).unwrap()).for_each(|(a, b)| *a -= *b);
        }
        Ok(())
    }

    fn hadamard_mul(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> 
    where T: std::ops::MulAssign + Copy
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_mut_iter(i).unwrap().zip(other.row_iter(i).unwrap()).for_each(|(a, b)| *a *= *b);
        }
        Ok(())
    }

    fn div(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> 
    where T: std::ops::DivAssign + Copy
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_mut_iter(i).unwrap().zip(other.row_iter(i).unwrap()).for_each(|(a, b)| *a /= *b);
        }
        Ok(())
    }

    fn mtx_mul(mtx: &Matrix<T, L>, other: &Matrix<T, LO>) -> Result<Matrix<T, L>, String> 
    where T: Default + Copy + std::ops::Mul<Output = T> + std::ops::AddAssign
    {
        if mtx.cols() != other.rows() {
            return Err("行列のサイズが一致しません".to_string());
        }

        let mut result = Matrix::with_size(mtx.rows(), other.cols());
        for i in 0..mtx.rows() {
            for j in 0..other.cols() {
                let mut temp = T::default();

                mtx.row_iter(i).unwrap().zip(other.col_iter(j).unwrap()).for_each(|(a, b)| {
                    temp += *a * *b;
                });

                *result.get_mut(i, j).unwrap() = temp;
            }
        }
        Ok(result)
    }
}