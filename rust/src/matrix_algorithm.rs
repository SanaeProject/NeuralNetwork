use crate::matrix_layout::MatrixLayout;
use crate::matrix::Matrix;
use crate::matrix_element::MatrixElement;
use rayon::prelude::*;

pub trait MatrixAlgorithm<T, L: MatrixLayout, LO: MatrixLayout> {
    fn add(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> where T: std::ops::AddAssign + MatrixElement;
    fn sub(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> where T: std::ops::SubAssign + MatrixElement;
    fn hadamard_mul(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> where T: std::ops::MulAssign + MatrixElement;
    fn div(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> where T: std::ops::DivAssign + MatrixElement;
    fn mtx_mul(mtx: &Matrix<T, L>, other: &Matrix<T, LO>) -> Result<Matrix<T, L>, String> where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MatrixElement + std::iter::Sum<T>;
}

pub struct NaiveAlgorithm;
impl<T, L: MatrixLayout, LO: MatrixLayout> MatrixAlgorithm<T, L, LO> for NaiveAlgorithm {
    fn add(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String>
    where T: std::ops::AddAssign + MatrixElement
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_iter_mut(i).unwrap().zip(other.row_iter(i).unwrap()).for_each(|(a, b)| *a += *b);
        }
        Ok(())
    }

    fn sub(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> 
    where T: std::ops::SubAssign + MatrixElement
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_iter_mut(i).unwrap().zip(other.row_iter(i).unwrap()).for_each(|(a, b)| *a -= *b);
        }
        Ok(())
    }

    fn hadamard_mul(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> 
    where T: std::ops::MulAssign + MatrixElement
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_iter_mut(i).unwrap().zip(other.row_iter(i).unwrap()).for_each(|(a, b)| *a *= *b);
        }
        Ok(())
    }

    fn div(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> 
    where T: std::ops::DivAssign + MatrixElement
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_iter_mut(i).unwrap().zip(other.row_iter(i).unwrap()).for_each(|(a, b)| *a /= *b);
        }
        Ok(())
    }

    fn mtx_mul(mtx: &Matrix<T, L>, other: &Matrix<T, LO>) -> Result<Matrix<T, L>, String> 
    where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MatrixElement + std::iter::Sum<T>
    {
        if mtx.cols() != other.rows() {
            return Err("行列のサイズが一致しません".to_string());
        }

        let mut result = Matrix::with_size(mtx.rows(), other.cols());
        for i in 0..mtx.rows() {
            for j in 0..other.cols() {
                let temp = mtx.row_iter(i).unwrap()
                    .zip(other.col_iter(j).unwrap())
                    .map(|(&a, &b)| a * b)
                    .sum::<T>();
                *result.get_mut(i, j).unwrap() = temp;
            }
        }
        Ok(result)
    }
}

pub struct ParallelAlgorithm;
impl<T, L: MatrixLayout, LO: MatrixLayout> MatrixAlgorithm<T, L, LO> for ParallelAlgorithm {
    fn add(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String>
    where
        T: std::ops::AddAssign + Copy + Sync + Send
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_par_iter_mut(i).unwrap().zip(other.row_par_iter(i).unwrap()).for_each(|(a, b)| *a += *b);
        }

        Ok(())
    }

    fn sub(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> 
    where T: std::ops::SubAssign + MatrixElement
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_par_iter_mut(i).unwrap().zip(other.row_par_iter(i).unwrap()).for_each(|(a, b)| *a -= *b);
        }
        Ok(())
    }

    fn hadamard_mul(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> 
    where T: std::ops::MulAssign + MatrixElement
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_par_iter_mut(i).unwrap().zip(other.row_par_iter(i).unwrap()).for_each(|(a, b)| *a *= *b);
        }
        Ok(())
    }

    fn div(mtx: &mut Matrix<T, L>, other: &Matrix<T, LO>) -> Result<(), String> 
    where T: std::ops::DivAssign + MatrixElement
    {
        if mtx.rows() != other.rows() || mtx.cols() != other.cols() {
            return Err("行列のサイズが一致しません".to_string());
        }

        for i in 0..mtx.rows() {
            mtx.row_par_iter_mut(i).unwrap().zip(other.row_par_iter(i).unwrap()).for_each(|(a, b)| *a /= *b);
        }
        Ok(())
    }

    fn mtx_mul(mtx: &Matrix<T, L>, other: &Matrix<T, LO>) -> Result<Matrix<T, L>, String> 
    where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MatrixElement + std::iter::Sum<T>
    {
        if mtx.cols() != other.rows() {
            return Err("行列のサイズが一致しません".to_string());
        }

        let mut result = Matrix::with_size(mtx.rows(), other.cols());
        for i in 0..mtx.rows() {
            for j in 0..other.cols() {
                let temp: T = mtx.row_par_iter(i).unwrap()
                    .zip(other.col_par_iter(j).unwrap())
                    .map(|(&a, &b)| a * b)
                    .sum();

                *result.get_mut(i, j).unwrap() = temp;
            }
        }
        Ok(result)
    }
}