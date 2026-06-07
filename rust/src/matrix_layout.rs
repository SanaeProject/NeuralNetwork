use rayon::prelude::*;
use crate::{matrix::Matrix, matrix_element::MatrixElement};

pub trait MatrixLayout: Sync + Send {
    #[doc = include_str!("../docs/matrix_layout/get_index.md")]
    fn get_index(row: usize, col: usize, matrix_row: usize, matrix_col: usize) -> Option<usize>;

    #[doc = include_str!("../docs/matrix_layout/row_stride.md")]
    fn row_stride(row: usize, matrix_row: usize, matrix_col: usize) -> Option<(usize, usize)>;

    #[doc = include_str!("../docs/matrix_layout/col_stride.md")]
    fn col_stride(col: usize, matrix_row: usize, matrix_col: usize) -> Option<(usize, usize)>;

    #[doc = include_str!("../docs/matrix_layout/major_dir_iter.md")]
    fn major_dir_iter<T>(mtx: &Vec<T>, matrix_row: usize, matrix_col: usize) -> impl Iterator<Item = &[T]>;
    
    #[doc = include_str!("../docs/matrix_layout/major_dir_iter.md")]
    fn major_dir_iter_mut<T>(mtx: &mut Vec<T>, matrix_row: usize, matrix_col: usize) -> impl Iterator<Item = &mut [T]>;
    #[doc = include_str!("../docs/matrix_layout/major_dir_par_iter.md")]
    fn major_dir_par_iter<T>(mtx: &Vec<T>, matrix_row: usize, matrix_col: usize) -> impl IndexedParallelIterator<Item = &[T]> where T: Sync;
    #[doc = include_str!("../docs/matrix_layout/major_dir_par_iter.md")]
    fn major_dir_par_iter_mut<T>(mtx: &mut Vec<T>, matrix_row: usize, matrix_col: usize) -> impl IndexedParallelIterator<Item = &mut [T]> where T: Send;

    #[doc = include_str!("../docs/matrix_layout/get_un_major_iter.md")]
    fn get_un_major_iter<T, L: MatrixLayout>(mtx: &Matrix<T, L>, i: usize) -> impl Iterator<Item = &T>;
    #[doc = include_str!("../docs/matrix_layout/get_un_major_iter.md")]
    fn get_un_major_iter_mut<T, L: MatrixLayout>(mtx: &mut Matrix<T, L>, i: usize) -> impl Iterator<Item = &mut T>;
    #[doc = include_str!("../docs/matrix_layout/get_un_major_iter.md")]
    fn get_un_major_par_iter<T, L: MatrixLayout>(mtx: &Matrix<T, L>, i: usize) -> impl IndexedParallelIterator<Item = &T> where T: Sync;
    #[doc = include_str!("../docs/matrix_layout/get_un_major_iter.md")]
    fn get_un_major_par_iter_mut<T, L: MatrixLayout>(mtx: &mut Matrix<T, L>, i: usize) -> impl IndexedParallelIterator<Item = &mut T> where T: Send + Sync;

    type InverseLayout;
}

pub struct RowMajor;
impl MatrixLayout for RowMajor {
    fn get_index(row: usize, col: usize, _m_row: usize, m_col: usize) -> Option<usize> {
        if _m_row == 0 || m_col == 0 || row >= _m_row || col >= m_col {
            return None;
        }
        
        Some(row * m_col + col)
    }
    fn row_stride(row: usize, _m_row: usize, m_col: usize) -> Option<(usize, usize)> {
        if _m_row == 0 || m_col == 0 || row >= _m_row {
            return None;
        }

        Some((row * m_col, 1))
    }
    fn col_stride(col: usize, _m_row: usize, m_col: usize) -> Option<(usize, usize)> {
        if _m_row == 0 || m_col == 0 || col >= m_col {
            return None;
        }

        Some((col, m_col))
    }

    fn major_dir_iter<T>(mtx: &Vec<T>, _matrix_row: usize, matrix_col: usize) -> impl Iterator<Item = &[T]> {
        mtx.chunks(matrix_col)
    }
    fn major_dir_iter_mut<T>(mtx: &mut Vec<T>, _matrix_row: usize, matrix_col: usize) -> impl Iterator<Item = &mut [T]> {
        mtx.chunks_mut(matrix_col)
    }
    fn major_dir_par_iter<T>(mtx: &Vec<T>, _matrix_row: usize, matrix_col: usize) -> impl IndexedParallelIterator<Item = &[T]> where T: Sync {
        mtx.par_chunks(matrix_col)
    }
    fn major_dir_par_iter_mut<T>(mtx: &mut Vec<T>, _matrix_row: usize, matrix_col: usize) -> impl IndexedParallelIterator<Item = &mut [T]> where T: Send {
        mtx.par_chunks_mut(matrix_col)
    }
    fn get_un_major_iter<T, L: MatrixLayout>(mtx: &Matrix<T, L>, i: usize) -> impl Iterator<Item = &T> {
        mtx.col_iter(i).unwrap()
    }
    fn get_un_major_iter_mut<T, L: MatrixLayout>(mtx: &mut Matrix<T, L>, i: usize) -> impl Iterator<Item = &mut T> {
        mtx.col_iter_mut(i).unwrap()
    }
    fn get_un_major_par_iter<T, L: MatrixLayout>(mtx: &Matrix<T, L>, i: usize) -> impl IndexedParallelIterator<Item = &T> where T: Sync {
        mtx.col_par_iter(i).unwrap()
    }
    fn get_un_major_par_iter_mut<T, L: MatrixLayout>(mtx: &mut Matrix<T, L>, i: usize) -> impl IndexedParallelIterator<Item = &mut T> where T: Send + Sync{
        mtx.col_par_iter_mut(i).unwrap()
    }

    type InverseLayout = ColumnMajor;
}
pub struct ColumnMajor;
impl MatrixLayout for ColumnMajor {
    fn get_index(row: usize, col: usize, m_row: usize, _m_col: usize) -> Option<usize> {
        if m_row == 0 || _m_col == 0 || row >= m_row || col >= _m_col {
            return None;
        }
        Some(col * m_row + row)
    }
    fn row_stride(row: usize, m_row: usize, _m_col: usize) -> Option<(usize, usize)> {
        if m_row == 0 || _m_col == 0 || row >= m_row {
            return None;
        }
        Some((row, m_row))
    }
    fn col_stride(col: usize, m_row: usize, _m_col: usize) -> Option<(usize, usize)> {
        if m_row == 0 || _m_col == 0 || col >= _m_col {
            return None;
        }
        Some((col * m_row, 1))
    }
    fn major_dir_iter<T>(mtx: &Vec<T>, matrix_row: usize, _matrix_col: usize) -> impl Iterator<Item = &[T]> {
        mtx.chunks(matrix_row)
    }
    fn major_dir_iter_mut<T>(mtx: &mut Vec<T>, matrix_row: usize, _matrix_col: usize) -> impl Iterator<Item = &mut [T]> {
        mtx.chunks_mut(matrix_row)
    }
    fn major_dir_par_iter<T>(mtx: &Vec<T>, matrix_row: usize, _matrix_col: usize) -> impl IndexedParallelIterator<Item = &[T]> where T: Sync {
        mtx.par_chunks(matrix_row)
    }
    fn major_dir_par_iter_mut<T>(mtx: &mut Vec<T>, matrix_row: usize, _matrix_col: usize) -> impl IndexedParallelIterator<Item = &mut [T]> where T: Send {
        mtx.par_chunks_mut(matrix_row)
    }
    fn get_un_major_iter<T, L: MatrixLayout>(mtx: &Matrix<T, L>, i: usize) -> impl Iterator<Item = &T> {
        mtx.row_iter(i).unwrap()
    }
    fn get_un_major_iter_mut<T, L: MatrixLayout>(mtx: &mut Matrix<T, L>, i: usize) -> impl Iterator<Item = &mut T> {
        mtx.row_iter_mut(i).unwrap()
    }
    fn get_un_major_par_iter<T, L: MatrixLayout>(mtx: &Matrix<T, L>, i: usize) -> impl IndexedParallelIterator<Item = &T> where T: Sync {
        mtx.row_par_iter(i).unwrap()
    }
    fn get_un_major_par_iter_mut<T, L: MatrixLayout>(mtx: &mut Matrix<T, L>, i: usize) -> impl IndexedParallelIterator<Item = &mut T> where T: Send + Sync {
        mtx.row_par_iter_mut(i).unwrap()
    }

    type InverseLayout = RowMajor;
}