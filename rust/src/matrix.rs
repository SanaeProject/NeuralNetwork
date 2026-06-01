use core::fmt;
use crate::matrix_layout::{ RowMajor, MatrixLayout};

pub struct Matrix<T, L: MatrixLayout = RowMajor> {
    data: Vec<T>,
    row: usize,
    col: usize,
    _marker: std::marker::PhantomData<L>,
}

impl<T, L: MatrixLayout> Matrix<T, L> {
    pub fn with_size(row: usize, col: usize) -> Self
    where
        T: Default + Clone,
    {
        Self {
            data: vec![T::default(); row * col],
            row,
            col,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.row || col >= self.col { return None; }

        let idx = L::get_index(row, col, self.row, self.col)?;
        self.data.get(idx)
    }

    pub fn get_row(&self, row: usize) -> Option<impl Iterator<Item = &T>> {
        let (start, step) = L::row_stride(row, self.row, self.col)?;
        Some(self.data.iter().skip(start).step_by(step).take(self.col))
    }

    pub fn get_column(&self, col: usize) -> Option<impl Iterator<Item = &T>> {
        let (start, step) = L::col_stride(col, self.row, self.col)?;
        Some(self.data.iter().skip(start).step_by(step).take(self.row))
    }
}

impl<T: std::fmt::Display, L: MatrixLayout> std::fmt::Display for Matrix<T, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (0..self.row).for_each(|r| {
            if let Some(row_iter) = self.get_row(r) {
                row_iter.for_each(|c| {
                    _ = write!(f, "{}\t", c);
                });
            }
            _ = write!(f, "\n");
        });
        
        Ok(())
    }
}