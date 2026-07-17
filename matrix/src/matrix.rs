use core::fmt;
use std::marker::PhantomData;
use rayon::prelude::*;
use crate::{algorithm::{MatrixAlgorithm, NaiveAlgorithm}, element::MatrixElement, layout::{ MatrixLayout, RowMajor }};

pub struct Matrix<T, L: MatrixLayout = RowMajor> {
    pub(crate) data : Vec<T>,
    rows    : usize,
    cols    : usize,
    _marker : std::marker::PhantomData<L>,
}

impl<T, L: MatrixLayout> Matrix<T, L> {
    // ----------------------------------------------------------------------
    // SECTION: Constructors
    // ----------------------------------------------------------------------
    
    #[doc = include_str!("../docs/matrix/new.md")]
    pub fn new<const ROWS: usize, const COLS: usize>(data: [[T; COLS]; ROWS]) -> Self
    where
        T: Default + Copy
    {
        let mut vec: Vec<T> = std::iter::repeat_with(T::default)
        .take(ROWS * COLS)
        .collect();

        for row in 0..ROWS {
            for col in 0..COLS {
                let idx = L::get_index(row, col, ROWS, COLS).unwrap();
                vec[idx] = data[row][col];
            }
        }

        Self { data: vec, rows: ROWS, cols: COLS, _marker: PhantomData }
    }

    #[doc = include_str!("../docs/matrix/with_size.md")]
    pub fn with_size(row: usize, col: usize) -> Self
    where
        T: Default + Copy
    {
        Self { data: vec![T::default(); row * col], rows: row, cols: col, _marker: std::marker::PhantomData }
    }

    #[doc = include_str!("../docs/matrix/with_data.md")]
    pub fn with_data(data: Vec<T>, row: usize, col: usize) -> Self {
        assert!(data.len() == row * col, "Data length must match matrix dimensions");
        Self { data, rows: row, cols: col, _marker: std::marker::PhantomData }
    }

    // !SECTION: Constructors

    // ----------------------------------------------------------------------
    // SECTION: Utility functions
    // ----------------------------------------------------------------------

    #[doc = include_str!("../docs/matrix/clone.md")]
    pub fn clone(&self) -> Self
    where 
        T: Clone
    {
        Self { data: self.data.clone(), rows: self.rows, cols: self.cols, _marker: PhantomData }
    }

    #[doc = include_str!("../docs/matrix/rows.md")]
    pub fn rows(&self) -> usize { self.rows }

    #[doc = include_str!("../docs/matrix/cols.md")]
    pub fn cols(&self) -> usize { self.cols }

    // !SECTION: Utility functions

    // ----------------------------------------------------------------------
    // SECTION: Getters and Iterators
    // ----------------------------------------------------------------------

    #[doc = include_str!("../docs/matrix/get.md")]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> 
    {
        if row >= self.rows || col >= self.cols { return None; }

        let idx = L::get_index(row, col, self.rows, self.cols)?;
        self.data.get(idx)
    }

    #[doc = include_str!("../docs/matrix/get_mut.md")]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> 
    {
        if row >= self.rows || col >= self.cols { return None; }

        let idx = L::get_index(row, col, self.rows, self.cols)?;
        self.data.get_mut(idx)
    }

    #[doc = include_str!("../docs/matrix/iter.md")]
    pub fn iter(&self) -> impl Iterator<Item = &T> 
    {
        self.data.iter()
    }
    
    #[doc = include_str!("../docs/matrix/iter.md")]
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &T>
    where 
        T: Sync 
    {
        self.data.par_iter()
    }

    #[doc = include_str!("../docs/matrix/iter.md")]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> 
    {
        self.data.iter_mut()
    }

    #[doc = include_str!("../docs/matrix/iter.md")]
    pub fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut T>
    where 
        T: MatrixElement
    {
        self.data.par_iter_mut()
    }

    #[doc = include_str!("../docs/matrix/row_iter.md")]
    pub fn row_iter(&self, row: usize) -> Option<impl Iterator<Item = &T>> {
        let (start, step) = L::row_stride(row, self.rows, self.cols)?;
        Some(self.data.iter().skip(start).step_by(step).take(self.cols))
    }

    #[doc = include_str!("../docs/matrix/row_par_iter.md")]
    pub fn row_par_iter(&self, row: usize) -> Option<impl IndexedParallelIterator<Item = &T>>
    where 
        T: Sync 
    {
        let (start, step) = L::row_stride(row, self.rows, self.cols)?;
        Some(self.data.par_iter().skip(start).step_by(step).take(self.cols))
    }

    #[doc = include_str!("../docs/matrix/row_iter_mut.md")]
    pub fn row_iter_mut(&mut self, row: usize) -> Option<impl Iterator<Item = &mut T>> 
    {
        let (start, step) = L::row_stride(row, self.rows, self.cols)?;
        Some(self.data.iter_mut().skip(start).step_by(step).take(self.cols))
    }

    #[doc = include_str!("../docs/matrix/row_par_iter_mut.md")]
    pub fn row_par_iter_mut(&mut self, row: usize) -> Option<impl IndexedParallelIterator<Item = &mut T>>
    where 
        T: Sync + Send
    {
        let (start, step) = L::row_stride(row, self.rows, self.cols)?;
        Some(self.data.par_iter_mut().skip(start).step_by(step).take(self.cols))
    }

    #[doc = include_str!("../docs/matrix/col_iter.md")]
    pub fn col_iter(&self, col: usize) -> Option<impl Iterator<Item = &T>> 
    {
        let (start, step) = L::col_stride(col, self.rows, self.cols)?;
        Some(self.data.iter().skip(start).step_by(step).take(self.rows))
    }

    #[doc = include_str!("../docs/matrix/col_par_iter.md")]
    pub fn col_par_iter(&self, col: usize) -> Option<impl IndexedParallelIterator<Item = &T>>
    where 
        T: Sync 
    {
        let (start, step) = L::col_stride(col, self.rows, self.cols)?;
        Some(self.data.par_iter().skip(start).step_by(step).take(self.rows))
    }

    #[doc = include_str!("../docs/matrix/col_iter_mut.md")]
    pub fn col_iter_mut(&mut self, col: usize) -> Option<impl Iterator<Item = &mut T>> 
    {
        let (start, step) = L::col_stride(col, self.rows, self.cols)?;
        Some(self.data.iter_mut().skip(start).step_by(step).take(self.rows))
    }

    #[doc = include_str!("../docs/matrix/col_par_iter_mut.md")]
    pub fn col_par_iter_mut(&mut self, col: usize) -> Option<impl IndexedParallelIterator<Item = &mut T>>
    where
        T: Sync + Send,
    {
        let (start, step) = L::col_stride(col, self.rows, self.cols)?;
        Some(self.data.par_iter_mut().skip(start).step_by(step).take(self.rows))
    }

    // !SECTION: Getters and Iterators

    // ----------------------------------------------------------------------
    // SECTION: Matrix Operations
    // ----------------------------------------------------------------------

    #[doc = include_str!("../docs/matrix/add.md")]
    pub fn add(&mut self, other: &Matrix<T, L>) -> Result<(), String> 
    where 
        T: std::ops::AddAssign + MatrixElement
    {
        NaiveAlgorithm::add(self, other)
    }

    #[doc = include_str!("../docs/matrix/add_with.md")]
    pub fn add_with<Calc: MatrixAlgorithm, OL: MatrixLayout>(&mut self, other: &Matrix<T, OL>) -> Result<(), String> 
    where 
        T: std::ops::AddAssign + MatrixElement
    {
        Calc::add(self, other)
    }

    #[doc = include_str!("../docs/matrix/sub.md")]
    pub fn sub(&mut self, other: &Matrix<T, L>) -> Result<(), String> 
    where 
        T: std::ops::SubAssign + MatrixElement
    {
        NaiveAlgorithm::sub(self, other)
    }

    #[doc = include_str!("../docs/matrix/sub_with.md")]
    pub fn sub_with<Calc: MatrixAlgorithm, OL: MatrixLayout>(&mut self, other: &Matrix<T, OL>) -> Result<(), String> 
    where 
        T: std::ops::SubAssign + MatrixElement
    {
        Calc::sub(self, other)
    }

    #[doc = include_str!("../docs/matrix/mul.md")]
    pub fn mul(&mut self, other: &Matrix<T, L>) -> Result<(), String> 
    where 
        T: std::ops::Mul<Output = T> + std::ops::AddAssign + MatrixElement + std::iter::Sum<T>
    {
        NaiveAlgorithm::mtx_mul(self, other).map(|result| *self = result)
    }

    #[doc = include_str!("../docs/matrix/mul_with.md")]
    pub fn mul_with<Calc: MatrixAlgorithm, OL: MatrixLayout>(&mut self, other: &Matrix<T, OL>) -> Result<(), String> 
    where 
        T: std::ops::Mul<Output = T> + std::ops::AddAssign + MatrixElement + std::iter::Sum<T>
    {
        Calc::mtx_mul(self, other).map(|result| *self = result)
    }

    #[doc = include_str!("../docs/matrix/mul_scalar.md")]
    pub fn mul_scalar(&mut self, scalar: T) 
    where 
        T: std::ops::MulAssign + MatrixElement
    {
        NaiveAlgorithm::scalar_mul(self, scalar).unwrap();
    }

    #[doc = include_str!("../docs/matrix/mul_scalar_with.md")]
    pub fn mul_scalar_with<Calc: MatrixAlgorithm>(&mut self, scalar: T) 
    where 
        T: std::ops::MulAssign + MatrixElement
    {
        Calc::scalar_mul(self, scalar).unwrap();
    }

    #[doc = include_str!("../docs/matrix/hadamard_mul.md")]
    pub fn hadamard_mul(&mut self, other: &Matrix<T, L>) -> Result<(), String> 
    where 
        T: std::ops::MulAssign + MatrixElement
    {
        NaiveAlgorithm::hadamard_mul(self, other)
    }

    #[doc = include_str!("../docs/matrix/hadamard_mul_with.md")]
    pub fn hadamard_mul_with<Calc: MatrixAlgorithm, OL: MatrixLayout>(&mut self, other: &Matrix<T, OL>) -> Result<(), String> 
    where 
        T: std::ops::MulAssign + MatrixElement
    {
        Calc::hadamard_mul(self, other)
    }

    #[doc = include_str!("../docs/matrix/div.md")]
    pub fn div(&mut self, other: &Matrix<T, L>) -> Result<(), String> 
    where 
        T: std::ops::DivAssign + MatrixElement
    {
        NaiveAlgorithm::div(self, other)
    }

    #[doc = include_str!("../docs/matrix/div_with.md")]
    pub fn div_with<Calc: MatrixAlgorithm, OL: MatrixLayout>(&mut self, other: &Matrix<T, OL>) -> Result<(), String> 
    where 
        T: std::ops::DivAssign + MatrixElement
    {
        Calc::div(self, other)
    }

    // !SECTION: Matrix Operations

    // ----------------------------------------------------------------------
    // SECTION: Matrix Transformations
    // ----------------------------------------------------------------------

    #[doc = include_str!("../docs/matrix/transpose.md")]
    pub fn transpose(&self) -> Self
    where T: MatrixElement
    {
        let (r_rows, r_cols) =  (self.cols(), self.rows());
        let mut result = Self::with_size(r_rows, r_cols);

        L::major_dir_iter_mut(&mut result.data, r_rows, r_cols)
            .enumerate()
            .for_each(|(r_row, row)| {
            row.iter_mut().zip(L::get_un_major_iter(&self, r_row))
                .for_each(|(a, b)| *a = *b);
        });

        result
    }

    #[doc = include_str!("../docs/matrix/transpose_par.md")]
    pub fn transpose_par(&self) -> Self
    where T: MatrixElement
    {
        let (r_rows, r_cols) =  (self.cols(), self.rows());
        let mut result = Self::with_size(r_rows, r_cols);

        L::major_dir_par_iter_mut(&mut result.data, r_rows, r_cols)
            .enumerate()
            .for_each(|(r_row, row)| {
                row.par_iter_mut().zip(L::get_un_major_par_iter(&self, r_row))
                    .for_each(|(a, b)| *a = *b);
        });

        result
    }

    #[doc = include_str!("../docs/matrix/invert_layout.md")]
    pub fn invert_layout(&self) -> Matrix<T, L::InverseLayout>
    where
        T: MatrixElement,
        L::InverseLayout: MatrixLayout
    {
        Matrix::<T, L::InverseLayout>::with_data(self.transpose().data, self.rows(), self.cols())
    }
    #[doc = include_str!("../docs/matrix/invert_layout_par.md")]
    pub fn invert_layout_par(&self) -> Matrix<T, L::InverseLayout>
    where
        T: MatrixElement,
        L::InverseLayout: MatrixLayout
    {
        Matrix::<T, L::InverseLayout>::with_data(self.transpose_par().data, self.rows(), self.cols())
    }

    // !SECTION: Matrix Transformations
}

// ----------------------------------------------------------------------
// SECTION: Trait Implementations
// ----------------------------------------------------------------------

// NOTE: 行と列のインデックスを使用し、Matrix構造体へアクセスできるようIndexトレイトを実装します。
#[doc = include_str!("../docs/matrix/impl_index.md")]
impl<T, L: MatrixLayout> std::ops::Index<(usize, usize)> for Matrix<T, L> {
    type Output = T;

    #[doc = include_str!("../docs/matrix/index.md")]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output{
        self.get(row, col).expect("Index out of bounds")
    }
}

// ----------------------------------------------------------------------
// SECTION: Addition
// ----------------------------------------------------------------------

impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Add<&Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::AddAssign + MatrixElement
{
    type Output = Result<Matrix<T, L>, String>;

    #[doc = include_str!("../docs/matrix/add_trait.md")]
    fn add(mut self, other: &Matrix<T, OL>) -> Self::Output {
        assert!(self.rows == other.rows && self.cols == other.cols, "Matrix dimensions must match for addition");
        if self.rows == 0 || self.cols == 0 {
            return Err("Matrix dimensions must match for addition".to_string());
        }
        NaiveAlgorithm::add(&mut self, &other)?;
        Ok(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Add<&Matrix<T, OL>> for &Matrix<T, L> 
where  
    T: std::ops::AddAssign + MatrixElement
{
    type Output = Result<Matrix<T, L>, String>;

    #[doc = include_str!("../docs/matrix/add_trait.md")]
    fn add(self, other: &Matrix<T, OL>) -> Self::Output {
        let mut result = self.clone();
        if result.rows == 0 || result.cols == 0 {
            return Err("Matrix dimensions must match for addition".to_string());
        }
        NaiveAlgorithm::add(&mut result, &other)?;
        Ok(result)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::AddAssign<&Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::AddAssign + MatrixElement
{
    #[doc = include_str!("../docs/matrix/add_assign.md")]
    fn add_assign(&mut self, other: &Matrix<T, OL>){
        assert!(self.rows == other.rows && self.cols == other.cols, "Matrix dimensions must match for addition");

        assert_eq!(NaiveAlgorithm::add(self, &other), Ok(()), "Matrix addition failed");
    }
}

// !SECTION: Addition

// ----------------------------------------------------------------------
// SECTION: Subtraction
// ----------------------------------------------------------------------

impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Sub<&Matrix<T, OL>> for Matrix<T, L> 
where 
    T: std::ops::SubAssign + MatrixElement
{
    type Output = Result<Matrix<T, L>, String>;

    #[doc = include_str!("../docs/matrix/sub_trait.md")]
    fn sub(mut self, other: &Matrix<T, OL>) -> Self::Output {
        if self.rows == 0 || self.cols == 0 {
            return Err("Matrix dimensions must match for subtraction".to_string());
        }
        NaiveAlgorithm::sub(&mut self, &other)?;

        Ok(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Sub<&Matrix<T, OL>> for &Matrix<T, L> 
where 
    T: std::ops::SubAssign + MatrixElement
{
    type Output = Result<Matrix<T, L>, String>;

    #[doc = include_str!("../docs/matrix/sub_trait.md")]
    fn sub(self, other: &Matrix<T, OL>) -> Self::Output {
        if self.rows == 0 || self.cols == 0 {
            return Err("Matrix dimensions must match for subtraction".to_string());
        }
        let mut result = self.clone();
        NaiveAlgorithm::sub(&mut result, &other)?;

        Ok(result)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::SubAssign<&Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::SubAssign + MatrixElement
{
    #[doc = include_str!("../docs/matrix/sub_assign.md")]
    fn sub_assign(&mut self, other: &Matrix<T, OL>){
        assert!(self.rows == other.rows && self.cols == other.cols, "Matrix dimensions must match for subtraction");
        assert_eq!(NaiveAlgorithm::sub(self, &other), Ok(()), "Matrix subtraction failed");
    }
}

// !SECTION: Subtraction

// ----------------------------------------------------------------------
// SECTION: Multiplication
// ----------------------------------------------------------------------

impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Mul<&Matrix<T, OL>> for Matrix<T, L> 
where 
    T: std::ops::MulAssign + MatrixElement + PartialEq + std::ops::Mul<Output = T> + std::ops::AddAssign + std::iter::Sum<T>
{
    type Output = Result<Matrix<T, L>, String>;

    #[doc = include_str!("../docs/matrix/mul_trait.md")]
    fn mul(mut self, other: &Matrix<T, OL>) -> Self::Output {
        if self.rows == 0 || self.cols == 0 || other.rows == 0 || other.cols == 0 {
            return Err("Matrix dimensions must match for multiplication".to_string());
        }
        NaiveAlgorithm::mtx_mul(&self, &other).map(|result| {
            self = result;
        })?;

        Ok(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Mul<&Matrix<T, OL>> for &Matrix<T, L> 
where 
    T: std::ops::MulAssign + MatrixElement + PartialEq + std::ops::Mul<Output = T> + std::ops::AddAssign + std::iter::Sum<T>
{
    type Output = Result<Matrix<T, L>, String>;

    #[doc = include_str!("../docs/matrix/mul_trait.md")]
    fn mul(self, other: &Matrix<T, OL>) -> Self::Output {
        if self.rows == 0 || self.cols == 0 || other.rows == 0 || other.cols == 0 {
            return Err("Matrix dimensions must match for multiplication".to_string());
        }
        let mut result = self.clone();
        NaiveAlgorithm::mtx_mul(&self, &other).map(|res| {
            result = res;
        })?;

        Ok(result)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::MulAssign<&Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::MulAssign + MatrixElement + PartialEq + std::ops::Mul<Output = T> + std::ops::AddAssign + std::iter::Sum<T>
{
    #[doc = include_str!("../docs/matrix/mul_assign.md")]
    fn mul_assign(&mut self, other: &Matrix<T, OL>){
        assert!(self.cols == other.rows, "Incompatible matrix dimensions for multiplication");

        assert_eq!(NaiveAlgorithm::mtx_mul(&self, &other).map(|result| {
            *self = result;
        }), Ok(()), "Matrix multiplication failed");
    }
}

// !SECTION: Multiplication

// ----------------------------------------------------------------------
// SECTION: Division
// ----------------------------------------------------------------------

impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Div<&Matrix<T, OL>> for Matrix<T, L> 
where 
    T: std::ops::DivAssign + MatrixElement + PartialEq
{
    type Output = Result<Matrix<T, L>, String>;

    #[doc = include_str!("../docs/matrix/div_trait.md")]
    fn div(mut self, other: &Matrix<T, OL>) -> Self::Output {
        if self.rows == 0 || self.cols == 0 || other.rows == 0 || other.cols == 0 {
            return Err("Matrix dimensions must match for division".to_string());
        }
        if other.data.iter().all(|val| *val == T::default()) {
            return Err("Division by zero is not allowed".to_string());
        }
        NaiveAlgorithm::div(&mut self, &other)?;

        Ok(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Div<&Matrix<T, OL>> for &Matrix<T, L> 
where 
    T: std::ops::DivAssign + MatrixElement + PartialEq
{
    type Output = Result<Matrix<T, L>, String>;

    #[doc = include_str!("../docs/matrix/div_trait.md")]
    fn div(self, other: &Matrix<T, OL>) -> Self::Output {
        if self.rows == 0 || self.cols == 0 || other.rows == 0 || other.cols == 0 {
            return Err("Matrix dimensions must match for division".to_string());
        }
        assert!(self.rows == other.rows && self.cols == other.cols, "Matrix dimensions must match for division");
        if other.data.iter().all(|val| *val == T::default()) {
            return Err("Division by zero is not allowed".to_string());
        }

        let mut result = self.clone();
        NaiveAlgorithm::div(&mut result, &other)?;

        Ok(result)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::DivAssign<&Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::DivAssign + MatrixElement + PartialEq
{
    #[doc = include_str!("../docs/matrix/div_assign.md")]
    fn div_assign(&mut self, other: &Matrix<T, OL>){
        assert!(self.rows == other.rows && self.cols == other.cols, "Matrix dimensions must match for division");
        assert!(other.data.iter().all(|val| *val != T::default()), "Division by zero is not allowed");
        assert_eq!(NaiveAlgorithm::div(self, &other), Ok(()), "Matrix division failed");
    }
}

// !SECTION: Division

// NOTE: 表示用のフォーマットを実装するために、Displayトレイトを実装します。
impl<T: std::fmt::Display, L: MatrixLayout> std::fmt::Display for Matrix<T, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (0..self.rows).for_each(|r| {
            if let Some(row_iter) = self.row_iter(r) {
                row_iter.for_each(|c| {
                    _ = write!(f, "{}\t", c);
                });
            }
            _ = write!(f, "\n");
        });
        
        Ok(())
    }
}

// !SECTION: Trait Implementations