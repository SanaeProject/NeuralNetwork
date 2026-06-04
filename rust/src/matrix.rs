use core::fmt;
use std::{marker::PhantomData, ops::Mul};
use crate::{matrix_algorithm::{MatrixAlgorithm, NaiveAlgorithm}, matrix_layout::{ MatrixLayout, RowMajor }};

pub struct Matrix<T, L: MatrixLayout = RowMajor> {
    data    : Vec<T>,
    rows    : usize,
    cols    : usize,
    _marker : std::marker::PhantomData<L>,
}

impl<T, L: MatrixLayout> Matrix<T, L> {
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

    #[doc = include_str!("../docs/matrix/rows.md")]
    pub fn rows(&self) -> usize { self.rows }

    #[doc = include_str!("../docs/matrix/cols.md")]
    pub fn cols(&self) -> usize { self.cols }

    #[doc = include_str!("../docs/matrix/get.md")]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols { return None; }

        let idx = L::get_index(row, col, self.rows, self.cols)?;
        self.data.get(idx)
    }

    #[doc = include_str!("../docs/matrix/get_mut.md")]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= self.rows || col >= self.cols { return None; }

        let idx = L::get_index(row, col, self.rows, self.cols)?;
        self.data.get_mut(idx)
    }

    #[doc = include_str!("../docs/matrix/row_iter.md")]
    pub fn row_iter(&self, row: usize) -> Option<impl Iterator<Item = &T>> {
        let (start, step) = L::row_stride(row, self.rows, self.cols)?;
        Some(self.data.iter().skip(start).step_by(step).take(self.cols))
    }

    #[doc = include_str!("../docs/matrix/row_mut_iter.md")]
    pub fn row_mut_iter(&mut self, row: usize) -> Option<impl Iterator<Item = &mut T>> {
        let (start, step) = L::row_stride(row, self.rows, self.cols)?;
        Some(self.data.iter_mut().skip(start).step_by(step).take(self.cols))
    }

    #[doc = include_str!("../docs/matrix/col_iter.md")]
    pub fn col_iter(&self, col: usize) -> Option<impl Iterator<Item = &T>> {
        let (start, step) = L::col_stride(col, self.rows, self.cols)?;
        Some(self.data.iter().skip(start).step_by(step).take(self.rows))
    }

    #[doc = include_str!("../docs/matrix/col_mut_iter.md")]
    pub fn col_mut_iter(&mut self, col: usize) -> Option<impl Iterator<Item = &mut T>> {
        let (start, step) = L::col_stride(col, self.rows, self.cols)?;
        Some(self.data.iter_mut().skip(start).step_by(step).take(self.rows))
    }

    #[doc = include_str!("../docs/matrix/add.md")]
    pub fn add(&mut self, other: &Matrix<T, L>) -> Result<(), String> 
    where 
        T: std::ops::AddAssign + Copy 
    {
        NaiveAlgorithm::add(self, other)
    }

    #[doc = include_str!("../docs/matrix/add_with.md")]
    pub fn add_with<Calc: MatrixAlgorithm<T, L, OL>, OL: MatrixLayout>(&mut self, other: &Matrix<T, OL>) -> Result<(), String> 
    where 
        T: std::ops::AddAssign + Copy 
    {
        Calc::add(self, other)
    }

    #[doc = include_str!("../docs/matrix/sub.md")]
    pub fn sub(&mut self, other: &Matrix<T, L>) -> Result<(), String> 
    where 
        T: std::ops::SubAssign + Copy 
    {
        NaiveAlgorithm::sub(self, other)
    }

    #[doc = include_str!("../docs/matrix/sub_with.md")]
    pub fn sub_with<Calc: MatrixAlgorithm<T, L, OL>, OL: MatrixLayout>(&mut self, other: &Matrix<T, OL>) -> Result<(), String> 
    where 
        T: std::ops::SubAssign + Copy 
    {
        Calc::sub(self, other)
    }

    #[doc = include_str!("../docs/matrix/mul.md")]
    pub fn mul(&mut self, other: &Matrix<T, L>) -> Result<(), String> 
    where 
        T: std::ops::Mul<Output = T> + std::ops::AddAssign + Default + Copy 
    {
        NaiveAlgorithm::mtx_mul(self, other).map(|result| *self = result)
    }

    #[doc = include_str!("../docs/matrix/mul_with.md")]
    pub fn mul_with<Calc: MatrixAlgorithm<T, L, OL>, OL: MatrixLayout>(&mut self, other: &Matrix<T, OL>) -> Result<(), String> 
    where 
        T: std::ops::Mul<Output = T> + std::ops::AddAssign + Default + Copy 
    {
        Calc::mtx_mul(self, other).map(|result| *self = result)
    }

    #[doc = include_str!("../docs/matrix/hadamard_mul.md")]
    pub fn hadamard_mul(&mut self, other: &Matrix<T, L>) -> Result<(), String> 
    where 
        T: std::ops::MulAssign + Copy 
    {
        NaiveAlgorithm::hadamard_mul(self, other)
    }

    #[doc = include_str!("../docs/matrix/hadamard_mul_with.md")]
    pub fn hadamard_mul_with<Calc: MatrixAlgorithm<T, L, OL>, OL: MatrixLayout>(&mut self, other: &Matrix<T, OL>) -> Result<(), String> 
    where 
        T: std::ops::MulAssign + Copy 
    {
        Calc::hadamard_mul(self, other)
    }

    #[doc = include_str!("../docs/matrix/div.md")]
    pub fn div(&mut self, other: &Matrix<T, L>) -> Result<(), String> 
    where 
        T: std::ops::DivAssign + Copy 
    {
        NaiveAlgorithm::div(self, other)
    }

    #[doc = include_str!("../docs/matrix/div_with.md")]
    pub fn div_with<Calc: MatrixAlgorithm<T, L, OL>, OL: MatrixLayout>(&mut self, other: &Matrix<T, OL>) -> Result<(), String> 
    where 
        T: std::ops::DivAssign + Copy 
    {
        Calc::div(self, other)
    }
}

#[doc = include_str!("../docs/matrix/impl_index.md")]
impl<T, L: MatrixLayout> std::ops::Index<(usize, usize)> for Matrix<T, L> {
    type Output = T;

    #[doc = include_str!("../docs/matrix/index.md")]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output{
        self.get(row, col).expect("Index out of bounds")
    }
}

impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Add<Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::AddAssign + Copy
{
    type Output = Option<Matrix<T, L>>;

    #[doc = include_str!("../docs/matrix/add_trait.md")]
    fn add(mut self, other: Matrix<T, OL>) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols { return None; }

        NaiveAlgorithm::add(&mut self, &other).ok()?;
        Some(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::AddAssign<Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::AddAssign + Copy
{
    #[doc = include_str!("../docs/matrix/add_assign.md")]
    fn add_assign(&mut self, other: Matrix<T, OL>){
        assert!(self.rows == other.rows && self.cols == other.cols);

        NaiveAlgorithm::add(self, &other).ok();
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Sub<Matrix<T, OL>> for Matrix<T, L> 
where 
    T: std::ops::SubAssign + Copy
{
    type Output = Option<Matrix<T, L>>;

    #[doc = include_str!("../docs/matrix/sub_trait.md")]
    fn sub(mut self, other: Matrix<T, OL>) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols { return None; }

        NaiveAlgorithm::sub(&mut self, &other).ok()?;
        Some(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::SubAssign<Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::SubAssign + Copy
{
    #[doc = include_str!("../docs/matrix/sub_assign.md")]
    fn sub_assign(&mut self, other: Matrix<T, OL>){
        assert!(self.rows == other.rows && self.cols == other.cols);

        NaiveAlgorithm::sub(self, &other).ok();
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Mul<Matrix<T, OL>> for Matrix<T, L> 
where 
    T: std::ops::MulAssign + Copy + PartialEq + Default + std::ops::Mul<Output = T> + std::ops::AddAssign
{
    type Output = Option<Matrix<T, L>>;

    #[doc = include_str!("../docs/matrix/mul_trait.md")]
    fn mul(mut self, other: Matrix<T, OL>) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols { return None; }

        NaiveAlgorithm::mtx_mul(&self, &other).map(|result| {
            self = result;
        }).ok()?;

        Some(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::MulAssign<Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::MulAssign + Copy + PartialEq + Default + std::ops::Mul<Output = T> + std::ops::AddAssign
{
    #[doc = include_str!("../docs/matrix/mul_assign.md")]
    fn mul_assign(&mut self, other: Matrix<T, OL>){
        assert!(self.rows == other.rows && self.cols == other.cols);

        NaiveAlgorithm::mtx_mul(&self, &other).ok().map(|result| {
            *self = result;
        });
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Div<Matrix<T, OL>> for Matrix<T, L> 
where 
    T: std::ops::DivAssign + Copy + PartialEq + Default
{
    type Output = Option<Matrix<T, L>>;

    #[doc = include_str!("../docs/matrix/div_trait.md")]
    fn div(mut self, other: Matrix<T, OL>) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols { return None; }
        if other.data.iter().any(|val| *val == T::default()) { return None; }

        NaiveAlgorithm::div(&mut self, &other).ok()?;

        Some(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::DivAssign<Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::DivAssign + Copy + PartialEq + Default
{
    #[doc = include_str!("../docs/matrix/div_assign.md")]
    fn div_assign(&mut self, other: Matrix<T, OL>){
        assert!(self.rows == other.rows && self.cols == other.cols);
        assert!(other.data.iter().all(|val| *val != T::default()));

        NaiveAlgorithm::div(self, &other).ok();
    }
}

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