use core::fmt;
use std::marker::PhantomData;
use crate::matrix_layout::{ RowMajor, MatrixLayout };

pub struct Matrix<T, L: MatrixLayout = RowMajor> {
    data    : Vec<T>,
    rows    : usize,
    cols    : usize,
    _marker : std::marker::PhantomData<L>,
}

impl<T, L: MatrixLayout> Matrix<T, L> {
    /// 二次元配列から行列を作成します。
    /// * TはDefaultとCopyトレイトを実装している必要があります。
    /// # Examples
    /// use NeuralNetwork::matrix::Matrix;
    /// assert_eq!(m.get(0, 0), Some(&1));
    /// assert_eq!(m.get(0, 1), Some(&2));
    /// assert_eq!(m.get(1, 0), Some(&3));
    /// assert_eq!(m.get(1, 1), Some(&4));
    /// ```
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

    /// サイズを指定して行列を作成します。
    /// * 要素は全てTのデフォルト値で初期化されます。
    /// * TはDefaultとCopyトレイトを実装している必要があります。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let m: Matrix<i32> = Matrix::with_size(2, 2);
    /// assert_eq!(m.get(0, 0), Some(&0));
    /// assert_eq!(m.get(0, 1), Some(&0));
    /// assert_eq!(m.get(1, 0), Some(&0));
    /// assert_eq!(m.get(1, 1), Some(&0));
    /// ```
    pub fn with_size(row: usize, col: usize) -> Self
    where
        T: Default + Copy
    {
        Self { data: vec![T::default(); row * col], rows: row, cols: col, _marker: std::marker::PhantomData }
    }

    /// 行数を返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let m: Matrix<i32> = Matrix::with_size(3, 4); // [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    /// assert_eq!(m.rows(), 3);
    /// ```
    pub fn rows(&self) -> usize { self.rows }

    /// 列数を返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let m: Matrix<i32> = Matrix::with_size(3, 4); // [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    /// assert_eq!(m.cols(), 4);
    /// ```
    pub fn cols(&self) -> usize { self.cols }

    /// 指定した位置の要素への参照を返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let m: Matrix<i32> = Matrix::with_size(2, 2);
    /// assert_eq!(m.get(0, 0), Some(&0));
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols { return None; }

        let idx = L::get_index(row, col, self.rows, self.cols)?;
        self.data.get(idx)
    }

    /// 指定した位置の要素への可変参照を返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let mut m: Matrix<i32> = Matrix::with_size(2, 2);
    /// if let Some(val) = m.get_mut(0, 0) {
    ///     *val = 42;
    /// }
    /// assert_eq!(m.get(0, 0), Some(&42));
    /// ```
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= self.rows || col >= self.cols { return None; }

        let idx = L::get_index(row, col, self.rows, self.cols)?;
        self.data.get_mut(idx)
    }

    /// 指定した行の要素への参照を返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let m: Matrix<i32> = Matrix::with_size(2, 2);
    /// assert_eq!(m.get_row(0).unwrap().collect::<Vec<&i32>>(), vec![&0, &0]);
    /// ```
    pub fn get_row(&self, row: usize) -> Option<impl Iterator<Item = &T>> {
        let (start, step) = L::row_stride(row, self.rows, self.cols)?;
        Some(self.data.iter().skip(start).step_by(step).take(self.cols))
    }

    /// 指定した行の要素への可変参照を返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let mut m: Matrix<i32> = Matrix::with_size(2, 2);
    /// if let Some(row_iter) = m.get_row_mut(0) {
    ///     row_iter.for_each(|val| *val = 42);
    /// }
    /// assert_eq!(m.get_row(0).unwrap().collect::<Vec<&i32>>(), vec![&42, &42]);
    /// ```
    pub fn get_row_mut(&mut self, row: usize) -> Option<impl Iterator<Item = &mut T>> {
        let (start, step) = L::row_stride(row, self.rows, self.cols)?;
        Some(self.data.iter_mut().skip(start).step_by(step).take(self.cols))
    }

    /// 指定した列の要素への参照を返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let m: Matrix<i32> = Matrix::with_size(2, 2);
    /// assert_eq!(m.get_column(0).unwrap().collect::<Vec<&i32>>(), vec![&0, &0]);
    /// ```
    pub fn get_column(&self, col: usize) -> Option<impl Iterator<Item = &T>> {
        let (start, step) = L::col_stride(col, self.rows, self.cols)?;
        Some(self.data.iter().skip(start).step_by(step).take(self.rows))
    }

    /// 指定した列の要素への可変参照を返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let mut m: Matrix<i32> = Matrix::with_size(2, 2);
    /// if let Some(col_iter) = m.get_column_mut(0) {
    ///    col_iter.for_each(|val| *val = 42);
    /// }
    /// assert_eq!(m.get_column(0).unwrap().collect::<Vec<&i32>>(), vec![&42, &42]);
    /// ```
    pub fn get_column_mut(&mut self, col: usize) -> Option<impl Iterator<Item = &mut T>> {
        let (start, step) = L::col_stride(col, self.rows, self.cols)?;
        Some(self.data.iter_mut().skip(start).step_by(step).take(self.rows))
    }
}

/// タプルインデックスで要素にアクセスできるようにします。
impl<T, L: MatrixLayout> std::ops::Index<(usize, usize)> for Matrix<T, L> {
    type Output = T;

    /// タプルインデックスで要素にアクセスします。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let m: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    /// assert_eq!(m[(0, 0)], 1);
    /// assert_eq!(m[(0, 1)], 2);
    /// assert_eq!(m[(1, 0)], 3);
    /// assert_eq!(m[(1, 1)], 4]);
    /// ```
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output{
        self.get(row, col).expect("Index out of bounds")
    }
}

impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Add<Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::AddAssign + Clone
{
    type Output = Option<Matrix<T, L>>;

    /// 同じサイズの行列同士の要素ごとの加算を行います。
    /// * サイズが異なる行列同士の加算はNoneを返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    /// let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
    /// let m3 = m1 + m2;
    /// assert_eq!(m3.unwrap().get(0, 0), Some(&6));
    /// assert_eq!(m3.unwrap().get(0, 1), Some(&8));
    /// assert_eq!(m3.unwrap().get(1, 0), Some(&10));
    /// assert_eq!(m3.unwrap().get(1, 1), Some(&12));
    /// ```
    fn add(mut self, other: Matrix<T, OL>) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols { return None; }

        for row in 0..self.rows {
            self.get_row_mut(row).unwrap().zip(other.get_row(row).unwrap()).for_each(|(a, b)| {
                *a += b.clone();
            });
        }

        Some(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::AddAssign<Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::AddAssign + Clone
{
    /// 同じサイズの行列同士の要素ごとの加算を行います。
    /// * サイズが異なる行列同士の加算はパニックを引き起こします。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let mut m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    /// let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
    /// m1 += m2;
    /// assert_eq!(m1.get(0, 0), Some(&6));
    /// assert_eq!(m1.get(0, 1), Some(&8));
    /// assert_eq!(m1.get(1, 0), Some(&10));
    /// assert_eq!(m1.get(1, 1), Some(&12));
    /// ```
    fn add_assign(&mut self, other: Matrix<T, OL>){
        assert!(self.rows == other.rows && self.cols == other.cols);

        for row in 0..self.rows {
            self.get_row_mut(row).unwrap().zip(other.get_row(row).unwrap()).for_each(|(a, b)| {
                *a += b.clone();
            });
        }
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Sub<Matrix<T, OL>> for Matrix<T, L> 
where 
    T: std::ops::SubAssign + Clone
{
    type Output = Option<Matrix<T, L>>;

    /// 同じサイズの行列同士の要素ごとの減算を行います。
    /// * サイズが異なる行列同士の減算はNoneを返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    /// let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
    /// let m3 = m1 - m2;
    /// assert_eq!(m3.unwrap().get(0, 0), Some(&-4));
    /// assert_eq!(m3.unwrap().get(0, 1), Some(&-4));
    /// assert_eq!(m3.unwrap().get(1, 0), Some(&-4));
    /// assert_eq!(m3.unwrap().get(1, 1), Some(&-4));
    /// ```
    fn sub(mut self, other: Matrix<T, OL>) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols { return None; }

        for row in 0..self.rows {
            self.get_row_mut(row).unwrap().zip(other.get_row(row).unwrap()).for_each(|(a, b)| {
                *a -= b.clone();
            });
        }

        Some(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::SubAssign<Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::SubAssign + Clone
{
    /// 同じサイズの行列同士の要素ごとの減算を行います。
    /// * サイズが異なる行列同士の減算はパニックを引き起こします。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let mut m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    /// let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
    /// m1 -= m2;
    /// assert_eq!(m1.get(0, 0), Some(&-4));
    /// assert_eq!(m1.get(0, 1), Some(&-4));
    /// assert_eq!(m1.get(1, 0), Some(&-4));
    /// assert_eq!(m1.get(1, 1), Some(&-4));
    /// ```
    fn sub_assign(&mut self, other: Matrix<T, OL>){
        assert!(self.rows == other.rows && self.cols == other.cols);

        for row in 0..self.rows {
            self.get_row_mut(row).unwrap().zip(other.get_row(row).unwrap()).for_each(|(a, b)| {
                *a -= b.clone();
            });
        }
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Mul<Matrix<T, OL>> for Matrix<T, L> 
where 
    T: std::ops::MulAssign + Clone
{
    type Output = Option<Matrix<T, L>>;

    /// 同じサイズの行列同士の要素ごとの乗算を行います。
    /// * サイズが異なる行列同士の乗算はNoneを返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    /// let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
    /// let m3 = m1 * m2;
    /// assert_eq!(m3.unwrap().get(0, 0), Some(&5));
    /// assert_eq!(m3.unwrap().get(0, 1), Some(&12));
    /// assert_eq!(m3.unwrap().get(1, 0), Some(&21));
    /// assert_eq!(m3.unwrap().get(1, 1), Some(&32));
    /// ```
    fn mul(mut self, other: Matrix<T, OL>) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols { return None; }

        for row in 0..self.rows {
            self.get_row_mut(row).unwrap().zip(other.get_row(row).unwrap()).for_each(|(a, b)| {
                *a *= b.clone();
            });
        }

        Some(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::MulAssign<Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::MulAssign + Clone
{
    /// 同じサイズの行列同士の要素ごとの乗算を行います。
    /// * サイズが異なる行列同士の乗算はパニックを引き起こします。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let mut m1: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    /// let m2: Matrix<i32> = Matrix::new([[5, 6], [7, 8]]);
    /// m1 *= m2;
    /// assert_eq!(m1.get(0, 0), Some(&5));
    /// assert_eq!(m1.get(0, 1), Some(&12));
    /// assert_eq!(m1.get(1, 0), Some(&21));
    /// assert_eq!(m1.get(1, 1), Some(&32));
    /// ```
    fn mul_assign(&mut self, other: Matrix<T, OL>){
        assert!(self.rows == other.rows && self.cols == other.cols);

        for row in 0..self.rows {
            self.get_row_mut(row).unwrap().zip(other.get_row(row).unwrap()).for_each(|(a, b)| {
                *a *= b.clone();
            });
        }
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::Div<Matrix<T, OL>> for Matrix<T, L> 
where 
    T: std::ops::DivAssign + Clone + PartialEq + Default
{
    type Output = Option<Matrix<T, L>>;

    /// 同じサイズの行列同士の要素ごとの除算を行います。
    /// * サイズが異なる行列同士の除算はNoneを返します。
    /// * 0で割る要素がある場合もNoneを返します。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let m1: Matrix<i32> = Matrix::new([[10, 20], [30, 40]]);
    /// let m2: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    /// let m3 = m1 / m2;
    /// assert_eq!(m3.unwrap().get(0, 0), Some(&10));
    /// assert_eq!(m3.unwrap().get(0, 1), Some(&10));
    /// assert_eq!(m3.unwrap().get(1, 0), Some(&10));
    /// assert_eq!(m3.unwrap().get(1, 1), Some(&10));
    /// ```
    fn div(mut self, other: Matrix<T, OL>) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols { return None; }
        if other.data.iter().any(|val| *val == T::default()) { return None; }

        for row in 0..self.rows {
            self.get_row_mut(row).unwrap().zip(other.get_row(row).unwrap()).for_each(|(a, b)| {
                *a /= b.clone();
            });
        }

        Some(self)
    }
}
impl<T, L: MatrixLayout, OL: MatrixLayout> std::ops::DivAssign<Matrix<T, OL>> for Matrix<T, L> 
where  
    T: std::ops::DivAssign + Clone + PartialEq + Default
{
    /// 同じサイズの行列同士の要素ごとの除算を行います。
    /// * サイズが異なる行列同士の除算はパニックを引き起こします。
    /// * 0で割る要素がある場合もパニックを引き起こします。
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    /// let mut m1: Matrix<i32> = Matrix::new([[10, 20], [30, 40]]);
    /// let m2: Matrix<i32> = Matrix::new([[1, 2], [3, 4]]);
    /// m1 /= m2;
    /// assert_eq!(m1.get(0, 0), Some(&10));
    /// assert_eq!(m1.get(0, 1), Some(&10));
    /// assert_eq!(m1.get(1, 0), Some(&10));
    /// assert_eq!(m1.get(1, 1), Some(&10));
    /// ```
    fn div_assign(&mut self, other: Matrix<T, OL>){
        assert!(self.rows == other.rows && self.cols == other.cols);
        assert!(other.data.iter().all(|val| *val != T::default()));

        for row in 0..self.rows {
            self.get_row_mut(row).unwrap().zip(other.get_row(row).unwrap()).for_each(|(a, b)| {
                *a /= b.clone();
            });
        }
    }
}

impl<T: std::fmt::Display, L: MatrixLayout> std::fmt::Display for Matrix<T, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (0..self.rows).for_each(|r| {
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