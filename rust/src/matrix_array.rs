use core::fmt;

pub struct MatrixArray<T, const ROW_MAJOR: bool = true> {
    data: Vec<T>,
    row: usize,
    col: usize,
}

impl<T, const ROW_MAJOR: bool> MatrixArray<T, ROW_MAJOR> {
    pub fn with_size(row: usize, col: usize) -> Self
    where
        T: Default + Clone,
    {
        Self {
            data: vec![T::default(); row * col],
            row,
            col,
        }
    }
}

// ==========================================
// 行優先 (Row-Major) の実装
// ==========================================
impl<T> MatrixArray<T, true> {
    pub fn with_data<const ROW: usize, const COL: usize>(array_2d: [[T; COL]; ROW]) -> Self {
        Self {
            // 💡 into_iter() を使っているので ToOwned の制約自体を削除できます！
            data: array_2d.into_iter().flatten().collect::<Vec<T>>(),
            row: ROW,
            col: COL,
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        let idx = row * self.col + col;
        self.data.get(idx)
    }

    pub fn get_mut_row(&mut self, row: usize) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut().skip(self.col * row).take(self.col)
    }
    pub fn get_mut_column(&mut self, col: usize) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut().skip(col).step_by(self.col).take(self.row)
    }

    pub fn get_row(&self, row: usize) -> impl Iterator<Item = &T> {
        self.data.iter().skip(self.col * row).take(self.col)
    }
    pub fn get_column(&self, col: usize) -> impl Iterator<Item = &T> {
        self.data.iter().skip(col).step_by(self.col).take(self.row)
    }
}

// ==========================================
// 列優先 (Column-Major) の実装
// ==========================================
impl<T> MatrixArray<T, false> {
    pub fn with_data<const ROW: usize, const COL: usize>(array_2d: [[T; COL]; ROW]) -> Self
    where
        T: Clone,
    {
        let array_2d_ref = &array_2d;
        let data: Vec<T> = (0..COL)
            .flat_map(move |c| (0..ROW).map(move |r| array_2d_ref[r][c].clone()))
            .collect();

        Self {
            data,
            row: ROW,
            col: COL,
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        let idx = col * self.row + row;
        self.data.get(idx)
    }

    pub fn get_mut_row(&mut self, row: usize) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut().skip(row).step_by(self.row).take(self.col)
    }
    pub fn get_mut_column(&mut self, col: usize) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut().skip(col * self.row).take(self.row)
    }

    pub fn get_row(&self, row: usize) -> impl Iterator<Item = &T> {
        self.data.iter().skip(row).step_by(self.row).take(self.col)
    }
    pub fn get_column(&self, col: usize) -> impl Iterator<Item = &T> {
        self.data.iter().skip(col * self.row).take(self.row)
    }
}

// ==========================================
// トレイト実装
// ==========================================
impl<T: fmt::Display> fmt::Display for MatrixArray<T, true> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::new();

        for r in 0..self.row {
            for c in self.get_row(r) {
                result.push_str(&format!("{}\t", c));
            }
            result.push('\n');
        }

        write!(f, "{}", result)
    }
}
impl<T: fmt::Display> fmt::Display for MatrixArray<T, false> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::new();

        for r in 0..self.row {
            for c in self.get_row(r) {
                result.push_str(&format!("{}\t", c));
            }
            result.push('\n');
        }

        write!(f, "{}", result)
    }
}