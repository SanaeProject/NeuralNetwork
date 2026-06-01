pub trait MatrixLayout {
    /// row, colから一次元配列のindexを取得する
    /// # 引数
    /// - `row`: 行番号
    /// - `col`: 列番号
    /// - `matrix_row`: 行数
    /// - `matrix_col`: 列数
    /// # 戻り値
    /// - 一次元配列のindex
    fn get_index(row: usize, col: usize, matrix_row: usize, matrix_col: usize) -> Option<usize>;

    /// row, colから行・列のstrideを取得する
    /// # 引数
    /// - `row`: 行番号
    /// - `matrix_row`: 行数
    /// - `matrix_col`: 列数
    /// # 戻り値
    /// - (開始位置, ステップ)のタプル
    fn row_stride(row: usize, matrix_row: usize, matrix_col: usize) -> Option<(usize, usize)>;

    /// row, colから行・列のstrideを取得する
    /// # 引数
    /// - `col`: 列番号
    /// - `matrix_row`: 行数
    /// - `matrix_col`: 列数
    /// # 戻り値
    /// - (開始位置, ステップ)のタプル
    fn col_stride(col: usize, matrix_row: usize, matrix_col: usize) -> Option<(usize, usize)>;
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
}