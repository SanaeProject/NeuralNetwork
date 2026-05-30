pub struct MatrixArray<T, const ROW_MAJOR: bool = true>{
    data: Vec<T>,
    row: usize,
    col: usize
}
impl<T> MatrixArray<T>
where T: Default, T: Clone
{
    pub fn with_size(row: usize, col: usize) -> Self{
        Self { 
            data: vec![T::default(); row*col], 
            row: row, 
            col: col 
        }
    }
}
impl<T> MatrixArray<T, true>
{
    pub fn with_data<const ROW: usize,const COL: usize>(array_2d: [[T; COL];ROW]) -> Self
    where T: ToOwned<Owned = T>
    {
        Self { 
            data: array_2d.into_iter().flatten().collect::<Vec<T>>(), 
            row: ROW, 
            col: COL
        }
    }
    pub fn get(&self, row: usize, col: usize)->Option<&T>{
        let idx = row*self.col + col;
        self.data.get(idx)
    }
}
impl<T> MatrixArray<T, false>
{
    pub fn with_data<const ROW: usize,const COL: usize>(array_2d: [[T; COL];ROW]) -> Self
    where T: ToOwned<Owned = T>, T: Clone
    {
        let array_2d_ref = &array_2d;
        let data: Vec<T> = (0..COL)
            .flat_map(move |c| {
                (0..ROW).map(move |r| array_2d_ref[r][c].clone())
            })
            .collect();

        Self { 
            data, 
            row: ROW, 
            col: COL
        }
    }
    pub fn get(&self, row: usize, col: usize)->Option<&T>{
        let idx = col * self.row + row;
        self.data.get(idx)
    }
}