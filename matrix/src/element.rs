pub trait MatrixElement: Default + Copy + Sync + Send {}
impl MatrixElement for i8 {}
impl MatrixElement for i16 {}
impl MatrixElement for i32 {}
impl MatrixElement for i64 {}
impl MatrixElement for i128 {}
impl MatrixElement for u8 {}
impl MatrixElement for u16 {}
impl MatrixElement for u32 {}
impl MatrixElement for u64 {}
impl MatrixElement for u128 {}
impl MatrixElement for f32 {}
impl MatrixElement for f64 {}