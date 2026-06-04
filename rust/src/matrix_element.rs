pub trait MatrixElement: Default + Copy + Sync + Send {}
impl<T> MatrixElement for T where T: Default + Copy + Sync + Send {}