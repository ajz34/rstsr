use num::Integer;

/// Slicing for python (numpy) convention; somehow similar to Rust's range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice<T>
where
    T: Integer + Clone,
{
    pub(crate) start: Option<T>,
    pub(crate) stop: Option<T>,
    pub(crate) step: Option<T>,
}

/// In most cases, we will use isize for indexing.
pub type SliceI = Slice<isize>;

impl<T> Slice<T>
where
    T: Integer + Clone,
{
    pub fn new(
        start: impl Into<Option<T>>,
        stop: impl Into<Option<T>>,
        step: impl Into<Option<T>>,
    ) -> Self {
        Self { start: start.into(), stop: stop.into(), step: step.into() }
    }

    pub fn start(&self) -> Option<T> {
        self.start.clone()
    }

    pub fn stop(&self) -> Option<T> {
        self.stop.clone()
    }

    pub fn step(&self) -> Option<T> {
        self.step.clone()
    }
}

macro_rules! impl_from_slice {
    ($($t:ty),*) => {
        $(
            impl From<Slice<$t>> for Slice<isize> {
                fn from(slice: Slice<$t>) -> Self {
                    Self {
                        start: slice.start.map(|v| v as isize),
                        stop: slice.stop.map(|v| v as isize),
                        step: slice.step.map(|v| v as isize),
                    }
                }
            }
        )*
    };
}

impl_from_slice!(usize, u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);

impl<T> From<std::ops::Range<T>> for SliceI
where
    T: Integer + Clone,
    Slice<T>: Into<SliceI>,
{
    fn from(range: std::ops::Range<T>) -> Self {
        Slice::<T> { start: Some(range.start), stop: Some(range.end), step: None }.into()
    }
}

impl<T> From<std::ops::RangeFrom<T>> for SliceI
where
    T: Integer + Clone,
    Slice<T>: Into<SliceI>,
{
    fn from(range: std::ops::RangeFrom<T>) -> Self {
        Slice::<T> { start: Some(range.start), stop: None, step: None }.into()
    }
}

impl<T> From<std::ops::RangeTo<T>> for SliceI
where
    T: Integer + Clone,
    Slice<T>: Into<SliceI>,
{
    fn from(range: std::ops::RangeTo<T>) -> Self {
        Slice::<T> { start: None, stop: Some(range.end), step: None }.into()
    }
}

impl From<std::ops::RangeFull> for SliceI {
    fn from(_: std::ops::RangeFull) -> Self {
        SliceI { start: None, stop: None, step: None }
    }
}
