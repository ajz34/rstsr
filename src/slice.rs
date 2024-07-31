use num::Integer;

/// Slicing for python (numpy) convention; somehow similar to Rust's range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice<T>
where
    T: Integer + Clone,
{
    start: Option<T>,
    stop: Option<T>,
    step: Option<T>,
}

impl<T> Slice<T>
where
    T: Integer + Clone,
{
    pub fn new(start: Option<T>, stop: Option<T>, step: Option<T>) -> Self {
        Self { start, stop, step }
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

impl<T> From<std::ops::Range<T>> for Slice<T>
where
    T: Integer + Clone,
{
    fn from(range: std::ops::Range<T>) -> Self {
        Self {
            start: Some(range.start),
            stop: Some(range.end),
            step: None,
        }
    }
}

#[macro_export]
macro_rules! slice {
    ($stop:expr) => {
        Slice::<isize>::from(
            Slice { start: None, stop: $stop.into(), step: None }
        )
    };
    ($start:expr, $stop:expr) => {
        Slice::<isize>::from(
            Slice { start: $start.into(), stop: $stop.into(), step: None }
        )
    };
    ($start:expr, $stop:expr, $step:expr) => {
        Slice::<isize>::from(
            Slice { start: $start.into(), stop: $stop.into(), step: $step.into() }
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice() {
        let s = slice!(1, 2, 3);
        assert_eq!(s.start(), Some(1));
        assert_eq!(s.stop(), Some(2));
        assert_eq!(s.step(), Some(3));
    }
}
