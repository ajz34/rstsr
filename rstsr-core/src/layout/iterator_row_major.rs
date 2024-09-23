//! Layout (double-ended) iterator; only row-major iterator is implemented.

use crate::prelude_dev::*;

/// Layout iterator (row-major).
///
/// This iterator only handles row-major iterator.
///
/// # Note
///
/// This crate implements row-major iterator only; the layout iterator that
/// actaully works is internal realization; though it's public struct, it is not
/// intended to be exposed to user.
#[derive(Debug, Clone)]
pub struct IterLayoutRowMajor<D>
where
    D: DimDevAPI,
{
    layout: Layout<D>,

    index_start: D, // this is not used for buffer-order
    iter_start: usize,
    offset_start: usize,

    index_end: D, // this is not used for buffer-order
    iter_end: usize,
    offset_end: usize,
}



