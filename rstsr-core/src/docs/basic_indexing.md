Basic slicing and indexing of a tensor, in the sense of NumPy's *basic
indexing*: each axis of the tensor is indexed by a slice (`1..4`), an integer
(`2`), a new-axis marker (`None`), or [`Ellipsis`], and the result is a
**view** of the original data - no element is ever copied.

The preferred call form is the short associated method [`TensorAny::i`] with a
tuple of indexers:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let m = rt::arange((12, &device)).into_shape([3, 4]);
let second_row  = m.i(1);              // one indexer: the first axis
let top_right   = m.i((0..2, 2..4));   // tuple: one indexer per axis
let last_column = m.i((Ellipsis, -1)); // ellipsis + integer
let expanded    = m.i((.., None));     // None: insert a new axis
# assert_eq!(second_row.to_vec(), vec![4, 5, 6, 7]);
# assert_eq!(top_right.shape(), &[2, 2]);
# assert_eq!(last_column.to_vec(), vec![3, 7, 11]);
# assert_eq!(expanded.shape(), &[3, 1, 4]);
```

The free function `rt::slice(&tensor, index)` and the method
[`TensorAny::slice`] are the same operation; [`TensorAny::i`] is its alias.
For writing through a slice, [`TensorAny::i_mut`] takes the same indexers and
returns a mutable view (see
[#Writing through a mutable slice](#writing-through-a-mutable-slice) below).
The fallible forms return `Result` (see
[#Variants of this function](#variants-of-this-function)).

This function behaves identically under [`RowMajor`] and [`ColMajor`] device default orders.

# Overloads Table

The `index` argument accepts anything convertible into
[`AxesIndex<Indexer>`]. Two layers of overloading exist: the form of
the *whole* index, and the type of each *per-axis indexer* inside it.

## Per-axis indexers

Each axis can be indexed by one of the following. Inside tuples and the
[`s!`] macro, different indexer types can be mixed freely.

| Per-axis indexer | Written as | Effect on the axis |
|--|--|--|
| Rust range | `1..4`, `1..`, `..4`, `..` | kept, narrowed (Python slicing rules) |
| stepped slice ([`slice!`] macro) | `slice!(1, 7, 2)`, `slice!(None, None, -1)` | kept, strided; a negative step reverses the axis |
| integer (`i32`/`isize`/`usize`, ...) | `2`, `-1` | one position selected, **axis dropped** |
| `None` (or [`NewAxis`]) | `None` | new axis of size 1 inserted |
| [`Ellipsis`] | `Ellipsis` | expands to as many `..` as needed |

## Whole-index forms

- `m.i((.., 1..3))`: a tuple of 1 to 10 per-axis indexers, one per axis.
  Fewer indexers than axes means the trailing axes are taken in full (as in
  NumPy); this is the **preferred form**.
- `a.i(1..4)`, `a.i(2)`, `a.i(None)`, `a.i(slice!(1, 7, 2))`: a single
  per-axis indexer, applied to the first axis; the same as a 1-tuple written
  without the parentheses.
- `m.i(s![1..3, 2])`: the [`s!`] macro, producing `&[Indexer]`. Accepts
  runtime values and mixed indexer types.
- `m.i([1, 2])`, `m.i(vec![0, 2])`: an array or `Vec` of one *homogeneous*
  indexer type (e.g. all integers); useful when the number of indexers is only
  known at runtime.

# Parameters

- `tensor`: [`&TensorAny<R, T, B, D>`](TensorAny): the tensor to slice.
- `index`: the per-axis indexers; see the
  [#Overloads Table](#overloads-table) for all accepted forms.

# Returns

- [`TensorView`]: a view of the sliced tensor.
  - No data is copied; only shape, stride and offset change.
  - The returned dimensionality is dynamic (`IxD`) even for statically-dimensioned
    inputs; use [`into_dim`] to recover a static dimensionality.
  - Axes selected by integers are dropped; axes marked `None`/[`NewAxis`] are added.

# Examples

First steps on a one-dimensional tensor - a range narrows the axis, an integer
selects one element (and returns a 0-dim view, which prints as a bare value),
negative bounds count from the back:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((10, &device));
println!("{}", a.i(1..4));
// [ 1 2 3]
println!("{}", a.i((1..4,))); // a 1-tuple indexes the first axis, too
// [ 1 2 3]
println!("{}", a.i(2));
// 2
println!("{}", a.i(-3..));
// [ 7 8 9]
# assert_eq!(format!("{}", a.i(1..4)), "[ 1 2 3]");
# assert_eq!(a.i(2).shape(), &[] as &[usize]);
```

For a matrix, one indexer per axis, written as a tuple. `..` takes the whole
axis; an integer selects one row/column and drops that axis:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let m = rt::arange((12, &device)).into_shape([3, 4]);
println!("{}", m.i(1));
// [ 4 5 6 7]
println!("{}", m.i((.., 1..3)));
// [[ 1 2]
//  [ 5 6]
//  [ 9 10]]
println!("{}", m.i((1..3, ..)));
// [[ 4 5 6 7]
//  [ 8 9 10 11]]
```

`None` inserts a new axis of size one. Two tensors so aligned broadcast
element-wise - here building an outer sum, as in NumPy's
`x[:, None] + x[None, :]`:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let m = rt::arange((12, &device)).into_shape([3, 4]);
println!("{:?}", m.i((.., None)).shape());
// [3, 1, 4]

let v = rt::arange((5, &device));
let outer = &v.i((.., None)) + &v.i((None, ..));
println!("{}", outer);
// [[ 0 1 2 3 4]
//  [ 1 2 3 4 5]
//  [ 2 3 4 5 6]
//  [ 3 4 5 6 7]
//  [ 4 5 6 7 8]]
# assert_eq!(outer.shape(), &[5, 5]);
```

Ranges in Rust cannot carry a step; stepped slices are written with the
[`slice!`] macro, where any bound may be `None`:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((10, &device));
println!("{}", a.i(slice!(2, 7, 2)));
// [ 2 4 6]

let m = rt::arange((12, &device)).into_shape([3, 4]);
println!("{}", m.i((.., slice!(None, None, 2))));
// [[ 0 2]
//  [ 4 6]
//  [ 8 10]]
```

# Elaborated examples

## Ranges: open ends, negative bounds, clamping

All four Rust range forms work per axis. Bounds follow Python rules: negative
values count from the back of the axis, and a stop past the end is clamped
(which can produce an empty axis rather than an error):

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((10, &device));
println!("{}", a.i(2..8));
// [ 2 3 4 5 6 7]
println!("{}", a.i(2..));
// [ 2 3 4 5 6 7 8 9]
println!("{}", a.i(..3));
// [ 0 1 2]
println!("{}", a.i(..));
// [ 0 1 2 ... 7 8 9]
// (long outputs are truncated with an ellipsis in the display)

println!("{}", a.i(7..20)); // stop past the end: clamped
// [ 7 8 9]
println!("{}", a.i(3..3).size()); // empty, not an error
// 0
# assert_eq!(a.i(7..20).shape(), &[3]);
# assert_eq!(a.i(3..3).shape(), &[0]);
```

## Stepped slices and reversing an axis

`slice!(start, stop, step)` is rstsr's spelling of Python's `start:stop:step`.
With a negative step the axis is traversed towards smaller indices;
`slice!(None, None, -1)` is the idiomatic axis reversal. Defaults are as in
Python: `start` defaults to `0` (`n - 1` for a negative step), `stop` to `n`
(`-1`), and `step` to `1`. `step == 0` is an error (see [#Panics](#panics)).

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((10, &device));
println!("{}", a.i(slice!(1, 7, 2))); // NumPy x[1:7:2]
// [ 1 3 5]
println!("{}", a.i(slice!(None, None, -1))); // reversed
// [ 9 8 7 ... 2 1 0]
println!("{}", a.i(slice!(-1, None, -2))); // every second element, from the back
// [ 9 7 5 3 1]
# assert_eq!(a.i(slice!(1, 7, 2)).to_vec(), vec![1, 3, 5]);
# assert_eq!(a.i(slice!(-1, None, -2)).to_vec(), vec![9, 7, 5, 3, 1]);
```

## Integer selection drops the axis

An integer indexer selects one position along the axis and removes that axis
from the result - exactly like NumPy's `x[1]`. Use a length-one range
(`1..2`) when the axis should be *kept*. A tuple of integers indexes every
axis and yields a 0-dim view; read it with [`TensorAny::to_scalar`] or the
`[]` operator ([`Index`]), which is boundary-checked scalar access:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let m = rt::arange((12, &device)).into_shape([3, 4]);
println!("{:?}", m.i(1..2).shape()); // range keeps the axis
// [1, 4]
println!("{:?}", m.i(1).shape()); // integer drops it
// [4]
println!("{}", m.i((-1, ..))); // negative: last row
// [ 8 9 10 11]

let b = rt::arange((24, &device)).into_shape([2, 3, 4]);
println!("{}", b.i((1, 2, 3))); // 0-dim view of one element
// 23
assert_eq!(b.i((1, 2, 3)).to_scalar(), 23);
assert_eq!(b[[1, 2, 3]], 23); // same value through the [] operator
```

## New axes: `None` and [`NewAxis`]

`None` inserts an axis of size one; [`NewAxis`] is the named spelling of the
same indexer, useful where `Option` inference is ambiguous. The new axis lands
exactly where NumPy puts it - compare with `x[1, None, :]` and friends:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((10, &device));
println!("{:?}", a.i(None).shape()); // single None: axis at the front
// [1, 10]

let b = rt::arange((24, &device)).into_shape([2, 3, 4]);
println!("{:?}", b.i((1, NewAxis, ..)).shape()); // NumPy x[1, None, :]
// [1, 3, 4]
println!("{:?}", b.i((None, 1)).shape()); // NumPy x[None, 1]
// [1, 3, 4]
println!("{:?}", b.i((1, None, Ellipsis, 2)).shape()); // NumPy x[1, None, ..., 2]
// [1, 3]
```

## [`Ellipsis`]

[`Ellipsis`] stands for "as many whole-axis ranges as needed", so a long
trailing `.., .., ..` can be collapsed - the natural way to index the last
axes of a high-dimensional tensor. At most one [`Ellipsis`] is allowed. It
composes with every other indexer:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let b = rt::arange((24, &device)).into_shape([2, 3, 4]);
println!("{:?}", b.i((Ellipsis, 2)).shape()); // NumPy x[..., 2]
// [2, 3]
println!("{:?}", b.i((1, Ellipsis)).shape()); // NumPy x[1, ...]
// [3, 4]
# assert_eq!(b.i((Ellipsis, 2)).shape(), &[2, 3]);
# assert_eq!(b.i((1, Ellipsis)).shape(), &[3, 4]);
```

## Slicing returns a view

`i()` only rewrites the layout - shape, stride, offset - and borrows the data.
A step multiplies the axis stride; an integer select advances the offset. The
`Debug` format shows the resulting layout:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let m = rt::arange((12, &device)).into_shape([3, 4]);
println!("{:?}", m.i((.., 2)));
// === Debug Tensor Print ===
// [ 2 6 10]
// DeviceCpuSerial { default_order: C }
// 1-Dim (dyn), contiguous: Custom
// shape: [3], stride: [4], offset: 2
// Type: rstsr_core::tensorbase::TensorBase<rstsr_core::storage::device::Storage<rstsr_core::storage::data::DataRef<'_, alloc::vec::Vec<i32>>, i32, rstsr_core::device_cpu_serial::device::DeviceCpuSerial>, alloc::vec::Vec<usize>>
// ==========================
# assert_eq!(m.i((.., 2)).stride(), &[4]);
# assert_eq!(m.i((.., 2)).offset(), 2);
```

A strided view is typically not contiguous. When an algorithm needs
contiguous input, materialize a copy with [`to_contig`]; the view itself stays
zero-cost:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let b = rt::arange((24, &device)).into_shape([2, 3, 4]);
let s = b.i((.., .., slice!(None, None, 2)));
assert!(!s.c_contig());
let t = s.to_contig(RowMajor);
assert!(t.c_contig());
assert_eq!(t.shape(), &[2, 3, 2]);
```

Because a view only borrows the parent, the Rust type system enforces what
NumPy documents as a caveat: a small slice keeps the parent's buffer alive
(the view's lifetime is tied to the parent's). To slice an owned tensor and
keep its ownership kind, use [`into_slice`]:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let d = rt::arange((6, &device));
let d0 = d.into_slice(1..3); // still an owned tensor
println!("{}", d0);
// [ 1 2]
```

Slicing is oblivious to the memory layout: the same logical tensor stored
column-major slices to the same values and shapes.

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let m = rt::arange((12, &device)).into_shape([3, 4]);
let m_f = m.to_contig(ColMajor); // same values, F-contiguous storage
assert!(m_f.f_contig());
println!("{}", m_f.i((.., 1..3)));
// [[ 1 2]
//  [ 5 6]
//  [ 9 10]]
```

## Chaining slices (views of views)

Slicing a view yields another view of the same data, at zero cost. Note that
intermediate views must be bound to a variable - a chained
`t.i(..).i(..)` does not compile, because the intermediate view is a
temporary (see [#Tips on common compilation errors](#tips-on-common-compilation-errors)):

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let b = rt::arange((24, &device)).into_shape([2, 3, 4]);
let v = b.i((1, ..)); // 3x4
let c = v.i((.., 1..3)); // 3x2
println!("{}", c);
// [[ 13 14]
//  [ 17 18]
//  [ 21 22]]
# assert_eq!(c.shape(), &[3, 2]);
```

When the whole index is known at once, prefer a single combined tuple indexer
over chained steps - it builds one layout directly and reads clearer:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let b = rt::arange((24, &device)).into_shape([2, 3, 4]);
let c2 = b.i((1, .., 1..3)); // the same view as the chained example above
assert_eq!(c2.shape(), &[3, 2]);
# assert_eq!(format!("{c2}"), "[[ 13 14]\n [ 17 18]\n [ 21 22]]");
```

## Writing through a mutable slice

[`i_mut`](slice_mut()) returns a [`TensorMut`] that writes through to the
parent tensor. For a compound assignment on a freshly returned view, Rust
needs a place expression: spell it `*&mut t.i_mut(..) += ...` (the same
consideration as [#Tips on common compilation errors](#tips-on-common-compilation-errors)):

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let mut a: Tensor<i32, _> = rt::arange((5, &device));
let mut v = a.i_mut(1..4);
v += 10;
println!("{a}");
// [ 0 11 12 13 4]

let mut b: Tensor<i32, _> = rt::arange((6, &device)).into_shape([2, 3]);
*&mut b.i_mut((1, ..)) += 10;
println!("{b}");
// [[ 0 1 2]
//  [ 13 14 15]]
```

As in NumPy, slicing can overwrite regions but never grows a tensor; the
written values must be broadcastable to the sliced shape.

## Indexing with runtime values

Indexers do not have to be literals: tuples accept runtime values of any
integer type directly, so computed positions plug in as-is. The [`s!`] macro
is equally accepted. Reach for an owned `Vec` (not `&Vec`) of one integer
type only when the *number* of indexers is itself unknown at compile time,
since tuple arities top out at 10:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let m = rt::arange((12, &device)).into_shape([3, 4]);
let i: usize = 2;
let j: usize = 0;
println!("{}", m.i((i, j))); // m[2, 0] as a 0-dim view
// 8

let p: usize = 1;
let q: isize = -2;
println!("{}", m.i(s![p, q]));
// 6

let b = rt::arange((24, &device)).into_shape([2, 3, 4]);
let idx: Vec<isize> = vec![1, -1, 2]; // three axes, count known only at runtime
println!("{}", b.i(idx));
// 22
# assert_eq!(m.i((2_usize, 0)).to_scalar(), 8);
# assert_eq!(b.i(vec![1_isize, -1, 2]).to_scalar(), 22);
```

# Ownership Semantics between [`slice`](slice()), [`slice_mut`] and [`into_slice`]

All three never copy data; they differ in borrows and ownership only.

| Function | Input | Output | Data copied |
|--|--|--|--|
| [`slice`](slice()) / [`TensorAny::i`] | borrowed [`&TensorAny`](TensorAny) | [`TensorView`] (borrow) | never |
| [`slice_mut`] / [`TensorAny::i_mut`] | mutably borrowed `&mut` [`TensorAny`] | [`TensorMut`] (mutable borrow) | never |
| [`into_slice`] | owned [`TensorAny`] | same ownership kind as input | never |

# Tips on common compilation errors

Rust's inclusive ranges (`1..=4`, `..=4`) are not indexers; bounds are
exclusive as in Python. Use an exclusive end, or [`slice!`] for open bounds:

```compile_fail
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((10, &device));
let v = a.i(1..=4);
```

```text
error[E0277]: the trait bound `AxesIndex<Indexer>: From<RangeInclusive<{integer}>>` is not satisfied
   |
   |     let v = a.i(1..=4);
   |               - ^^^^^ the trait `From<RangeInclusive<{integer}>>` is not implemented
   |               |
   |               = help: the following other types implement trait `From<T>`:
   |                       `AxesIndex<T>` implements `From<T>`
   |                       `AxesIndex<T>` implements `From<[T; N]>`
   |                       ...
```

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((10, &device));
let v = a.i(1..5); // 1..=4 written with an exclusive end
println!("{v}");
// [ 1 2 3 4]
```

A `Vec` index is consumed by value; `&Vec` is not an accepted form. Pass the
owned `Vec`, clone it if it is reused, or use [`s!`] with runtime values:

```compile_fail
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let m = rt::arange((12, &device)).into_shape([3, 4]);
let idx: Vec<usize> = vec![0, 2];
let v = m.i(&idx);
```

```text
error[E0277]: the trait bound `AxesIndex<Indexer>: From<&Vec<usize>>` is not satisfied
   |
   |     let v = m.i(&idx);
   |               - ^^^^ the trait `From<&Vec<usize>>` is not implemented
   |               |
   |               = help: the following other types implement trait `From<T>`:
   |                       `AxesIndex<T>` implements `From<T>`
   |                       `AxesIndex<T>` implements `From<[T; N]>`
   |                       ...
```

Compound assignment needs a place on the left-hand side; a method call is a
value. Write `*&mut t.i_mut(..) += ...`, or bind the view first (as in
[#Writing through a mutable slice](#writing-through-a-mutable-slice)):

```compile_fail
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let mut m: Tensor<i32, _> = rt::arange((12, &device)).into_shape([3, 4]);
m.i_mut(1..2) += 10;
```

```text
error[E0067]: invalid left-hand side of assignment
   |
   |     m.i_mut(1..2) += 10;
   |     ------------- ^^
   |     |
   |     cannot assign to this expression
```

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let mut m: Tensor<i32, _> = rt::arange((12, &device)).into_shape([3, 4]);
*&mut m.i_mut(1..2) += 10; // correct: *&mut makes the temporary a place
# assert_eq!(m.i((1, ..)).to_vec(), vec![14, 15, 16, 17]);
```

Chaining slicing calls on temporaries fails borrow checking, because the
intermediate view borrows the receiver of the next call. Bind each step to a
variable (see [#Chaining slices (views of views)](#chaining-slices-views-of-views)):

```compile_fail
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let b = rt::arange((24, &device)).into_shape([2, 3, 4]);
let c = b.i((1, ..)).i((.., 1..3));
println!("{c}");
```

```text
error[E0716]: temporary value dropped while borrowed
   |
   |     let c = b.i((1, ..)).i((.., 1..3));
   |             ^^^^^^^^^^^^^            - temporary value is freed at the end of this statement
   |             |
   |             = help: consider using a `let` binding to create a longer lived value
```

# Notes of API accordance

- Array-API: basic indexing via `x[indices]` (`__getitem__`) ([`indexing`](https://data-apis.org/array-api/2024.12/API_specification/indexing.html))
- NumPy: basic slicing and indexing, `x[1:4]`, `x[2]`, `x[:, None]` ([`numpy basics.indexing`](https://numpy.org/doc/stable/user/basics.indexing.html#basic-slicing-and-indexing))
- RSTSR: `tensor.i(index)` (preferred), `tensor.slice(index)`, or `rt::slice(&tensor, index)`.

RSTSR implements NumPy's basic-indexing semantics (positional per-axis
indexers, negative bounds, clamping, integer-axis removal, `None`/`Ellipsis`
placement, fewer indexers than axes) and every rule verified against NumPy
2.5.2 matches. The surface differs, being Rust instead of bracket syntax:

- Indexers are function arguments - a tuple of per-axis indexers, or a single
  indexer - rather than comma-separated entries inside `[]`. Stepped slices
  are written with the [`slice!`] macro (`slice!(start, stop, step)`), since
  Rust ranges carry no step.
- Inclusive bounds (`1..=4`) are not supported; ends are exclusive as in
  Python (`1..5`).
- `None` is the new-axis marker and the only meaningful `Option` value: a
  `Some(n)` indexer compiles but always panics. The named forms [`NewAxis`]
  and [`Ellipsis`] also exist.
- Slicing returns a dynamic-dimensionality view ([`TensorView`] with `IxD`).

# Panics

- Panics if an integer indexer is out of bounds for its axis (e.g. `a.i(10)`
  on a length-10 axis); negative integers are normalized first, so `a.i(-1)`
  is the last position.
- Panics if the indexers contain more slice/integer entries than the tensor
  has axes ([`Ellipsis`] can only cover missing axes, not extra ones).
- Panics if a stepped slice has `step == 0`.
- Panics if more than one [`Ellipsis`] is present.
- Panics if a `Some(_)` value is used where `None` (new axis) is expected.
- Range bounds are never out of bounds: they are clamped as in Python,
  yielding empty axes rather than panics.

For a fallible version, use [`slice_f`].

# See also

## Similar function from other crates/libraries

- NumPy: [basic indexing](https://numpy.org/doc/stable/user/basics.indexing.html#basic-slicing-and-indexing)
- rust `ndarray`: [`ArrayBase::slice`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.slice)
  (ndarray's `s![]` macro is the namesake of rstsr's [`s!`])

## Related functions in RSTSR

- [`diagonal`]: diagonal views.
- [`index_select`] / [`take`] / [`bool_select`]: advanced indexing by integer
  arrays or boolean masks (these copy).
- Operator `[]` ([`Index`]) / `[]=` ([`IndexMut`]): scalar element access, e.g. `a[[1, 2]]`.
- [`to_contig`]: materialize a strided view as a contiguous tensor.
- [`expand_dims`] / [`moveaxis`] / [`squeeze`]: axis bookkeeping without slicing.
- [`flip`]: reverse axes without [`slice!`] gymnastics.

## Variants of this function

- [`slice_f`]: fallible version.
- [`slice_mut`] / [`slice_mut_f`]: mutable views.
- [`into_slice`] / [`into_slice_f`]: ownership-consuming forms.
- [`TensorAny::i`] / [`TensorAny::i_f`] / [`TensorAny::i_mut`] / [`TensorAny::i_mut_f`]:
  short aliases of [`slice`](slice()) / [`slice_f`] /
  [`slice_mut`](slice_mut()) / [`slice_mut_f`].
- Associated methods on [`TensorAny`]: [`TensorAny::slice`] / [`TensorAny::slice_f`] /
  [`TensorAny::slice_mut`] / [`TensorAny::slice_mut_f`] / [`TensorAny::into_slice`] /
  [`TensorAny::into_slice_f`].
