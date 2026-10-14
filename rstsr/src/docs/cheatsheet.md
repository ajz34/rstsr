# RSTSR ↔ NumPy cheat sheet

A side-by-side reference for NumPy users: each table maps a NumPy call to its
RSTSR equivalent. This page covers **core tensor operations**; the faer-backed
linear algebra family lives on the [`linalg`](crate::cheatsheet::linalg) page.

The page assumes:

```rust,ignore
use rstsr::prelude::*;
```

- Most functions and types live in the `rt` module; the prelude also imports the
  traits needed to call them.
- **Row-major throughout.** Every example assumes the device default order is
  [`RowMajor`]. Under [`ColMajor`] several operations (creation, reshape, the
  `pack_*`/`unpack_*` helpers) behave differently; see [`order_semantics`](rstsr_core::order_semantics).
- **Tuple overloading.** Functions with more than one argument take the extra
  arguments as a single tuple: `rt::func(x, y)` and `rt::func((x, y))` are
  different signatures (two parameters vs one tuple parameter). The group types
  (`ReshapeArgs`, `RepeatArgs`, `AxesIndex`, ...) supply the accepted forms.
- **Fallible twins.** A function `rt::foo` panics on error; `rt::foo_f` returns
  `rt::Result` instead. The same holds for methods (`a.foo` / `a.foo_f`).

## Common non-computational operations

### Printing

| NumPy | RSTSR | Description |
|--|--|--|
| `print(a)` | `println!("{a}")` | Print a tensor (large axes are elided with `...`) |
| `print(a.shape, a.strides)` | `println!("{a:?}")` | Debug print: layout, device and concrete type |
| formatted print | `println!("{a:8.3}")` | Per-element width and precision |

### Layout information

| NumPy | RSTSR | Description |
|--|--|--|
| `a.shape` | `a.shape()` | Shape |
| `a.strides` | `a.stride()` | Stride per axis (counts **elements**, not bytes) |
| `a.ndim` | `a.ndim()` | Number of dimensions |
| `a.flags.c_contiguous` | `a.c_contig()` | Row-major contiguous |
| `a.flags.f_contiguous` | `a.f_contig()` | Column-major contiguous |
| `a.size` | `a.size()` | Total number of elements |
| `a.dtype` | `a.dtype()` | Element data type |
| `a.ctypes.data` | `a.raw().as_ptr().add(a.offset())` | Pointer to the first element |

## Tensor creation

### Device parameter

RSTSR functions take a device. The default is `DeviceFaer` (with the
`faer_as_default` feature):

```rust,ignore
let device = DeviceFaer::default();
let a = rt::zeros(([2, 3], &device));
```

### From Python/Rust lists

| NumPy | RSTSR | Description |
|--|--|--|
| `np.array(l)` | `rt::asarray((l, &device))` | Owned tensor (takes ownership of `l`) |
| `np.array(l)` (no copy) | `rt::asarray((&l, &device))` | `TensorView` borrowing `l` |
| `np.array(l)` (mutable) | `rt::asarray((&mut l, &device))` | `TensorMut` borrowing `l` |
| `np.array(l).reshape(2, 3)` | `rt::asarray((l, [2, 3], &device))` | Create with an explicit shape |
| nested lists | `rt::tensor_from_nested!([[1, 2], [3, 4]], &device)` | Tensor from nested literals |

### Special tensors

| NumPy | RSTSR | Description |
|--|--|--|
| `np.zeros((2, 3))` | `rt::zeros(([2, 3], &device))` | Zeros |
| `np.ones((2, 3))` | `rt::ones(([2, 3], &device))` | Ones |
| `np.full((2, 3), 7)` | `rt::full(([2, 3], 7, &device))` | Constant fill |
| `np.eye(3)` | `rt::eye((3, &device))` | Identity |
| `np.arange(6.0)` | `rt::arange((6.0, &device))` | Evenly spaced (step 1, left-inclusive) |
| `np.linspace(0, 1, 5)` | `rt::linspace((0.0, 1.0, 5, &device))` | Evenly spaced, endpoint included |
| `np.zeros((2, 3), order="F")` | `rt::zeros(([2, 3].f(), &device))` | Column-major zeros (`.c()` for row-major) |

### From existing tensors

| NumPy | RSTSR | Description |
|--|--|--|
| `np.zeros_like(a)` | `a.zeros_like()` | Zeros matching `a`'s shape and device |
| `np.ones_like(a)` | `a.ones_like()` | Ones matching `a` |
| `np.tril(a)` | `a.tril()` | Lower triangle (rest zeroed) |
| `np.triu(a)` | `a.triu()` | Upper triangle (rest zeroed) |
| `np.diag(a)` | `a.diag()` | Diagonal of a matrix, or diagonal matrix from a vector |

## Indexing

### Element indexing

Indexing a `Tensor<T, B, D>` (or variant) yields a `T`, `&T` or `&mut T`.

| NumPy | RSTSR | Description |
|--|--|--|
| `a[1, 4]` | `a[[1, 4]]` | Value at a multi-index |
| `a[1, 4] = 3` | `a[[1, 4]] = 3` | Assign one element |

### Basic indexing (views)

Basic indexing never copies data; it returns a view.

| NumPy | RSTSR | Description |
|--|--|--|
| `a[2]` | `a.i(2)` | Index one axis |
| `a[:, 3]` | `a.i((.., 3))` | Index a later axis |
| `a[2:6]` | `a.i(2..6)` | Slice one axis |
| `a[0:2, 2:4]` | `a.i((0..2, 2..4))` | Slice several axes |
| `a[..., -1]` | `a.i((Ellipsis, -1))` | Ellipsis plus an integer |
| `a[:, None]` | `a.i((.., None))` | Insert a new axis |
| `a[2:10:2]` | `a.i(slice!(2, 10, 2))` | Stepped slice |
| `a[::-1]` | `a.i(slice!(None, None, -1))` | Reversed axis |
| `a[:, 3] = b` | `a.i_mut((.., 3)).assign(&b)` | Write through a mutable view |
| `a[2:] += b` | `a.i_mut(2..).add_assign(&b)` <br> `*&mut a.i_mut(2..) += &b` | In-place update through a view |

Free functions [`slice()`] and [`slice_mut()`] are the same as the `.i` / `.i_mut`
methods.

### Diagonal

| NumPy | RSTSR | Description |
|--|--|--|
| `a.diagonal()` | `a.diagonal(0)` | Main diagonal of axes `(0, 1)` |
| `a.diagonal(2)` | `a.diagonal(2)` | Offset diagonal |
| `a.diagonal(-4, -2, -1)` | `a.diagonal((-4, -2, -1))` | Offset with explicit axes |
| `np.fill_diagonal(a, d)` | `a.diagonal_mut(0).assign(&d)` | Write the diagonal |

### Array indexing

Array (fancy) indexing, done with integer arrays or boolean masks, returns a
**copy**, unlike the basic slicing above, which returns a view. In RSTSR the
index arrays travel through the method `a.array_index(...)` rather than the
`a[...]` syntax (which is used for multi-index element access); several rows
also have a dedicated method.

Integer-array indexing:

| NumPy | RSTSR | Description |
|--|--|--|
| `a[[0, 2, 4]]` | `a.array_index([0, 2, 4])` | Gather entries of the first axis |
| `a[[0, 2, 4], [0, 1, 2]]` | `a.array_index(([0, 2, 4], [0, 1, 2]))` | One index array per axis (broadcast together) |
| `a[[0, 2, 4], 1]` | `a.array_index(([0, 2, 4], 1))` | An index array combined with a scalar |
| `a[:, [1, 8, 7]]` | `a.array_index((.., [1, 8, 7]))` <br> `a.index_select(1, [1, 8, 7])` | Gather along one axis |
| `np.take_along_axis(a, idx, axis)` | `a.take_along_axis(&idx, -1)` | Gather per-position indices |

Boolean-array indexing:

| NumPy | RSTSR | Description |
|--|--|--|
| `a[mask]` | `a.mask_select(&mask)` <br> `a.array_index(&mask)` | Select the `true` entries (flattened) |
| `a[mask, :]` (prefix mask) | `a.mask_select(&mask)` <br> `a.array_index(&mask)` | A prefix mask keeps the trailing axes |
| `a[:, mask]` | `a.bool_select(1, mask)` <br> `a.array_index((.., &mask))` | Boolean selection along a given axis |

## Manipulation

### Reshape

`reshape` / `to_*` / `change_*` return a `TensorCow` (a view **or** an owned
copy, chosen automatically); `into_*` return an owned `Tensor`.

| NumPy | RSTSR | Description |
|--|--|--|
| `a.reshape(3, 4)` | `a.reshape((3, 4))` | `TensorCow`; copies only if needed |
| `a.reshape(3, 4).copy()` | `a.into_shape((3, 4))` | Owned `Tensor`; consumes `a` |
| `a.reshape(-1)` | `a.reshape(-1)` | Flatten to 1-D |
| `a.reshape((3, 4), order="F")` | `a.to_layout([3, 4].f())` | Reshape to an f-contiguous layout |

### Layout-only operations

These change layout only; `to_*`/no-prefix borrow and return a view, `into_*`
consume the tensor.

| NumPy | RSTSR | Description |
|--|--|--|
| `a.T` | `a.t()` | Reverse all axes |
| `a.transpose(1, 0, 2)` | `a.transpose((1, 0, 2))` | Permute axes |
| `a.swapaxes(1, 2)` | `a.swapaxes(1, 2)` | Swap two axes |
| `np.moveaxis(a, 0, 2)` | `a.moveaxis(0, 2)` | Move one axis to a new position |
| `a.squeeze(2)` | `a.squeeze(2)` | Drop a length-1 axis |
| `np.expand_dims(a, 2)` | `a.expand_dims(2)` | Insert a length-1 axis |
| `np.flip(a, axis=2)` | `a.flip(2)` | Reverse one axis |
| `np.broadcast_to(a, (2, 3))` | `a.to_broadcast([2, 3])` | Broadcast (no copy) |
| `np.broadcast_arrays(a, b)` | `rt::broadcast_arrays([&a, &b])` | Broadcast several tensors together |

### Joining and repeating

| NumPy | RSTSR | Description |
|--|--|--|
| `np.concatenate([a, b])` | `rt::concat([&a, &b])` | Join along axis 0 |
| `np.concatenate([a, b], axis=1)` | `rt::concat(([&a, &b], 1))` | Join along a given axis |
| `np.stack([a, b])` | `rt::stack([&a, &b])` | Stack along a new axis 0 |
| `np.stack([a, b], axis=1)` | `rt::stack(([&a, &b], 1))` | Stack along a new axis |
| `np.tile(a, (2, 3))` | `a.tile([2, 3])` | Repeat a tensor |
| `np.repeat(a, 2)` | `a.repeat((2, None))` | Repeat elements |
| `np.roll(a, 2)` | `a.roll((2, None))` | Cyclic shift |
| `np.trace(a)` | `a.trace(())` | Sum of the main diagonal |

## Operations

### Operators

Rust's arithmetic operators work elementwise with broadcasting: `+`, `-`, `*`,
`/`, the bitwise `&` / `|` / `^`, unary `-`, and `!`. In-place forms (`+=`, ...)
reuse the left operand's storage when possible.

**`%` is matrix multiplication**, not remainder: `a % b` is `a.matmul(&b)`.
For the remainder, use the free function `rt::rem(&a, &b)`: the method `.rem`
is the `%` operator itself, so it is **not** the elementwise remainder.

| NumPy | RSTSR | Description |
|--|--|--|
| `a + b` | `&a + &b` / `a + &b` / `a + b` | Elementwise add (owned operands may be reused) |
| `a + 1` | `&a + 1` | Tensor-scalar arithmetic |
| `np.add(a, b, out=c)` | `rt::add_with_output(&a, &b, &mut c)` | Write into a given output (free function only) |
| `a @ b` | `&a % &b` / `a.matmul(&b)` | Matrix product (broadcasting like `np.matmul`) |
| `a % b` | `rt::rem(&a, &b)` | Elementwise remainder (free function; `.rem` is `%`) |

### Elementwise functions

RSTSR implements common unary/binary functions directly as tensor methods:
`a.sin()`, `a.cos()`, `a.exp()`, `a.log()`, `a.sqrt()`, `a.abs()`, `a.sign()`,
`a.floor()`, `a.round()`, and so on, plus `a.pow(..)`, `a.maximum(..)`,
`a.minimum(..)`, `a.clip(..)`, `a.hypot(..)`.

Comparison operators are methods too: `a.equal(..)` / `a.not_equal(..)`,
`a.less(..)` / `a.less_equal(..)`, `a.greater(..)` / `a.greater_equal(..)`
(free-function aliases `rt::eq` / `rt::ne` / `rt::lt` / `rt::le` / `rt::gt` /
`rt::ge` also exist).

| NumPy | RSTSR | Description |
|--|--|--|
| `np.sin(a)` | `a.sin()` | Sine (also `cos`, `tan`, ...) |
| `np.power(a, b)` | `a.pow(&b)` | Elementwise power |
| `np.clip(a, lo, hi)` | `a.clip((lo, hi))` | Clip to an interval |
| `a > 2` | `a.greater(2)` | Elementwise comparison (boolean output) |
| `np.where(c, a, b)` | `c.r#where(&a, &b)` | Choose by a condition |
| custom elementwise | `a.mapv(\|x\| x * x + 1.0)` | Apply a Rust closure elementwise |

### Reductions

`a.sum()` and friends reduce **all** elements; the `_axes` suffix reduces along
given axes; the `_with_args` suffix takes a `ReduceArgs` group `(axes, keepdims)`.

| NumPy | RSTSR | Description |
|--|--|--|
| `a.sum()` | `a.sum()` | Sum of all elements |
| `a.sum(axis=-1)` | `a.sum_axes(-1)` | Sum along the last axis |
| `a.sum(axis=(2, -1))` | `a.sum_axes((2, -1))` | Sum along several axes |
| `a.sum(axis=-1, keepdims=True)` | `a.sum_with_args((-1, true))` | Reduce keeping the axes (`axes`, `keepdims`) |
| `a.mean()` | `a.mean()` | Mean (also `min`, `max`, `prod`, `var`, `std`, `l2_norm`) |
| `np.argmax(a)` | `a.argmax()` | Index of the maximum |
| `np.all(a)` / `np.any(a)` | `a.all()` / `a.any()` | Boolean reductions |
| `np.count_nonzero(a)` | `a.count_nonzero()` | Number of non-zero elements |
| `np.cumsum(a)` | `a.cumulative_sum(None)` | Cumulative sum |

### Sorting, searching and set operations

| NumPy | RSTSR | Description |
|--|--|--|
| `np.sort(a)` | `a.sort(None)` | Sorted copy |
| `np.argsort(a)` | `a.argsort(None)` | Indices that would sort `a` |
| `np.searchsorted(a, v)` | `a.searchsorted(&v, None)` | Binary-search insertion points |
| `np.nonzero(a)` | `a.nonzero()` | Indices of non-zero elements |
| `np.diff(a)` | `a.diff(-1, 1, None, None)` | Discrete differences |
| `np.unique(a)` | `a.unique_values()` | Unique values |
| `np.isin(a, b)` | `a.isin(&b, false)` | Membership test |

### Comparing tensors

| NumPy | RSTSR | Description |
|--|--|--|
| `np.allclose(a, b)` | `a.allclose(&b, None)` | Element-wise closeness within a tolerance |
| `np.array_equal(a, b)` | `a == b` (on `1`-element views) | Exact equality of contents |

## Where to go next

- [`linalg`](crate::cheatsheet::linalg): the faer-backed `rt::linalg` family
  (`eigh`, `svd`, `solve`, `inv`, ...), with SciPy references.
- [`api_specification`](crate::api_specification): the full identifier list.
- [`array_api_standard`](crate::array_api_standard): Python Array API conformance.
- [`order_semantics`](rstsr_core::order_semantics): row-major vs column-major behavior.
