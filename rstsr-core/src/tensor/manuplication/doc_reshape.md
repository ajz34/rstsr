Reshapes the given tensor to the specified shape.

```rust
# use rstsr::prelude::*;
# let device = DeviceCpuSerial::default();
# let tensor = rt::arange((6, &device));
# let shape = [2, 3];
rt::reshape!(tensor, shape, order = None, copy = None); // -> TensorCow
# let tensor = rt::arange((6, &device));
# let shape = [2, 3];
rt::reshape!(tensor, shape, order = None, copy = None, into); // -> Tensor
# let tensor = rt::arange((6, &device));
# let shape = [2, 3];
rt::reshape!(tensor, shape, order = None, copy = None, change); // -> TensorCow
# let tensor = rt::arange((6, &device));
# let shape = [2, 3];
// fallible also applies to reshape/into_shape/change_shape variants
rt::reshape!(tensor, shape, order = None, copy = None, fallible); // -> Result<TensorCow>
```

We provide the macro [`reshape!`](crate::prelude::rt::reshape!) for full overloads. However, this function is better to be used as associated method `tensor.reshape(shape)` in most cases.

---

<div class="warning">

**Row/Column Major Notice**

This function behaves differently on default orders ([`RowMajor`] and [`ColMajor`]) of device.

</div>

# Function signature

## Parameters

- `tensor`

  **The input tensor to be reshaped.**

  - `reshape`: [`&TensorAny<R, T, B, D>`](TensorAny) (borrowed)
  - `into_shape`: [`TensorAny<R, T, B, D>`](TensorAny) (consumes ownership)
  - `change_shape`: [`TensorAny<R, T, B, D>`](TensorAny) (consumes ownership)

- `shape`: TryInto [`AxesIndex<isize>`]

  **The new shape of the tensor.**

  You can use list `[2, 3]` (or `(2, 3)`, `vec![2, 3]`) to indicate the new shape.
  The size of new shape must be the same as the size of original shape.
  Please note that whatever type you use for shape, the output tensor will always be dynamic-dimensional ([`IxD`]).

  You can substitute one `-1` in shape list to indicate *infer this dimension*.
  For example, for a tensor with size 6, `reshape([2, -1])` will infer the second dimension as 3, and `reshape([-1, 3])` will infer the first dimension as 2.

  Also, you can use integer `reshape(-1)` (or exactly the same size of original tensor) to flatten the tensor into 1-dimension vector.

- `order`: into [`Option<FlagOrder>`](FlagOrder)

  **Read the elements of input tensor in the specified order.**

  *This parameter only works in function [`reshape_with_args`] and macro [`reshape!`](crate::prelude::rt::reshape!).*

  Valid values for `order` are:
  - `None` (default): use the default order of device.
  - `RowMajor`: read the elements in row-major order (C-contiguous).
  - `ColMajor`: read the elements in column-major order (F-contiguous).

  Note this option also affects the preference of how to write the tensor.
  If the new shape is not compatible with the original tensor, then the output tensor will be written in the order user provided.

- `copy`: into [`Option<bool>`](bool)

  **Determine whether to copy the data during reshape.**

  *This parameter only works in function [`reshape_with_args`] and macro [`reshape!`](crate::prelude::rt::reshape!).*

  Valid values for `copy` are:
  - `true`: always copy data, and return an owned tensor with contiguous memory layout (order specified by user or device's default).
  - `false`: never copy data; if new shape is *not compatible* to the original tensor's layout, then panics.
  - `None` (default): copy data if new shape is *not compatible* to the original tensor's layout, but try to avoid copying if *compatible*.

  *Compatibility* means whether input tensor's layout is sufficiently contiguous with the tensor order specified.
  If input tensor is contiguous in the specified order, then reshape will always returns a view and no data cloning will occur.

  For more details on compatibility, please refer to the section of [Occasions of data cloning](#occasions-of-data-cloning) in this document.

## Returns

The reshaped tensor.
  
This function will try to avoid data cloning, reusing owned data or returning a view if possible.

- `reshape`: [`TensorCow<'a, T, B, IxD>`](TensorCow)

  - If shape compatible, a view will be returned.
  - If shape not-compatible, an owned tensor will be returned, cloning the data.
  
  Cow (clone-on-write) semantics is used for representing either view or owned tensor.
  - Use associated function `tensor.is_owned()` to check whether the output tensor is owned or view.
  - Use `tensor.view()` to get a view of the output tensor for future usage.
  - Use `tensor.into_owned()` to get an owned tensor; if the output tensor is already owned, then no cloning will occur; if the output tensor is a view, then cloning will occur.

- `into_shape`: [`Tensor<T, B, IxD>`](Tensor)

  - If shape compatible and input tensor owns data, then the input tensor will be returned without cloning, only changing the layout of tensor.
  - Otherwise, an owned tensor will be returned, cloning the data.

- `change_shape`: [`TensorCow<'a, T, B, IxD>`](TensorCow)

  - If shape compatible and input tensor owns data, then a view will be returned, sharing the same data with input tensor.
  - If shape compatible but input tensor does not own data, then a view will be returned, borrowing the data from input tensor.
  - If shape not-compatible, an owned tensor will be returned, cloning the data.

## Argument overloads

```rust
# use rstsr::prelude::*;
# let device = DeviceCpu::default();
# let tensor = rt::arange((6, &device));
# let shape = [2, 3];
rt::reshape!(tensor, shape, order = None, copy = None);
# let shape = [2, 3];
tensor.reshape(shape);
# let shape = [2, 3];
rt::reshape(&tensor, shape);
# let shape = [2, 3];
# let order = None;
# let copy = None;
tensor.reshape_with_args(shape, (order, copy));
# let shape = [2, 3];
# let order = None;
tensor.reshape_with_args(shape, order);
# let shape = [2, 3];
# let copy = None;
tensor.reshape_with_args(shape, copy);
# let shape = [2, 3];
# let order = None;
# let copy = None;
rt::reshape_with_args(&tensor, shape, (order, copy));
# let shape = [2, 3];
# let order = None;
rt::reshape_with_args(&tensor, shape, order);
# let shape = [2, 3];
# let copy = None;
rt::reshape_with_args(&tensor, shape, copy);
```

## Function variants

| type | to-variant | into-variant | change-variant |
|-|-|-|-|
| macro    | [`reshape!`](crate::prelude::rt::reshape!) | [`reshape!`](crate::prelude::rt::reshape!) | [`reshape!`](crate::prelude::rt::reshape!) |
| fn       | [`reshape`]<br>[`reshape_with_args`] | [`into_shape`]<br>[`into_shape_with_args`] | [`change_shape`]<br>[`change_shape_with_args`] |
| assoc fn | [`TensorAny::reshape`]<br>[`TensorAny::reshape_with_args`] | [`TensorAny::into_shape`]<br>[`TensorAny::into_shape_with_args`] | [`TensorAny::change_shape`]<br>[`TensorAny::change_shape_with_args`] |

For fallible variants, add suffix `_f` to the function name, or add `fallible` to the macro. Fallible versiion will return [`Result`].

# Examples

In row-major order, to reshape a vector of (6, ) to a matrix of (2, 3):
```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((6, &device));
let result = a.reshape([2, 3]);
println!("{result}");
// [[ 0 1 2]
//  [ 3 4 5]]
```

You can also use negative dimension, where -1 means *infer this dimension*:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
// in this case, unspecified axes length is inferred as 6 / 3 = 2
let a = rt::arange((6, &device));
let result = a.reshape([3, -1]);
println!("{result}");
// [[ 0 1]
//  [ 2 3]
//  [ 4 5]]
```

# Ownership Semantics between [`reshape`], [`into_shape`] and [`change_shape`]

[`into_shape`] and [`change_shape`] take ownership of the input tensor. They are important
variants to this function [`reshape`].

| Function | Input Ownership | Output Ownership | Cloning Condition |
|--|--|--|--|
| [`reshape`] | Borrowed <br> [`&TensorAny`](TensorAny) | View <br> [`TensorCow`] with [`DataCow::Ref`] | not cloned (layout-compatible) |
| | | Owned <br> [`TensorCow`] with [`DataCow::Owned`] | cloned (layout-not-compatible) |
| [`into_shape`] | Owned <br> [`Tensor`] | Owned <br> [`Tensor`] | not cloned (layout-compatible, input tensor owns data, input tensor is compact) |
| | | Owned <br> [`Tensor`] | cloned (otherwise) |
| | Otherwise <br> [`TensorAny`] | Owned <br> [`Tensor`] | cloned (always) |
| [`change_shape`] | Owned <br> [`Tensor`] | Owned <br> [`TensorCow`] with [`DataCow::Owned`] | not cloned (layout-compatible, input tensor owns data, input or is compact) |
| | | Owned <br> [`TensorCow`] with [`DataCow::Owned`] | cloned (otherwise) |
| | Otherwise <br> [`TensorAny`] | View <br> [`TensorCow`] with [`DataCow::Ref`] | not cloned (layout-compatible) |
| | | Owned <br> [`TensorCow`] with [`DataCow::Owned`] | cloned (layout-not-compatible) |

# Tips on common compilation errors

You may encounter ownership problem when you try to assign a reshaped tensor like this:

```compile_fail
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((6, &device)).reshape([2, 3]);
println!("a: {:?}", a);
```

The compiler may give an error like:

```text
704 |    let a = rt::arange((6, &device)).reshape([2, 3]);
    |            ^^^^^^^^^^^^^^^^^^^^^^^^                - temporary value is freed at the end of this statement
    |            |
    |            creates a temporary value which is freed while still in use
705 |    println!("a: {:?}", a);
    |                        - borrow later used here
    |
help: consider using a `let` binding to create a longer lived value
    |
704 ~    let binding = rt::arange((6, &device));
705 ~    let a = binding.reshape([2, 3]);
    |
```

The suggestion by compiler is correct. However, you have another simpler way to solve this
problem by using [`into_shape`] variant that takes ownership:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((6, &device)).into_shape([2, 3]);
```

# Notes of API accordance

- Array-API: `reshape(x, /, shape, *, copy=None)` ([`reshape`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.reshape.html))
- NumPy: `reshape(a, /, shape, order='C', *, copy=None)` ([`numpy.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)):
- RSTSR: `rt::reshape!(tensor, shape, order = None, copy = None)`
- RSTSR: `rt::reshape_with_args(tensor, shape, (order, copy))`
- RSTSR: `rt::reshape(tensor, shape)`

Please note that the `order` argument in RSTSR does not support NumPy's `'A'` (order='A' means 'F' if the array is Fortran contiguous, 'C' otherwise in NumPy).

# Elaborated examples

## Difference between [RowMajor] and [ColMajor]

Tensor can be uniquely iterated (into a 1-dimension vector), for either row-major or
column-major order.

**Reshape operation does not change the iterated sequence of a tensor**, by definition. In other
words, the following code always holds true:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(ColMajor);
let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
# let b = a.reshape([3, 2]);
// note iteration order of associated method `iter` depends on `device.default_order()`

// let b = a.reshape(... SOME SHAPE ...);
let a_vec = a.iter().collect::<Vec<_>>();
let b_vec = b.iter().collect::<Vec<_>>();
assert_eq!(a_vec, b_vec); // iterated sequence is the same
```

For example, in row-major order, reshape a matrix of (2, 3) to (3, 2):

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
// set to row-major order
device.set_default_order(RowMajor);
// a: [[0, 1, 2], [3, 4, 5]]
// b: [[0, 1], [2, 3], [4, 5]]
// iterated sequence: [0, 1, 2, 3, 4, 5]

let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
println!("{a}");
// [[ 0 1 2]
//  [ 3 4 5]]
let b = a.reshape([3, 2]);
println!("{b}");
// [[ 0 1]
//  [ 2 3]
//  [ 4 5]]

let a_vec = a.iter().cloned().collect::<Vec<_>>();
println!("{a_vec:?}");
// [0, 1, 2, 3, 4, 5]
let b_vec = b.iter().cloned().collect::<Vec<_>>();
println!("{b_vec:?}");
// [0, 1, 2, 3, 4, 5]
```

In the column-major order, reshape the same matrix of (2, 3) to (3, 2) will yield a different
result:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
// set to column-major order
device.set_default_order(ColMajor);
// a: [[0, 1, 2], [3, 4, 5]]
// b: [[0, 4], [3, 2], [1, 5]]
// iterated sequence: [0, 3, 1, 4, 2, 5]

let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
println!("{a}");
// [[ 0 1 2]
//  [ 3 4 5]]
let b = a.reshape([3, 2]);
println!("{b}");
// [[ 0 4]
//  [ 3 2]
//  [ 1 5]]

let a_vec = a.iter().cloned().collect::<Vec<_>>();
println!("{a_vec:?}");
// [0, 3, 1, 4, 2, 5]
let b_vec = b.iter().cloned().collect::<Vec<_>>();
println!("{b_vec:?}");
// [0, 3, 1, 4, 2, 5]
```

You can also use function [`reshape_with_args`]`(shape, order)` to specify the order for reading
the tensor.

## Occasions of data cloning

The following discussion assumes the tensor is in row-major order. Similar discussion applies to
column-major order.

If the tensor to be reshaped is already in C-contiguous if the device is also row-major, or
F-contiguous if the device is column-major, then the reshape operation can be performed without
any data cloning.

Otherwise, whether data cloning is necessary depends. For example, consider a tensor of shape
(4, 6, 9) but with non-contiguous strides:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
// contiguous situation: (4, [6, 9]), or say the last two dimensions are contiguous
let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
println!("{:?}", a.layout());
// 3-Dim (dyn), contiguous: c
// shape: [4, 6, 9], stride: [72, 9, 1], offset: 0
```

Those cases will not require data cloning (returns a view, or [`DataCow::Ref`] internally):

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
// split a single dimension into multiple dimensions
assert!(!a.reshape([2, 2, 6, 9]).is_owned()); // (4, 6, 9) -> ([2, 2], 6, 9)
assert!(!a.reshape([4, 3, 2, 9]).is_owned()); // (4, 6, 9) -> (4, [3, 2], 9)
assert!(!a.reshape([4, 2, 3, 3, 3]).is_owned()); // (4, 6, 9) -> (4, [2, 3], [3, 3])

// merge contiguous dimensions into a single dimension
assert!(!a.reshape([4, 54]).is_owned()); // (4, 6, 9) -> (4, 6 * 9)

// merge contiguous dimensions and then split
assert!(!a.reshape([4, 3, 6, 3]).is_owned()); // (4, [6, 9]) -> (4, [3, 6, 3])
```

However, the following cases will require data cloning (returns an owned tensor, or
[`DataCow::Owned`] internally):

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
assert!(a.reshape([24, 9]).is_owned()); // (4, 6, 9) -> (4 * 6, 9)
assert!(a.reshape(-1).is_owned()); // (4, 6, 9) -> (4 * 6 * 9)
assert!(a.reshape([12, 2, 9]).is_owned()); // (4, 6, 9) -> (4 * [3, 2], 9)
```

Please note that default order of device (row-major or column-major) matters. For the same
tensor slicing, if the device is column major, then behavior of merging contiguous dimensions
can be different:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(ColMajor);
// contiguous situation: ([4, 6], 9), or say the first two dimensions are contiguous
// this is different to  (4, [6, 9]) in row major case
let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
println!("{:?}", a.layout());
// 3-Dim (dyn), contiguous: f
// shape: [4, 6, 9], stride: [1, 4, 32], offset: 0

// merge dimensions into a single dimension, col-major will be different to row-major case
assert!(a.reshape([4, 54]).is_owned()); // (4, 6, 9) -> (4, 6 * 9)
assert!(!a.reshape([24, 9]).is_owned()); // ([4, 6], 9) -> (4 * 6, 9)
```

## Reshape with specified order and copy

You can specify the order for reading the tensor by argument `order`.

Following is an example of row-major reshape. This is independent to the original default-layout
of device.

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
println!("{a}");
// [[ 0 1 2]
//  [ 3 4 5]]
let a_row = rt::tensor_from_nested!([[0, 1], [2, 3], [4, 5]], &device);
println!("{a_row}");
// [[ 0 1]
//  [ 2 3]
//  [ 4 5]]
```

And here is an example of col-major reshape.

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::tensor_from_nested!([[0, 1, 2], [3, 4, 5]], &device);
println!("{a}");
// [[ 0 1 2]
//  [ 3 4 5]]
let a_col = rt::tensor_from_nested!([[0, 4], [3, 2], [1, 5]], &device);
println!("{a_col}");
// [[ 0 4]
//  [ 3 2]
//  [ 1 5]]
```

The following example shows that if `copy = false`, then an error will be raised when the new
shape is not compatible with the original shape. Given a strided tensor:

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
// shape: (4, 6, 9), stride: (72, 9, 1), not c-contiguous
// contiguous situation: (4, [6, 9]), or say the last two dimensions are contiguous
let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
assert_eq!(a.shape(), &[4, 6, 9]);
assert_eq!(a.stride(), &[72, 9, 1]);
assert!(!a.c_contig());
```

The following example shows the reshaping does not explicitly clones data, and `copy = false`
does not raise error.

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
// split a single dimension into multiple dimensions
assert!(a.reshape_with_args_f([2, 2, 6, 9], false).is_ok()); // (4, 6, 9) -> ([2, 2], 6, 9)
assert!(a.reshape_with_args_f([4, 3, 2, 9], false).is_ok()); // (4, 6, 9) -> (4, [3, 2], 9)
assert!(a.reshape_with_args_f([4, 2, 3, 3, 3], false).is_ok()); // (4, 6, 9) -> (4, [2, 3], [3, 3])

// merge contiguous dimensions into a single dimension
assert!(a.reshape_with_args_f([4, 54], false).is_ok()); // (4, 6, 9) -> (4, 6 * 9)

// merge contiguous dimensions and then split
assert!(a.reshape_with_args_f([4, 3, 6, 3], false).is_ok()); // (4, [6, 9]) -> (4, [3, 6, 3])
```

However, the following example will raise error due to shape-incompatible. Using `copy = None`
or `copy = true` will work, but the data will be cloned.

```rust
# use rstsr::prelude::*;
# let mut device = DeviceCpu::default();
# device.set_default_order(RowMajor);
let a = rt::arange((288, &device)).into_shape([4, 8, 9]).into_slice((.., 0..6, ..));
// merge non-contiguous dimensions
assert!(a.reshape_with_args_f([24, 9], false).is_err()); // (4, 6, 9) -> (4 * 6, 9)
assert!(a.reshape_with_args_f([-1], false).is_err()); // (4, 6, 9) -> (4 * 6 * 9)
assert!(a.reshape_with_args_f([12, 2, 9], false).is_err()); // (4, 6, 9) -> (4 * [3, 2], 9)
```

# See also

## Similar function from other crates/libraries

- Python Array API standard: [`reshape`](https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.reshape.html)
- NumPy: [`reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)
- ndarray: [`to_shape`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.to_shape)

## Related functions in RSTSR

- [`reshapeable_without_copy`]: Check whether the layout is compatible with the new shape.
- [`to_layout`]: Return a tensor with the specified layout.
- [`to_contig`]: Return an owned contiguous tensor.
