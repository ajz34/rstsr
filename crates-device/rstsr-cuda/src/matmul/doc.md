Matrix multiplication on [`DeviceCuda`](crate::device::DeviceCuda): `c = alpha * (a @ b) + beta * c`
for the tensor-level entries of `rstsr-core` ([`rt::matmul`], [`rt::matmul_from`],
[`rt::matmul_with_output`], their methods on the tensors and the `%` operator), plus the entries
that name a CUDA compute mode per call ([`TensorCudaMatMulAPI`]).

Every product runs through a **chain of drivers**: cuBLASLt, `cublasGemmEx`, the classic cuBLAS
entries, a tiled SIMT kernel and a naive kernel, tried in that order until one takes the problem.
One call therefore serves `f32`/`f64`/complex/half products on the tensor cores, integers and
mixed dtypes on the kernels, and whatever layout cuBLAS cannot express on the kernels that can.
Which routine ran is one flag away ([`CudaRuntimeFlags::trace_matmul`]), and which routines are
allowed is another ([`CudaRuntimeFlags::matmul_drivers`]).

The page opens with the entries most programs need, then describes the chain and the type triples
it serves, how to choose the routine and the compute mode, how to overlap uploads with products on
streams, the layouts, the resources the drivers keep, and the numerics.

# Quick start

The entry points are the ones every rstsr device implements, so a program moves between the CPU
devices and `DeviceCuda` by changing the device its tensors live on. Two things differ, both
consequences of the device being asynchronous: a call returns as soon as the product is enqueued,
and the host waits only where it reads a result back (`Display`, `to_scalar`, `to_device` to a CPU
device, [`synchronize`](crate::sync::CudaSyncAPI::synchronize)). The examples below skip on a
machine without a CUDA device, on which [`DeviceCuda::new`](crate::device::DeviceCuda::new)
panics; the probe is wrapped in `catch_unwind` because `is_available` panics too when `libcuda`
itself is missing.

## A plain product

```rust
use rstsr::prelude::*;
use rstsr::cuda::prelude::*;

// skip without a CUDA device (`is_available` panics when `libcuda` is missing, hence the catch)
if !std::panic::catch_unwind(cuda::is_available).unwrap_or(false) {
    return;
}
let device = DeviceCuda::new(0);
let a = rt::tensor_from_nested!([[1.0, 2.0], [3.0, 4.0]], &device);
let b = rt::tensor_from_nested!([[1.0, 0.0], [1.0, 1.0]], &device);
let c = rt::matmul(&a, &b); // also `a.matmul(&b)` and `&a % &b`
let cpu = DeviceCpuSerial::default();
println!("{}", c.to_device(&cpu)); // the copy back is where the host waits
// [[ 3 2]
//  [ 7 4]]
# assert_eq!(format!("{}", c.to_device(&cpu)), "[[ 3 2]\n [ 7 4]]");
```

The shape rules are those of [`rt::matmul`] on every device: the last two axes of each operand are
the matrix axes (the first two under column-major, see [Layouts and the default order](#layouts-and-the-default-order)),
leading axes broadcast, and a 1-D operand is promoted to a matrix and dropped again, so `[k] @ [k]`
is the inner product (a rank-0 tensor). The result is a fresh contiguous tensor on the device, of
the dtype `Mul` gives the operands (`f16 × f16 → f16`; an `f32` output of `f16` operands is named
explicitly, see [the compute mode of one call](#the-compute-mode-of-one-call-matmul_from_mode)).

## A GEMM into your own output

[`rt::matmul_from`] writes `c = alpha * (a @ b) + beta * c` into a tensor you hold, which is how a
loop avoids allocating an output per product; with `beta = 0` the previous contents of `c` are
never read (the BLAS convention, so `NaN`s in `c` do not leak), which is also what
[`rt::matmul_with_output`] does with `alpha = 1`. The output may be a strided view, as long as no
element is written twice:

```rust
# use rstsr::prelude::*;
# use rstsr::cuda::prelude::*;
# if !std::panic::catch_unwind(cuda::is_available).unwrap_or(false) { return; }
# let device = DeviceCuda::new(0);
# let cpu = DeviceCpuSerial::default();
# let a = rt::tensor_from_nested!([[1.0, 2.0], [3.0, 4.0]], &device);
# let b = rt::tensor_from_nested!([[1.0, 0.0], [1.0, 1.0]], &device);
let mut c: Tensor<f64, DeviceCuda> = rt::full(([2, 2], 1.0, &device));
c.matmul_from(&a, &b, 2.0, 3.0); // c = 2 * (a @ b) + 3 * c
println!("{}", c.to_device(&cpu));
// [[ 9 7]
//  [ 17 11]]
# assert_eq!(format!("{}", c.to_device(&cpu)), "[[ 9 7]\n [ 17 11]]");

let mut big: Tensor<f64, DeviceCuda> = rt::zeros(([4, 4], &device));
{
    let mut block = big.i_mut(s![1..3, 1..3]); // a strided view into a larger matrix
    a.matmul_with_output(&b, &mut block); // block = a @ b, nothing else of `big` is touched
}
println!("{}", big.to_device(&cpu));
// [[ 0 0 0 0]
//  [ 0 3 2 0]
//  [ 0 7 4 0]
//  [ 0 0 0 0]]
# assert_eq!(format!("{}", big.to_device(&cpu)), "[[ 0 0 0 0]\n [ 0 3 2 0]\n [ 0 7 4 0]\n [ 0 0 0 0]]");
```

## The compute mode of one call: `matmul_from_mode`

The cuBLAS drivers take a *compute mode*: plain `f32` arithmetic, or the tensor-core paths that
down-convert the inputs (TF32, FP16, BF16), or a pedantic mode that forbids every such shortcut.
[`rt::matmul`] reads it from the device flags (see [Compute modes](#compute-modes)); the entries of
[`TensorCudaMatMulAPI`] pass it per call instead. The mode is keyed by the **output** type, so the
enum is `ComputeType32F` for an `f32` output, `ComputeType64F` for `f64`, and so on, and the
compiler checks the pairing:

```rust
use rstsr::prelude::*;
use rstsr::cuda::prelude::*; // `ComputeType32F` and `TensorCudaMatMulAPI` are here

# if !std::panic::catch_unwind(cuda::is_available).unwrap_or(false) { return; }
# let device = DeviceCuda::new(0);
# let cpu = DeviceCpuSerial::default();
// this `f32` product on TF32 tensor cores, whatever the device flags say
let a: Tensor<f32, DeviceCuda> = rt::tensor_from_nested!([[1.0f32, 2.0], [3.0, 4.0]], &device);
let b: Tensor<f32, DeviceCuda> = rt::tensor_from_nested!([[1.0f32, 0.0], [1.0, 1.0]], &device);
let mut c: Tensor<f32, DeviceCuda> = rt::zeros(([2, 2], &device));
c.matmul_from_mode(&a, &b, 1.0, 0.0, ComputeType32F::FastTF32);

// `f16` operands into an `f32` output: `rt::matmul` infers an `f16` output from `Mul`, so the
// output type is named at `out`, and the mode is that of the `f32` output
use half::f16;
let a16 = rt::tensor_from_nested!([[f16::from_f32(1.0), f16::from_f32(2.0)], [f16::from_f32(3.0), f16::from_f32(4.0)]], &device);
let b16 = rt::tensor_from_nested!([[f16::from_f32(1.0), f16::from_f32(0.0)], [f16::from_f32(1.0), f16::from_f32(1.0)]], &device);
let c: Tensor<f32, DeviceCuda> = a16.matmul_with_mode(&b16, ComputeType32F::Pedantic).out::<f32>();
println!("{}", c.to_device(&cpu));
// [[ 3 2]
//  [ 7 4]]
# assert_eq!(format!("{}", c.to_device(&cpu)), "[[ 3 2]\n [ 7 4]]");
```

Every entry of the trait has a fallible `_f` twin; the infallible ones panic with the same error
on a shape that does not follow the rules, on mismatched devices, or when no enabled driver can
serve the product.

## Overlapping uploads with products

A pageable `to_device` blocks the host until the copy is done, so uploads and products
issued from one thread alternate. Page-locked host memory ([`DeviceCudaPinned`](crate::pinned::DeviceCudaPinned))
uploads without waiting, and a second stream lets the upload of the next chunk run under the
product of the last one:

```rust
# use rstsr::prelude::*;
# use rstsr::cuda::prelude::*;
# if !std::panic::catch_unwind(cuda::is_available).unwrap_or(false) { return; }
let base = DeviceCuda::new(0);
let dev_h = base.new_stream(); // uploads
let dev_c = base.new_stream(); // products
let pinned = DeviceCudaPinned::new(0);

// page-locked host chunks, filled before the loop
let chunks: Vec<Tensor<f64, DeviceCudaPinned>> = (0..4)
    .map(|i| pinned.tensor(pinned.from_vec(vec![i as f64; 1 << 20]), [1024, 1024]))
    .collect();

let mut products = Vec::new();
let mut uploads = Vec::new();
for x in &chunks {
    let a = x.to_device(&dev_h); // H2D enqueued on `dev_h`; the host runs on
    let b = a.to_device(&dev_c); // the same buffer, re-tagged onto `dev_c` (no copy)
    products.push(rt::matmul(&b, &b)); // enqueued on `dev_c`, after this upload only
    uploads.push(a); // keep it alive: a free would order `dev_h` behind the product
}
dev_c.synchronize(); // the host waits here, once
# let cpu = DeviceCpuSerial::default();
# let p = products[3].to_device(&cpu);
# assert_eq!(p[&[0usize, 0][..]], 9216.0); // 1024 terms of 3.0 * 3.0
```

On an RTX 5070 Laptop this form runs 4 chunks of `[4096, 4096]` `f32` in 47 ms where the pageable
flow takes 142 ms; the rules behind it, and what breaks the overlap, are in
[Overlapping uploads with products on streams](#overlapping-uploads-with-products-on-streams).

# The entries most programs use

| entry | computes | output | fallible twin |
|---|---|---|---|
| [`rt::matmul`]`(a, b)`, `a.matmul(b)`, `a % b` | `a @ b` | fresh tensor, dtype by `Mul` | [`rt::matmul_f`], `a.matmul_f(b)` |
| [`rt::matmul_with_output`]`(a, b, c)`, `a.matmul_with_output(b, c)` | `c = a @ b` (`c` not read) | `c`, any mutable view | [`rt::matmul_with_output_f`] |
| [`rt::matmul_from`]`(c, a, b, alpha, beta)`, `c.matmul_from(a, b, alpha, beta)` | `c = alpha * (a @ b) + beta * c` | `c`, any mutable view | [`rt::matmul_from_f`] |
| [`TensorCudaMatMulAPI::matmul_from_mode`]`(c, a, b, alpha, beta, mode)` | the GEMM under `mode` | `c` | [`matmul_from_mode_f`](TensorCudaMatMulAPI::matmul_from_mode_f) |
| [`TensorCudaMatMulAPI::matmul_with_output_mode`]`(a, b, c, mode)` | `c = a @ b` under `mode` | `c` | [`matmul_with_output_mode_f`](TensorCudaMatMulAPI::matmul_with_output_mode_f) |
| [`TensorCudaMatMulAPI::matmul_with_mode`]`(a, b, mode)`.[`out::<TC>()`](WithMode::out) | `alpha * (a @ b)` into a fresh `TC` output | `Tensor<TC, DeviceCuda, IxD>` | [`out_f`](WithMode::out_f) |
| [`matmul_with_mode_f`]`(a, b, alpha, mode)` | `alpha * (a @ b)` into a fresh output of the mode's type | `Result<Tensor<TC, DeviceCuda, DC>>` | — (this is the fallible one) |

[`rt::matmul_from`] and [`rt::matmul_with_output`] carry no `Mul` bound: the operands may have
other dtypes than the output (`f32 @ f64 -> f64`, `f16 @ f16 -> f32`), and the naive kernel casts
each operand to the output type. [`rt::matmul`] and `%` infer the output from `Mul`, so a product
of `f16` operands gives an `f16` output; name an `f32` one with `matmul_with_output`, with
`matmul_from_mode`, or with [`WithMode::out`].

Related entries: [`rt::tensordot`] is a scheduler over this module (every contraction it plans
runs on the chain); [`rt::vecdot`], [`rt::outer`] and [`rt::trace`] are their own kernels and do
not pass through here. `rt::ext_matmul` (the Array-API mixed-dtype promotion) is not implemented
for `DeviceCuda`. The device-level traits the core functions call,
[`DeviceMatMulAPI::matmul`](rstsr_core::operators::matmul::DeviceMatMulAPI::matmul) and
[`matmul_uninit`](rstsr_core::operators::matmul::DeviceMatMulAPI::matmul_uninit), take raw buffers
and layouts; the implementation is [the chain](#which-cuda-routine-runs).

# Which CUDA routine runs

## The chain

`c = alpha * (a @ b) + beta * c` is served by a chain of drivers, tried in order until one takes
the problem. Which drivers are *candidates* is a pure table of the type triple and the compute
mode; whether a candidate can *express the layout* is answered at run time by each driver, which
declines with a reason and hands the problem on. A driver that declines enqueues nothing, so
fall-through cannot change the answer, and an error is returned only when nothing ran.

| [`MatMulDrivers`] flag | routine | serves | declines when |
|---|---|---|---|
| `CUBLAS_LT` | `cublasLtMatmul`; the algorithm is chosen by `cublasLtMatmulAlgoGetHeuristic` and cached per problem | every cuBLAS triple below, `Fast16BF` included | the layout has no cuBLAS expression; no algorithm for the problem; `libcublasLt` is not loadable; the workspace could not be allocated |
| `CUBLAS_EX` | `cublasGemmEx` / `cublasGemmStridedBatchedEx`, the compute type as an argument | every cuBLAS triple below but the FP64 fixed-point emulation | as above, `libcublas`; the fixed-point emulation |
| `CUBLAS` | the typed entries `cublasSgemm`/`Dgemm`/`Cgemm`/`Zgemm` (and strided-batched), the mode as a handle math mode; `cublasSsyrk`/`Dsyrk`/`Csyrk`/`Zsyrk` for [the symmetric form](#the-symmetric-form) | `f32`, `f64`, `c32`, `c64` | as above; a mode with no handle math mode (`Fast16BF`; every mode but `Default` without the CUDA 12.9 bindings) |
| `TILED` | the tiled SIMT kernel (`drivers/tiled.rs`, `64 x 64` tiles of the output) | `f32` and `f64`, one dtype throughout; the output strides are free (a transposed output), and an operand needs a unit-stride or extent-1 axis (a reversed operand whose other axis is contiguous) | a batch that is not one linear run; an operand with no unit-stride axis; more than 65535 row tiles or batch slices |
| `NAIVE` | one thread per output element (`drivers/naive.rs`), full layout addressing | every type triple and every layout the rule table admits | a batch of rank over 16 |

The naive driver accepts everything, so the chain is total for the triples of the rule table. The
tiled and naive kernels are mode-agnostic: they compute in the output precision or better, which
every mode allows, so only the cuBLAS drivers read the mode.

Before the chain, one special form is tried: a fresh output of `a @ a.t()` is half the work of the
gemm and runs as one `?syrk` call when the conditions hold (see
[the symmetric form](#the-symmetric-form)).

## Type triples

The candidate table is keyed by `(a, b) -> c`. The middle column is what cuBLAS is told: the
storage type of the three matrices and the compute type; "32F" is the family of
[`ComputeType32F`], whose mode the flags or the call carry.

| `(a, b) -> c` | storage, compute | candidates, in order |
|---|---|---|
| `f32 -> f32` | `CUDA_R_32F`, 32F (any mode) | cuBLASLt, `cublasEx`, cuBLAS, tiled, naive |
| `f64 -> f64` | `CUDA_R_64F`, 64F (`Default`, `Pedantic`) | cuBLASLt, `cublasEx`, cuBLAS, tiled, naive |
| `f64 -> f64` under `EmulatedFixedPoint` | fixed-point emulation, a handle math mode | cuBLAS, tiled, naive |
| `Complex<f32> -> Complex<f32>`, `Complex<f64> -> Complex<f64>` | `CUDA_C_32F` / `CUDA_C_64F`, 32F / 64F | cuBLASLt, `cublasEx`, cuBLAS, naive |
| `f16 -> f16`, `bf16 -> bf16` | `CUDA_R_16F` / `CUDA_R_16BF`, 16F, `f32` scales (`f16` under `Acc16F`) | cuBLASLt, `cublasEx`, naive |
| `f16 -> f32`, `bf16 -> f32` | half storage, `CUDA_R_32F` output, 32F, `f32` scales | cuBLASLt, `cublasEx`, naive |
| integers (`i8` to `i64`, `u8` to `u64`, `isize`, `usize`); operands of different dtypes (`f32 @ f64 -> f64`, `f16 @ bf16 -> f32`, `f32 @ Complex<f64> -> Complex<f64>`, ...) | no cuBLAS problem | naive |

A mixed triple needs no entry of its own: [`rt::matmul_from`] and [`rt::matmul_with_output`] carry
no `Mul` bound, and the naive kernel casts each operand to the output type (the accumulator is
`f32` for a half output, the output type otherwise) and rounds once. The product is exact for
integers (modulo overflow) and within one rounding of the CPU kernel for the real types; it is
DRAM-bound, which is why the naive driver runs last.

## The symmetric form

`a @ a.t()` into a fresh output of one type (`f32`, `f64`, `c32`, `c64`) is run as one cuBLAS
`?syrk` call before the chain is tried: `b` must be `a` with reversed strides in the same buffer
(a view, not a copy), `beta` must be zero, and the product unbatched. cuBLAS writes one triangle;
the other is mirrored across the diagonal by a small kernel. Which output types take the form is
[`MatMulUseSyrk`] in [`CudaRuntimeFlags::matmul_use_syrk`], not size:

- `Scientific` (the default): `f64`, `c32` and `c64`, the types that gain 1.3x to 3.6x on an
  RTX 5070 Laptop; `f32` stays on the gemm chain, because cuBLAS's `f32` syrk kernels lose
  against its gemm on thin operands (`k` well below `n`), by up to 2.3x.
- `Allow`: `f32` as well.
- `Disallow`: never; the gemm chain always.

The form follows the compute mode like any driver: `Default` and `Pedantic` take it, a mode that
changes the arithmetic (`Fast*`, the fixed-point emulation) declines it, because the typed `?syrk`
entries do not apply a math mode (measured: a `?syrk` under the TF32 math mode is plain `f32`
arithmetic) and the chain's gemm serves the mode instead. A product that looks alike but is not
(`beta != 0`, the conjugate transpose `a @ conj(a).t()` where `b` is another buffer, a copy of
the transpose, a batch) runs the chain as before.

## What cuBLAS cannot express

cuBLAS is column-major with one leading dimension per matrix and no transpose flag for the output,
so the three cuBLAS drivers (which share one argument derivation, `cublas_args.rs`) decline:

- an output whose `n` axis is not the contiguous one (a transposed output view, say): `C` must be
  presentable as `n x m` column-major through `ldc` alone;
- an operand with no unit-stride axis (a 2-D block taken every second element in both directions);
- a negative stride with extent over 1 (a reversed view): no negative leading dimension;
- a batch that is not one linear run per operand (`[4, 3]` with strides `[6, 1]`): the
  strided-batched entries take one step per operand;
- a dimension, a leading dimension or a batch stride over `i32`;
- a broadcast along a matmul axis (a zero stride with extent over 1).

A zero stride on an extent-1 axis is served (the leading dimension floors at 1); a broadcast
*batch* axis is served (step 0). What cuBLAS declines, the tiled kernel takes when it is an `f32`
or `f64` product with a unit-stride (or extent-1) axis on each operand and an affine batch (a
reversed operand whose other axis is contiguous, a transposed output, a broadcast along a matmul
axis whose other axis is contiguous), and the naive kernel takes all of it. A stepped operand (no
unit stride on either axis) or a non-affine batch therefore runs on the naive kernel.

## When no enabled driver can serve

An empty enable set, or a set none of whose members is a candidate or can express the layout, is
an error naming the triple, the shape, the enabled drivers and every refusal; nothing was
enqueued. [`rt::matmul`] panics with it, [`rt::matmul_f`] returns it:

```text
cuda matmul: no enabled driver can serve double x double -> double (2x2 @ 2x2, batch 1);
enabled MatMulDrivers(cublasLt); declined: cublasLt: cannot serve: no column-major expression
(stepped or reversed layout, no unit-stride operand axis, non-affine batch, or a dimension over i32)
```

# Choosing the routine: `matmul_drivers`

Every device value carries its flags; the candidates of a product are filtered by
[`CudaRuntimeFlags::matmul_drivers`] at launch. The set is built from the [`MatMulDrivers`]
constants with `|` and `-` (`all()`, `none()`, `minus`), and a set with one member forces that
driver: a product the driver cannot serve is then the error above, never a different driver.

```rust
# use rstsr::prelude::*;
# use rstsr::cuda::prelude::*;
# if !std::panic::catch_unwind(cuda::is_available).unwrap_or(false) { return; }
// this device value: the tiled kernel and the naive kernel, nothing from cuBLAS
let mut flags = CudaRuntimeFlags::default();
flags.matmul_drivers = MatMulDrivers::TILED | MatMulDrivers::NAIVE;
let device = DeviceCuda::new(0).with_flags(flags);

// or one driver alone, here `cublasGemmEx`
flags.matmul_drivers = MatMulDrivers::CUBLAS_EX;
let device = DeviceCuda::new(0).with_flags(flags);

// or everything but cuBLASLt (which also switches the symmetric form off)
flags.matmul_drivers = MatMulDrivers::all() - MatMulDrivers::CUBLAS_LT;
let device = DeviceCuda::new(0).with_flags(flags);
# let _ = (device, flags);
```

Typical uses are benchmarking one routine against another, ruling a library out when a product
looks wrong, and a machine without `libcublas` (where the cuBLAS drivers decline by themselves
and the kernels take over, so nothing has to be set).

The flags are scoped in two layers: a device value pins them with
[`with_flags`](crate::device::DeviceCuda::with_flags), and every device value without an override
reads the process-global default at launch, which [`set_default_flags`] replaces (start from
[`default_flags`] to change one field). `CudaRuntimeFlags` is `#[non_exhaustive]`: build it with
`CudaRuntimeFlags::default()` and assign the fields. Flags are part of the device's identity for
`same_device`, compared as stored: a tensor made on a device value with pinned flags and one made
on the plain device value are *not* `same_device`, even when the pinned flags equal the global
default, and a product of the two is a device mismatch. Upload all operands of a product through
one device value.

To see which routine ran, set [`CudaRuntimeFlags::trace_matmul`]: every launch then prints one
line to stderr with the triple (`m x k x n`, the batch, the mode), the driver tried and its
verdict, the symmetric form included:

```text
[rstsr-cuda matmul] float x float -> float 2x2x2 batch 1 mode Default: cublasLt -> ran
[rstsr-cuda matmul] double x double -> double 2x2x2 batch 1 mode Default: cublas (syrk) -> ran
[rstsr-cuda matmul] double x double -> double 2x2x2 batch 1 mode Default: cublasLt -> declined (cannot serve: no column-major expression (stepped or reversed layout, no unit-stride operand axis, non-affine batch, or a dimension over i32))
[rstsr-cuda matmul] double x double -> double 2x2x2 batch 1 mode Default: cublasEx -> declined (cannot serve: no column-major expression (stepped or reversed layout, no unit-stride operand axis, non-affine batch, or a dimension over i32))
[rstsr-cuda matmul] double x double -> double 2x2x2 batch 1 mode Default: cublas -> declined (cannot serve: no column-major expression (stepped or reversed layout, no unit-stride operand axis, non-affine batch, or a dimension over i32))
[rstsr-cuda matmul] double x double -> double 2x2x2 batch 1 mode Default: tiled -> declined (cannot serve: an operand has no unit-stride axis (strides of the left operand))
[rstsr-cuda matmul] double x double -> double 2x2x2 batch 1 mode Default: naive -> ran
[rstsr-cuda matmul] int x int -> int 2x2x2 batch 1 mode (): naive -> ran
```

# Compute modes

## The three families

The mode is keyed by the **output** type: [`CudaMatMulMode`] names the enum of the output type's
family, and the flags carry one field per family, so an `f64` call never reads the 32F field.

| flag | read by calls with an output of | variants |
|---|---|---|
| [`CudaRuntimeFlags::compute_type_32f`]: [`ComputeType32F`] | `f32`, `Complex<f32>`, or half operands into `f32` | `Default`, `Pedantic`, `Fast16F`, `Fast16BF`, `FastTF32` |
| [`CudaRuntimeFlags::compute_type_64f`]: [`ComputeType64F`] | `f64`, `Complex<f64>` | `Default`, `Pedantic`, `EmulatedFixedPoint` |
| [`CudaRuntimeFlags::compute_type_16f`]: [`ComputeType16F`] | `f16`, `bf16` | `Default`, `Pedantic`, `Acc16F` |

- `Default`: `CUBLAS_COMPUTE_32F` / `64F`, cuBLAS's default compute, which does not down-convert
  the inputs: `f32` products in `f32`, `f64` in `f64`; half products accumulate in `f32`.
- `Pedantic`: the `_PEDANTIC` compute types (`CUBLAS_PEDANTIC_MATH` on the classic handle): exact
  arithmetic semantics and the specified internal storage format. Served by every accelerated
  triple and never downgraded.
- `FastTF32`, `Fast16F`, `Fast16BF` (32F): allow down-converting the `f32` inputs to TF32 (10
  mantissa bits), FP16 or BF16 (7) for the tensor-core pipes. The crate's tests hold them to a
  relative error of 1e-2, 1e-2 and 5e-2 against an `f64` reference, where `Default` and `Pedantic`
  are held to 1e-5.
- `Acc16F` (16F): accumulate an `f16` product in `f16` (`CUBLAS_COMPUTE_16F`): faster on the
  tensor cores, and the sum keeps fewer bits than the output rounding alone gives. `bf16` has no
  16-bit accumulation.
- `EmulatedFixedPoint` (64F): emulate the `f64` product in fixed point, cuBLAS's
  `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH`, a win where the native FP64 rate is low (the consumer
  cards); 1e-9 in the tests. Real `f64` only, the classic driver only, and it needs `cudarc`'s CUDA
  13.0 bindings. [`CudaRuntimeFlags::fixedpoint_mantissa_bits`] sets the mantissa width; `0` keeps
  the library default, the only value `cudarc` 0.19 can honour (any other is an error naming the
  field).

## Which driver serves which mode

| mode | cuBLASLt | `cublasGemmEx` | classic cuBLAS | `?syrk` |
|---|---|---|---|---|
| `Default` | yes | yes | yes (handle untouched) | yes |
| `Pedantic` | yes | yes | yes, `CUBLAS_PEDANTIC_MATH` (CUDA 12.9 bindings or newer) | yes |
| `FastTF32`, `Fast16F` | yes | yes | yes, `CUBLAS_TF32_TENSOR_OP_MATH` / `CUBLAS_TENSOR_OP_MATH` (12.9 or newer) | declines, the gemm serves it |
| `Fast16BF` | yes | yes | declines, no math mode converts to bf16 | declines, the gemm serves it |
| `Acc16F` | yes | yes | — (no half entries) | — |
| `EmulatedFixedPoint` | — | — | yes, `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` (13.0 or newer) | declines, the gemm serves it |

A driver that cannot express a mode declines, and the next candidate runs; forced alone, the
decline is the error. The bindings are chosen by `CUDARC_CUDA_VERSION` (or `nvcc`), see the
[crate documentation](crate#running-the-tests): with the 12.6 bindings the classic entries take no
math mode at all, so `Pedantic` and the `Fast*` modes leave the classic driver out of a run, and
the chain still serves them on cuBLASLt and `cublasGemmEx`.

## Modes a triple cannot serve

A mode the triple cannot serve is the family default by the time a driver sees it, silently: the
`Fast*` modes convert `f32` inputs, so they do not apply to complex inputs nor to half inputs with
an `f32` output; `Acc16F` scales in `f16`, so a `bf16` output keeps the default; the fixed-point
emulation is real `f64` only, so `c64` keeps the default. `Pedantic` is served by every accelerated
triple and is never downgraded.

## Setting a mode

Through the device flags, for every product of the family on that device (the same two layers as
[`matmul_drivers`](crate::flags::CudaRuntimeFlags::matmul_drivers)):

```rust
# use rstsr::prelude::*;
# use rstsr::cuda::prelude::*;
# if !std::panic::catch_unwind(cuda::is_available).unwrap_or(false) { return; }
let mut flags = CudaRuntimeFlags::default();
flags.compute_type_32f = ComputeType32F::FastTF32; // every `f32` product of this device
let device = DeviceCuda::new(0).with_flags(flags);
# let _ = device;
```

Or per call, with [the entries of `TensorCudaMatMulAPI`](#the-compute-mode-of-one-call-matmul_from_mode):
only the mode moves to the call; `matmul_drivers`, `matmul_use_syrk`,
`fixedpoint_mantissa_bits` and `trace_matmul` still come from the device flags.

# Overlapping uploads with products on streams

Every op is issued on the stream of the device value its tensors carry and returns at once; the
[example above](#overlapping-uploads-with-products) overlaps the upload of chunk `i + 1` with the
product of chunk `i` with four ingredients, each of which the `bench_matmul_stream` benchmark
removes in turn to measure what it buys:

1. **Page-locked host memory.** The pageable `to_device` of a `DeviceCpuSerial` tensor copies
   through a staging buffer and then synchronizes the stream: the host waits for the copy, and for
   everything enqueued before it. A [`DeviceCudaPinned`](crate::pinned::DeviceCudaPinned) tensor
   (`from_vec` / `zeros` / `alloc`, wrapped by `tensor`) uploads through a non-blocking
   copy on the device value's stream; its event orders every later host access (a host read of the
   chunk waits for the copies recorded on it).
2. **Two streams.** [`DeviceCuda::new_stream`](crate::device::DeviceCuda::new_stream) returns a
   device value on a fresh stream of the same ordinal. The upload is issued on `dev_h`; the
   uploaded tensor is re-tagged onto `dev_c` with `to_device`, which for two device values of one
   ordinal is a zero-copy view; the product is then enqueued on `dev_c`, which is made to wait for
   the upload of its operands (an event recorded on `dev_h`), so the two streams proceed
   concurrently. Both device values are `same_device`, so a product may mix their tensors.
3. **Issue order per chunk.** Upload `i`, product `i`, upload `i + 1`, product `i + 1`, ...: an
   event recorded at issue time covers everything enqueued on its stream so far, so a product
   enqueued after all the uploads would wait for all of them, and the overlap is gone. Issue the
   stages per chunk, in stage order.
4. **Uploaded buffers stay alive until the final synchronize.** A device buffer is freed on its
   allocation stream, after the reads recorded on it: dropping the upload of chunk `i` inside the
   loop (correct, nothing breaks) makes `dev_h` wait for the product that reads it before the free,
   and so before the next upload; measured, that alone removes the overlap (76 ms against 47 ms on
   the run above). Keep the uploads in a `Vec` until after `dev_c.synchronize()`.

The products are read back after the synchronize (`to_device` to a CPU device, or a non-blocking
copy into a pinned buffer with `rstsr_cuda::pinned::transfer::d2h_pinned_async`, which is how a
third stream hides the downloads as well; see `bench_streaming`). For 4 chunks of `[4096, 4096]`
`f32` on an RTX 5070 Laptop: uploads 37.5 ms, products 37.2 ms, the pageable flow 142 ms, the
pinned flow on one stream 75 ms, on two streams 47 ms (the bound is the larger phase, 37.5 ms,
plus the first upload). With 8 chunks the same run gives 86 ms against 272 ms.

# Layouts and the default order

Every layout the rule table admits is served: strided and negative strides, broadcast and
non-affine batches, either default order; which driver takes it is [the chain's](#which-cuda-routine-runs)
business. A zero stride is served on an extent-1 axis; a broadcast along a matmul axis has no
cuBLAS expression and runs on the tiled or the naive driver. The inner dot (`[k] @ [k]`) is served as the
`1 x k @ k x 1` GEMM it is, which cuBLAS runs at memory bandwidth.

Where the batch axis sits follows the default order of the operands, as in every rstsr matmul:
the matmul axes are the **last two** axes under row-major and the **first two** under column-major,
so a batch of `[n, k]` matrices is `[p, n, k]` in row-major and `[n, k, p]` in column-major
(`a.transpose(...)` swaps the two matmul axes only: `[0, 2, 1]` and `[1, 0, 2]` respectively). A
tensor built as `[p, n, k]` on a column-major device is a batch of `[k, p]` matrices, and
`a @ a.t()` on it is refused as a shape mismatch. `asarray(..).into_shape(s)` keeps the logical
shape `s` whatever the order; only the strides follow the order, so the data of a column-major
tensor has to be given in column-major positions.

# Resources and thread safety

The cuBLAS drivers share one handle per stream, kept behind a lock: a `cublasHandle_t` is not
thread-safe, and every `DeviceCuda::new(0)` of a process runs on one default stream, so two host
threads may enter a `gemm` on one handle at the same time. The classic driver takes the lock
around the call and sets a handle math mode inside it, restoring the previous mode before the
lock is released; serializing the entries costs nothing measurable, since a `gemm` returns after
enqueueing onto the stream the work was ordered on anyway. cuBLASLt keeps one handle and one
heuristic cache per ordinal behind a lock of its own, held across the call (a problem is described
to it once, keyed by every descriptor field and the low address bits of the three pointers, since
an algorithm states an alignment requirement per matrix; up to 1024 entries, dropped whole when
full), and one 32 MiB workspace per stream, allocated on the first matmul that stream runs
through it: the kernels' scratch memory, so two streams of one ordinal never share it. A missing
`libcublas` or `libcublasLt` is probed once per stream and cached: the drivers decline from then
on, and the only signal is one panic message on stderr.

# Numerics

- The cuBLAS drivers compute as [the mode](#compute-modes) says. The tiled and naive kernels
  compute in the output precision or better whatever the mode: `f32` and `f64` products accumulate
  in that type, a half output accumulates in `f32`; they are exact for integers and within one
  rounding of the CPU serial kernel for the real types.
- Operands of different dtypes are cast to the output type in the naive kernel (no copy, no
  cuBLAS): `f16 @ f16 -> f32` accumulates in `f32`, `f32 @ f64 -> f64` rounds the `f32` operand
  to `f64` first.
- Integer products wrap on overflow, with C semantics, as NumPy does; the CPU reference devices
  panic in debug builds.
- The additions of a product are not in the CPU's order, and not in the same order on every driver
  (`Fast*` aside, cuBLASLt's algorithm and the tiled kernel's slab accumulation differ), so
  compare against the CPU with a tolerance, never with `==`; the crate-level page's numerics
  section spells out the three tiers. Reproducibility across runs holds: nothing here draws a
  random seed, and the cuBLASLt algorithm is a pure function of the problem key.

# Benchmarking and tests

```text
cargo bench -p rstsr-cuda --bench bench_matmul          # type triple x shape x driver; the symmetric form and the dot
cargo bench -p rstsr-cuda --bench bench_matmul_stream   # upload / product overlap, with the controls
cargo bench -p rstsr-cuda --bench bench_streaming       # pinned H2D -> compute -> D2H over three streams
cargo test -p rstsr-cuda --test test_matmul_modes       # every mode and triple, the layout matrix, both orders
cargo test -p rstsr-cuda --test test_matmul_forms       # the symmetric form, the inner dot, mixed dtypes
cargo test -p rstsr-cuda --test test_matmul_tiled       # the tiled kernel against cuBLAS
cargo test -p rstsr-cuda --test test_matmul_naive       # the naive kernel, the on-device oracle
```

`CUDARC_CUDA_VERSION` picks the cuBLAS bindings `cudarc` compiles, hence what the drivers can
serve (see the [crate documentation](crate#running-the-tests)); left unset, the version comes from
`nvcc`. Without a visible device every hardware test, and every example of this page under
`cargo test -p rstsr-cuda --doc`, skips and passes, so a green run on a machine without a GPU means
nothing ran.

[`rt::matmul`]: rstsr_core::prelude::rt::matmul
[`rt::matmul_f`]: rstsr_core::prelude::rt::matmul_f
[`rt::matmul_from`]: rstsr_core::prelude::rt::matmul_from
[`rt::matmul_from_f`]: rstsr_core::prelude::rt::matmul_from_f
[`rt::matmul_with_output`]: rstsr_core::prelude::rt::matmul_with_output
[`rt::matmul_with_output_f`]: rstsr_core::prelude::rt::matmul_with_output_f
[`rt::tensordot`]: rstsr_core::prelude::rt::tensordot
[`rt::vecdot`]: rstsr_core::prelude::rt::vecdot
[`rt::outer`]: rstsr_core::prelude::rt::outer
[`rt::trace`]: rstsr_core::prelude::rt::trace
[`set_default_flags`]: crate::flags::set_default_flags
[`default_flags`]: crate::flags::default_flags
[`CudaRuntimeFlags::trace_matmul`]: crate::flags::CudaRuntimeFlags::trace_matmul
[`CudaRuntimeFlags::matmul_drivers`]: crate::flags::CudaRuntimeFlags::matmul_drivers
[`CudaRuntimeFlags::matmul_use_syrk`]: crate::flags::CudaRuntimeFlags::matmul_use_syrk
[`CudaRuntimeFlags::compute_type_32f`]: crate::flags::CudaRuntimeFlags::compute_type_32f
[`CudaRuntimeFlags::compute_type_64f`]: crate::flags::CudaRuntimeFlags::compute_type_64f
[`CudaRuntimeFlags::compute_type_16f`]: crate::flags::CudaRuntimeFlags::compute_type_16f
[`CudaRuntimeFlags::fixedpoint_mantissa_bits`]: crate::flags::CudaRuntimeFlags::fixedpoint_mantissa_bits
