# Changelog

## v0.9.0 -- 2026-10-02

v0.9 will mark as the final version of full human-driven. We will be more aggressive in
leveraging AI-assisted development in future versions.

API breaking changes

- Element/axes iteration is now view-only: `iter` / `indexed_iter` / `axes_iter` /
  `indexed_axes_iter` are methods of `TensorView` (iterators tied to the view's inner
  data lifetime), and `iter_mut` / `indexed_iter_mut` / `axes_iter_mut` /
  `indexed_axes_iter_mut` are consuming methods of `TensorMut`. Owned/arc/cow tensors
  must call `.view()` / `.view_mut()` first; iterating an owned temporary is a compile
  error. This removes an unsound lifetime transmute in `Tensor::iter` that could
  reference freed data (use-after-free UB). (RESTGroup/rstsr#103)
- `meshgrid` return types now split by input ownership: reference inputs
  (`Vec<&TensorAny>`, arrays of references) return `Vec<TensorCow>`, where `copy = false`
  yields true NumPy-style broadcast views over the inputs' memory (previously owned
  copies were returned on all paths); by-value overloads now take concrete `Tensor`
  inputs (the previous generic by-value form also accepted views inside the `Vec`; use
  the reference forms instead). (RESTGroup/rstsr#99)

New features

- `rt::nanargmin` / `rt::nanargmax` (with `_f`, `_all`, `_axes` variants and `Tensor`
  methods) following NumPy `nanarg*` semantics: NaN elements are skipped wherever they
  appear, and an all-NaN slice raises `InvalidValue`; available on `DeviceCpuSerial`
  and rayon devices, with new device traits `OpNanArgMinAPI` / `OpNanArgMaxAPI`.
  (RESTGroup/rstsr#100)
- `broadcast_arrays` with reference-input overloads returning stride-0 broadcast views
  (no copy), as in NumPy; prelude-exported via `BroadcastArraysAPI`. (RESTGroup/rstsr#99)
- New storage accessors with documented validity contracts:
  `DataRef::try_as_true_ref` / `as_slice_ref`, `DataMut::try_into_true_mut` /
  `into_slice_mut`. (RESTGroup/rstsr#103)

Enhancement

- Performance: 8-lane contiguous fast path for `argmin`/`argmax` (serial and rayon),
  about 50x on large contiguous f64 inputs; the rayon variant combines chunk results
  deterministically, fixing a pairing-order-dependent wrong index. (RESTGroup/rstsr#100)
- Performance: blocked (64x64 tiled) 2-D iteration fast path for strided elementwise
  kernels (`op_mutc_refa_refb_func`, `op_muta_refb_func`, serial and rayon), with
  negative strides and non-zero offsets supported; strided `a + b^T` at 2048^2 is
  about 3x faster. (RESTGroup/rstsr#101)
- Performance: 2-D transpose-pattern assignments (equal-shape `a.t()`-pattern pairs,
  fast strides exactly +1) are routed through cache-friendly blocked orderchange
  kernels in `assign`/`assign_arbitary` (all promote/uninit/plain families, serial and
  rayon). (RESTGroup/rstsr#102)
- `Layout::check_strides` no longer allocates in the common case and raises
  `InvalidLayout` on span overflow instead of wrapping in release mode.
  (RESTGroup/rstsr#105)
- The `uninitialized_vec` allocation family now has a written contract
  (`alloc_vec_contract.md`, rustdoc-included on all three allocation functions).
  (RESTGroup/rstsr#105)
- Wholesale API documentation retrofit of rstsr-core: every docstring example is
  verified as rustdoc doctests (now CI-gated) with `doc_draft` twin tests asserting
  captured printed output. (RESTGroup/rstsr#99)

Behavior change

- Writes through broadcast (stride-0) layouts are now rejected on every write path
  (in-place assign family, `mapi` family, matmul `*_with_output` drivers,
  `op_with_func` drivers, `vecdot_from_f`, and the `iter_mut`/`axes_iter_mut`
  constructors). Previously a broadcast output under rayon handed overlapping offsets
  to different threads (data race), and the serial path silently aliased writes.
  Constructing and reading a broadcast `TensorMut` stays legal; unary in-place ops on
  owned broadcasted tensors fall back to producing a fresh output.
  (RESTGroup/rstsr#105)
- `to_contig` decides view-vs-copy by NumPy-style contiguity flags (`c_contig()` /
  `f_contig()`) instead of exact layout equality; padded-singleton contiguous tensors
  (e.g. shape `[3, 1]`, stride `[1, 3]`) are no longer copied. (RESTGroup/rstsr#99)

API breaking changes (user should not feel that)

- `Send`/`Sync` bounds of the storage wrappers `DataRef`/`DataCow`/`DataArc`/
  `DataReference` tightened: a wrapper that can expose a shared reference is
  `Send`/`Sync` only when `C: Send + Sync` (previously `DataRef<Cell<i32>>` could be
  sent across threads). `IntoParallelIterator` for the axes parallel iterators
  additionally requires `B::Raw: Sync`. (RESTGroup/rstsr#104)
- `DeviceMatMulAPI` gains the required method `matmul_uninit` (write-only
  `c = alpha * a @ b` into `MaybeUninit` storage), implemented by all in-repo devices.
  (RESTGroup/rstsr#105)
- `DimShapeAPI::unravel_index_f`/`unravel_index_c` and `Layout::index_uncheck` are no
  longer `unsafe`; `ChangeableDefault::change_default` is now a safe fn (defaults
  backed by `AtomicU8`); faer owned-`Mat` `into_rstsr` gained a `T: Clone` bound (it
  now copies; see Bug Fix). `IntoRSTSR` is exported via the prelude `rstsr_traits`.
  (RESTGroup/rstsr#105)

Bug Fix

- Fix heap out-of-bounds in `rt::full((layout, fill_value))` when the user-supplied
  `Layout` has non-zero offset or negative strides; allocation now uses
  `bounds_index()`, exactly like `empty`/`zeros`/`ones`/`uninit`. (RESTGroup/rstsr#104)
- Fix allocation-size overflow in `aligned_uninitialized_vec` (`aligned_alloc` feature,
  rstsr-common): the byte count is now `checked_mul`'d and raises `RuntimeError` on
  overflow. (RESTGroup/rstsr#104)
- Fix pointer-provenance UB in parallel kernels: the write-through
  `c.as_ptr() as *mut` pattern is replaced by hoisted `AtomicPtr` at all rstsr-
  native-impl rayon kernel sites and the rstsr-sci-traits cdist rayon kernels; the
  batched-broadcast gemm parallel-outer branch and syrk write-back now derive
  per-task pointers with unique provenance. (RESTGroup/rstsr#105)
- Fix an unsound allocation re-home in faer owned-`Mat` `into_rstsr`
  (`mem::forget` + `Vec::from_raw_parts` mismatched the dealloc layout on numeric
  conversion); it now copies column-wise, and the zero-copy route is
  `mat.as_ref().into_rstsr()` (a `TensorView`). (RESTGroup/rstsr#105)
- Fix a data race on the process-wide `TensorIterOrder` default (was a `static mut`
  read by safe code on effectively every tensor operation). (RESTGroup/rstsr#105)
- Fix `axes_iter`/`axes_iter_mut` (and indexed variants) panicking on an empty axes
  list; `axes_iter(())` now yields a single whole-tensor view, matching the NumPy
  `ndindex()` convention. (RESTGroup/rstsr#105)
- Fix naive matmul kernels violating the `beta = 0` non-read convention: a non-finite
  value in `c` could propagate through `0 * c`. (RESTGroup/rstsr#105)
- Fix `diag`/`concatenate` dropping uninitialized memory as `T` for allocatable
  element types; both now use `uninit_impl` + `assign_uninit` + `assume_init_impl`.
  (RESTGroup/rstsr#105)
- Fix GEMM/SYHEMM/TRSM wrappers in rstsr-blas-traits passing allocation-base pointers
  (ignoring `layout.offset()`) and hard-coded `ldc`/`ldb = m`: for f-prefer views with
  non-zero offset or padded leading dimension, operands were read from a wrong window
  and outputs written at the wrong stride (silently wrong results plus writes into
  parent elements outside the view). (RESTGroup/rstsr#106)
- Fix `driver_gesvd` order detection in rstsr-blas-traits: an inverted condition took
  the col-major driver path for row-major input and vice versa. (RESTGroup/rstsr#106)
- Fix GESVD/GESDD on empty matrices (`min(m, n) == 0`): both now return LAPACK-shaped
  zero-filled factors directly, and `superb` is sized via `saturating_sub` (was
  `usize` underflow). (RESTGroup/rstsr#106)
- Fix `getrf` allocating `ipiv` with `n` entries (LAPACK contract is `min(m, n)`);
  `getri` now rejects a short `ipiv` instead of reading out of bounds.
  (RESTGroup/rstsr#106)
- Fix ColMajor `eye` returning shape `(n_cols, n_rows)`; the layout is now
  `[n_rows, n_cols].f()`, identical logical content under both orders.
  (RESTGroup/rstsr#99)

Dev infrastructure

- Update agent-workflow references to the shared rstsr-agents workflow; NumPy tracking
  headers are machine-independent (checkout location recorded per-developer in
  `AGENTS.local.md`). (RESTGroup/rstsr#98)
- CI: new doctest job for rstsr-core (`cargo test -p rstsr-core --doc --release`).
  (RESTGroup/rstsr#99)

## v0.8.0 -- 2026-09-01

Behavior change

- The `openmp` feature of rstsr-openblas is now a default feature, and `rstsr`
  default features propagate it whenever the `openblas` feature is on. (RESTGroup/rstsr#94)
  Default-feature builds must now link an OpenMP runtime (`gomp`/`omp`);
  pthread-built OpenBLAS libraries remain compatible. Opt out with
  `default-features = false, features = ["linalg"]`.
  This default-feature change is the main motivation for the minor version bump.

Enhancement

- Add `ExtNum::ext_sign` following NumPy `np.sign` semantics: `-1`/`0`/`1` for
  signed integers (comparison form, no overflow at the type minimum), `0`/`1` for
  unsigned integers, NaN/±inf and signed-zero handling for floats, `z / |z|` for
  complex, and the same rules for `f16`/`bf16`. (RESTGroup/rstsr#93)
  `rt::sign` now accepts integer dtypes, and the previous `x / |x|` formula
  divergences are fixed: `sign(±inf)` no longer returns NaN, and `sign(-0.0)`
  returns `+0.0`.

API breaking changes (user should not feel that)

- `ExtNum` gained the new required method `ext_sign`; crates implementing `ExtNum`
  themselves must add it. The closed `ExtNum` type set also replaces the open
  `ComplexFloat` bound as the fallback for `rt::sign`, so not all `ComplexFloat`
  types remain accepted. (RESTGroup/rstsr#93)

Bug Fix

- Fix `OpenBLASConfig::get_parallel` panicking when both `openmp` and
  `dynamic_loading` were disabled, even on pthread builds; it now panics only
  when the library itself reports an OpenMP build. (RESTGroup/rstsr#94)
- Fix `no_std` builds: use rstsr-cblas-base 0.1.2 (which fixes its own `no_std`
  support), and replace `std` float operations and imports with `core`/`alloc`
  equivalents via the `num` crate. (RESTGroup/rstsr#96)

## v0.7.10 -- 2026-08-14

API breaking changes (user should not feel that)

- Rename module `manuplication` to `manipulation` (rstsr-core). (RESTGroup/rstsr#89)
  Only direct `rstsr_core::tensor::manuplication` import paths move; the `rstsr`
  facade prelude is unaffected.

Enhancement

- Introduce `AxisError` and `IndexError` error types in rstsr-common, replacing
  generic errors raised on axis/index misuse. (RESTGroup/rstsr#91)

Bug Fix

- Fix NumPy-parity bugs in reshape, argmax/argmin, sign, indexing and
  not-equal comparisons. (RESTGroup/rstsr#92)
- Fix dynamic-linking setup of rstsr-openblas for development on macOS. (RESTGroup/rstsr#88)

## v0.7.9 -- 2026-07-22

Bug Fix

- Fix index_select/bool_select with zero-size mask (RESTGroup/rstsr#86)

## v0.7.8 -- 2026-06-25

Bug Fix

- Fix thread oversubscription of broadcast matmul (BLAS devices) (RESTGroup/rstsr#85)

## v0.7.7 -- 2026-06-22

Bug Fix

- Fix matmul with broadcast in multiple-dimensions

  Now `[1, M, K] * [B, K, N] -> [B, M, N]` (the 7th rule of [matmul](https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html) with better broadcast support) now works.

## v0.7.6 -- 2026-06-07

Enhancement (also behavior change)

- Supports broadcasted matmul in column major. (RESTGroup/rstsr#81)

Behavior change

- Will not use aligned allocation in any situation when operating system is not Linux/MacOS. (RESTGroup/rstsr#82)

## v0.7.5 -- 2026-05-24

All changes at RESTGroup/rstsr#80.

Enhancement

- Allow LowerExp/UpperExp (scientific float print).

API breaking changes (user should not feel that)

- Changes the behavior of Display print, now the tensor will only be referenced when printing at least for CPU devices.

We still make Debug print to copy the whole tensor. In this way, the debug print uses minimal trait bounds, easier for debugging in other devices. This behavior may also change if necessary.

## v0.7.4 -- 2026-05-21

All changes at RESTGroup/rstsr#79

Bug Fix

- Fix function `rt::vecdot_from`, which previously falsely returned result instead of panic.

Enhancement

- add associated function of `vecdot` series
- add trait function (use as associated function) `einsum_from`

## v0.7.3 -- 2026-04-13

Bug Fix

- Fix previous behavior that layout equality check does not involve offset. (issue RESTGroup/rstsr#77, pr RESTGroup/rstsr#78)

## v0.7.2 -- 2026-04-10

All changes at RESTGroup/rstsr#76.

API breaking changes

- Refactored the `flip` methods for `FlagSide` and `FlagUpLo` enums to return the flipped value directly instead of a `Result`, since the operation cannot fail. Updated all call sites to remove unnecessary `?` error propagation.

Enhancements

- Re-exported the BLAS traits prelude under the new feature in `rstsr/src/prelude.rs` for easier access.
- Made all BLAS flag enums (`FlagOrder`, `TensorIterOrder`, `FlagTrans`, `FlagSide`, `FlagUpLo`, `FlagDiag`, `FlagSymm`) derive `Serialize` and `Deserialize`, and added Serde `rename` attributes for better control over their string representations.

Crate structure

- Added `serde` as a dependency in the relevant `Cargo.toml` files and imported the necessary traits in `flags.rs`.
- Symlinked `CHANGELOG.md` to rstsr crate directory.

## v0.7.1 -- 2026-03-30

Documentation update:

- rstsr-tblis: Updated documentation for einsum functions.

## v0.7.0 -- 2026-03-30

API breaking changes:

- rstsr-core: Removed unnecessary trait bounds on `DeviceCreationAPI` and related traits. Deprecated `arange_int_impl` as device function. Removed `DimStrideAPI` trait. (RESTGroup/rstsr#74)
- rstsr-core: Changed `TensorDotAxes` to `AxesPairIndex<T>` for general usage. Changed vecdot device trait function definition. (RESTGroup/rstsr#74)
- rstsr-core: Refactored AxesIndex trait bounds for improved error handling. (RESTGroup/rstsr#71)

New features:

- rstsr-core: Added `matrix_transpose` function for array-api compliance. (RESTGroup/rstsr#73)
- rstsr-core: Added `vecdot` traits and implementation (parallel for rayon devices). (RESTGroup/rstsr#73)
- rstsr-core: Added `reshape_with_args`, `into_compatible_shape` functions for flexible reshape operations. (RESTGroup/rstsr#70)
- rstsr-tblis: Implemented `tensordot` using einsum. (RESTGroup/rstsr#73)

Enhancements:

- rstsr-core: Efficiency improvement for `sum_axes` and related reduction with contiguous memory optimization. (RESTGroup/rstsr#74)
- rstsr-core: Parallel `arange` and `linspace` for devices supporting rayon. (RESTGroup/rstsr#74)
- rstsr-core: Changed output layout rule for binary functions; contiguous axes parts are preserved. (RESTGroup/rstsr#74)
- rstsr-core: Different-type reduction shares same implementation with same-type reduction. (RESTGroup/rstsr#74)
- rstsr-core: Restructured linalg directory; updated API documentation style. (RESTGroup/rstsr#73)
- rstsr-core: Enhanced stride checking in layout and reshape functions. Added NumPy-style tests for reshape and expand_dims. (RESTGroup/rstsr#70, RESTGroup/rstsr#71)
- rstsr-core: Testing framework updated for DeviceCpuSerial. Added reshape tests. (RESTGroup/rstsr#69)

Fixes:

- rstsr-core: Fix `change_layout` (contig may still require data copy).
- rstsr-tblis: Fixed threading and prelude imports. (RESTGroup/rstsr#74)
- rstsr-core: Various manipulation function bug fixes (expand_dims, flip). (RESTGroup/rstsr#71)

Dev infrastructure:

- Introduced Claude Code configuration for AI-assisted development. (RESTGroup/rstsr#72)

## v0.6.2 -- 2025-11-15

MSRV specified to 1.82.0, written to Cargo.toml.

Functionality changes:

- rstsr-core: `asarray` will not panic when layout is not compact if pass `&[T]` or `&mut [T]` to give view/mut-view tensor. However, pass `Vec<T>` to give owned tensor will still panic if layout is not compact.

## v0.6.0 -- 2025-11-03

Refactor:

- rstsr-core: Split `tensor/manuplication.rs` into a module with folder. (RESTGroup/rstsr#67)

API breaking changes:

- rstsr-core: Supporting type-promotion for assign. (RESTGroup/rstsr#65)
- rstsr-dtype-traits: Split previous dtype trait `PromotionAPI` to two parts `DTypePromoteAPI` and `DTypeCastAPI`. (RESTGroup/rstsr#66)
- Refactor of some dtype-related traits and impl. (RESTGroup/rstsr#66)
- rstsr-core: Add lifetime annotation to `TensorRefAPI` / `TensorRefMutAPI` (these two traits are currently not applied to any implementations). (RESTGroup/rstsr#67)
- rstsr-common: Added traceback in error-handling, which changed the fundamental data structure of `Error` of RSTSR. (RESTGroup/rstsr#67)
- rstsr-common: `DimBaseAPI::const_ndims` is method (with `&self` as argument) now. (RESTGroup/rstsr#67)

Enhancements:

- rstsr-dtype-traits: add function `isclose`. (RESTGroup/rstsr#66)
- rstsr-core: add function `allclose`. (RESTGroup/rstsr#66)
- rstsr-core: simplifies some trait bounds for obtaining view of tensor. (RESTGroup/rstsr#66)
- rstsr-core: Add macro `tensor_from_nested!` (similar to `ndarray::array!`). (RESTGroup/rstsr#67)
- rstsr-common: Added `rstsr_unwrap` to print unwrap with traceback. Added cargo feature `traceback` in many crates for printing traceback info of place of panic. (RESTGroup/rstsr#67)
- rstsr-common: Added `normalize_axes_index` for development (similar to `numpy.normalize_axis_tuple`). (RESTGroup/rstsr#67)
- rstsr-common: Added `Option<i/usize>` to `AxesIndex` trait implementation. (RESTGroup/rstsr#67)
- Some API documents updated. (RESTGroup/rstsr#67)

Fixes:

- Fixes some possible memory-safety problems of `uninitialized_vec` in reduce implementations in rstsr-native-impl. (RESTGroup/rstsr#66)
- Fix `no_std`. (RESTGroup/rstsr#67)
- Fixed `rt::expand_dims` and `rt::flip` in multiple axes cases (accordance to NumPy). (RESTGroup/rstsr#67)

## v0.6.0-alpha.1 -- 2025-09-30

This is not a completed version. May have other API breaking changes before v0.6.0 release.

API breaking changes:

- Using `DeviceRawAPI<MaybeUninit<T>>` instead of `DeviceRawAPI<T>` for output types in device operator traits, changing both parameter types and trait bounds (RESTGroup/rstsr#60).
- Using `ExtNum`, `ExtFloat`, `ExtReal` for trait extensions to crate `num`. Removed previous traits (per-functionality) `AbsAPI`, `ReImAPI`, etc. This affects trait bounds (RESTGroup/rstsr/#63).
- Using data type promotion rules in several common functions in CPU device, changing trait bounds (RESTGroup/rstsr#64).

Enhancements:

- Added TBLIS plugin (RESTGroup/rstsr#60). Now Einstein summation is available from `rt::tblis::einsum` (with cargo feature `rstsr/tblis` enabled, and linkage of libtblis.so).
- Added `uninit`, `assume_init` (RESTGroup/rstsr#61).
- Added `take`, `all`, `any`, `count_nonzero`, `nextafter`, `reciprocal` (RESTGroup/rstsr#63).
- Using data type promotion rules for several common functions in CPU device implementations (sin, greater, etc., making comparasion of different types, or sin to integer list be evaluatable) (RESTGroup/rstsr#64).

Refactor:

- Changed most internal device implementation that works with `empty_impl` to `uninit_impl` (RESTGroup/rstsr#61).
- Changed directory structure for device implementations (currently BLAS devices are categorized to directory `crates-device`) (RESTGroup/rstsr#62).
- Removed previous rstsr-book in this repository (RESTGroup/rstsr#63).

Parts of API document also updated. 

## v0.5.1 -- 2025-09-01

MSRV specified:

- 1.84.1: with crate faer built;
- 1.82.0: other cases (due to crate `half` and rust language usage of `unsafe extern "C"`).

Code refactor:

- Revert MSRV from 1.87 to 1.84/1.82 (RESTGroup/rstsr#59).

## v0.5.0 -- 2025-08-26

API breaking changes:

- Revert to non-default for dynamic loading (RESTGroup/rstsr#57)

Fixes (with API breaking behavior change):

- Change Behavior of `DeviceCpuRayon::generate_pool` (RESTGroup/rstsr#56, RESTGroup/rstsr-ffi#9)
- Fix KML threading lock in LAPACK functions (RESTGroup/rstsr#58)

## v0.4.1 -- 2025-08-05

Enhancements:

- Added CPU devices (backends) MKL ([#48](https://github.com/RESTGroup/rstsr/pull/48)), BLIS ([#49](https://github.com/RESTGroup/rstsr/pull/49)), AOCL ([#51](https://github.com/RESTGroup/rstsr/pull/51)), KML ([#53](https://github.com/RESTGroup/rstsr/pull/53))

Possible API breaking change:

- For conversion between CBLAS flags and RSTSR flags (defined in crate rstsr-common), previously CBLAS flags are in crate rstsr-lapack-ffi. Now those flags are defined in rstsr-cblas-base, and been applied in all FFI crates (at [RESTGroup/rstsr-ffi](https://github.com/RESTGroup/rstsr-ffi)).

Actions:

- Added ARM support ([#52](https://github.com/RESTGroup/rstsr/pull/52))

## v0.4.0 -- 2025-07-25

API breaking change: Supporting dynamic loading for OpenBLAS ([#47](https://github.com/RESTGroup/rstsr/pull/47))

- Update `rstsr-lapack-ffi` and `rstsr-openblas-ffi` version to v0.4.
- Default to `dynamic_loading` for using OpenBLAS.
- Changes internal ways to call BLAS and LAPACK functions.

If compile time and disk usage becomes very large for `rstsr-openblas-ffi`, you may wish to set those options in Cargo.toml:

```toml
[profile.dev.package.rstsr-lapack-ffi]
opt-level = 0
debug = false

[profile.dev.package.rstsr-openblas-ffi]
opt-level = 0
debug = false
```

## v0.3.10 -- 2025-07-22

Fix:
- Fix unpack_tri signature
- Fix gemm/syrk bug when k=0 ([#46](https://github.com/RESTGroup/rstsr/pull/46))

## v0.3.9 -- 2025-07-07

Enhancements:
- Feature with optional dependencies. Now in main crate `rstsr`, using feature `faer` and `openblas` along with `linalg` and `sci` should be ok, without explicitly declaring `rstsr-linalg-traits` and `rstsr-sci-traits` as dependencies. 

## v0.3.8 -- 2025-07-04

Bug Fix:
- Tested complex linalgs for Faer and OpenBLAS devices.

Enhancements:
- DeviceFaer: generalized eigen, triangular solve ([#44](https://github.com/RESTGroup/rstsr/pull/44))

## v0.3.7 -- 2025-06-25

Bug Fix:
- Fix panic when layout iterator size is zero ([#42](https://github.com/RESTGroup/rstsr/pull/42))

Enhancements:
- Add common function numa_refb and refb_numa implementation ([#42](https://github.com/RESTGroup/rstsr/pull/42))
- Implement clone for Tensor and TensorCow ([#42](https://github.com/RESTGroup/rstsr/pull/42))
- Added into_pack_array, into_unpack_array as associated function of TensorAny ([#42](https://github.com/RESTGroup/rstsr/pull/42))
- linalg: Solve-related functions supports vector (Ix1) RHS ([#43](https://github.com/RESTGroup/rstsr/pull/43))

## v0.3.6 -- 2025-06-05

Bug Fix:
- OpenBLAS device OpenMP `get_num_thread` function ([#40](https://github.com/RESTGroup/rstsr/pull/40))
    - Note that threading control is not stablized. There may be an incoming API breaking change on this feature for v0.4+.

Enhancements:
- Feature addition (meshgrid, concat, stack, bool_select) ([#38](https://github.com/RESTGroup/rstsr/pull/38))
- Feature addition (cdist, lebedev_rule) ([#39](https://github.com/RESTGroup/rstsr/pull/39))

Refactor:
- Eliminate `Error: From<I::Error>` trait bound ([#38](https://github.com/RESTGroup/rstsr/pull/38))

## v0.3.5 -- 2025-05-22

Bug Fix:
- Fix too strict stride check ([#36](https://github.com/RESTGroup/rstsr/pull/36))

API Breaking Change:
- Remove `into_slice_mut` ([#35](https://github.com/RESTGroup/rstsr/pull/35))

Enhancements:
- Diagonal arguments now allows i32 as input

## v0.3.4 -- 2025-05-20

Bug Fix:
- Fix rayon parallel in `op_muta_refb_func_cpu_rayon` ([#32](https://github.com/RESTGroup/rstsr/pull/32))
- Fix for conversion to self-device ([#33](https://github.com/RESTGroup/rstsr/pull/33))

## v0.3.3 -- 2025-05-19

Bug Fix:
- Fix Faer linalg functions when tensor offset != 0 ([#30](https://github.com/RESTGroup/rstsr/pull/30)).

## v0.3.2 -- 2025-05-19

Summary
- Added linalg functions for `DeviceFaer` ([#28](https://github.com/RESTGroup/rstsr/pull/28)).

API Breaking Change (user should not feel that):
- updates Faer version to v0.22, seems that v0.20/v0.21 changes handling logic for complex values
- Conversion from/to Faer made simple (but API breaking)
- Matmul made simple (but API breaking), now requires `faer::traits::ComplexField` type (trait impl based), instead of manually dispatch types (macro_rules based)

Enhancements:
- Functions added: cholesky, det, eigh (does not include generalized eigh), eigvalsh (same to eigh), inv, pinv, solve_general, svdvals

## v0.3.1 -- 2025-05-16

API Breaking Change:
- Remove `ge`, `gt`, `ne`, ... in traits `TensorGreaterAPI`, `TensorNotEqualAPI`, ... ([#25](https://github.com/RESTGroup/rstsr/pull/25))

Enhancements:
- linalg: functions added: slogdet, det, svd, eigvalsh, svdvals, pinv ([#23](https://github.com/RESTGroup/rstsr/pull/23))
- Summation to boolean tensor ([#25](https://github.com/RESTGroup/rstsr/pull/25))
- Basic advanced indexing function `index_select` ([#26](https://github.com/RESTGroup/rstsr/pull/26))
- Added TensorCow support for binary arithmetic operations ([#22](https://github.com/RESTGroup/rstsr/pull/22))

Something for fun:
- Changed logo to be ABBA-like style ([#24](https://github.com/RESTGroup/rstsr/pull/24))

## v0.3.0 -- 2025-05-09

API Breaking Change:
- Now `rt::linalg::eigh` returns `EighResult`, instead of simple 2-element tuple (eigenvalues, eigenvectors) ([#18](https://github.com/RESTGroup/rstsr/pull/18)).
- Now Lapack bindings will use Lapack (Fortran FFI) instead of LAPACKE (C FFI) by default ([#18](https://github.com/RESTGroup/rstsr/pull/18)).

Enhancements:
- Now `TensorCow` can perfrom binary arithmetic operations, such like `2.0 * a.reshape((2, 3, 4))`. Note that in some cases, rust compiler/rust-analyzer may not be able to deduce type of this result [#22](https://github.com/RESTGroup/rstsr/pull/18).
- Performed various refactor to linalg functions.
- Lapack (Fortran FFI) is supported and used by default. It is implemented like LAPACKE (but in rust), and many codes are generated by AI ([#18](https://github.com/RESTGroup/rstsr/pull/18)).

Various code refactors:
- Move more macro_rule implementatios to duplicate.

## v0.2.7 -- 2025-04-15

Internal refactor:
- Move out crate `rstsr-openblas-ffi` to rstsr-ffi repository, changes FFI bindings ([#19](https://github.com/RESTGroup/rstsr/pull/19)).
- Now `row_major` and `col_major` features are mutually exclusive in complie time.

## v0.2.6 -- 2025-04-02

Feature addition:
- Column major is now supported ([#16](https://github.com/RESTGroup/rstsr/pull/16)).

## v0.2.5 -- 2025-03-31

Bug fix:
- fix `pack_tril` (correctness fix for col-major case).

## v0.2.4 -- 2025-03-25

Bug fix:
- fix `pack_tril` (correctness fix, trait bound fix).

## v0.2.2 -- 2025-03-25

API breaking changes:
- Rayon thread pool getter function `get_pool` changed, added `get_current_pool`, removed `get_serial_pool` ([#14](https://github.com/RESTGroup/rstsr/pull/14)).

Code style changes ([#15](https://github.com/RESTGroup/rstsr/pull/15))

## v0.2.1 -- 2025-03-24

Bug fix:
- fix rayon pool memory blow up ([#13](https://github.com/RESTGroup/rstsr/pull/13))

## v0.2.0 -- 2025-03-11

This release features on BLAS and linalg implementations. Currently, functions such as `cholesky`, `eigh`, `solve_general` in `rstsr-linalg-traits` have been implemented.

Also many enhancements in `rstsr-core`.

## v0.1.0

Initial release. Most features in Python Array API has been implemented.
