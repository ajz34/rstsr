//! Unary reductions on the CUDA backend: eleven rows, of which nine fit a **single-accumulator**
//! fold (kernel in `kernels::reduce_unary`) and two need a **pair** (`kernels::reduce_pair`).
//!
//! | trait | op | acc | identity | note |
//! |---|---|---|---|---|
//! | `OpSumAPI` | `Sum` | `T` | `0` | |
//! | `OpProdAPI` | `Prod` | `T` | `1` | |
//! | `OpMeanAPI` | `Mean` | `T` | NaN | empty input divides 0/0, like the CPU |
//! | `OpL2NormAPI` | `L2Norm` | `T` | `0` | real dtypes only (`Real = Self`) |
//! | `OpMinAPI` | `Min` | `T` | `+inf` | NaN-propagating; empty input raises |
//! | `OpMaxAPI` | `Max` | `T` | `-inf` | NaN-propagating; empty input raises |
//! | `OpSumDtypeAPI` | `Sum` | `TOut` | `0` | `f32 → f64`, `i32 → f64` accumulate |
//! | `OpProdDtypeAPI` | `Prod` | `TOut` | `1` | |
//! | `OpMeanDtypeAPI` | `Mean` | `TOut` | NaN | |
//! | `OpAllAPI` | `All` | `int` | `1` | `bool` in, `bool` out |
//! | `OpAnyAPI` | `Any` | `int` | `0` | `bool` in, `bool` out |
//! | `OpCountNonZeroAPI` | `CountNonZero` | `u64` | `0` | `usize` out; real dtypes |
//! | `OpSumBoolAPI` | `CountNonZero` | `u64` | `0` | `bool` in, `usize` out |
//!
//! and the pair rows:
//!
//! | trait | op | acc | identity | note |
//! |---|---|---|---|---|
//! | `OpVarAPI` | `Var` | `(Σx, Σx²)` in `T` | `(0, 0)` | real dtypes only; one-pass `Σx²/n − (Σx/n)²` |
//! | `OpStdAPI` | `Std` | `(Σx, Σx²)` in `T` | `(0, 0)` | `sqrt` of the same expression |
//!
//! This is a sibling module to `operators.rs` rather than a block inside it: the nine impls, their
//! two helpers and their generic bounds are ~350 lines of one subject, and they are the only thing in
//! the crate with an accumulator type distinct from the operand type.
//!
//! # Why not the rest
//!
//! Four families are still absent, and the reasons are no longer "not typed out yet":
//!
//! - **`OpArgMinAPI`, `OpArgMaxAPI`, `OpNanArgMin/MaxAPI`, `OpUnraveledArgMin/MaxAPI`**: the
//!   accumulator carries an index beside the value, and the combine is **lexicographic** — the two
//!   slots cannot be reduced independently the way `var`'s can. The tie-break is not a guess: the CPU
//!   keeps the **smaller logical index** (`native_impl::cpu_serial::reduction::
//!   reduce_all_unraveled_arg_fold_cpu_serial`, "return smaller index"), i.e. the smaller C-order
//!   raveled index over the reduced axes, independent of memory layout. That means the kernel needs
//!   the raveled index of each element, which the plan does not carry today: it enumerates the
//!   contracted axes in stride order, not logical order. Same machinery as `var` (a second slot),
//!   plus that index. Next.
//! - **`OpCumSumAPI`, `OpCumProdAPI`** (+ dtype variants): a *scan*, not a reduction — every
//!   intermediate value is an output, so the tree shape is different and the plan does not describe
//!   it. Their traits also allocate and reshape (`include_initial` shifts the output extent).
//! - **`OpVarDtypeAPI`, `OpStdDtypeAPI`**: the *device* trait is the same pair fold, but the public
//!   `var_with_dtype`/`std_with_dtype` wrappers also require `Op_MutA_RefB_API` for the `correction`
//!   rescale, and that trait takes a `dyn Fn(&mut T, &T)`. A closure cannot cross into a kernel, so
//!   no GPU backend can implement it honestly. Options, for whoever needs the mixed-precision path:
//!   implement `Op_MutA_RefB_API` to refuse at runtime (then `correction: None` works and `Some`
//!   errors), or split `correction` out of the wrapper so the kernel path does not name a closure.
//!   Not decided here, because either choice changes a core API for every backend.
//! - **`OpAllCloseAPI`**: `allclose_axes` is `unimplemented!()` on the CPU too, so implementing it
//!   only here would be inventing semantics rather than porting them.
//!
#![allow(clippy::type_complexity)]

use core::mem::MaybeUninit;

use crate::dtype::CudaReal;
use crate::kernels::reduce_pair::{reduce_pair_launch, PairOp};
use crate::kernels::reduce_plan::{classify_reduce, ReducePlan};
use crate::kernels::reduce_unary::{reduce_unary_launch, AccSpec, ReduceAcc, UnaryOp};
use crate::prelude_dev::*;

/// The four things every one of these methods does, in one place.
///
/// The output layout is the only part that mirrors the CPU (`layout_for_array_copy(&lm, K)`); it is
/// part of the observable contract, because a different axis order would give the same numbers with
/// different strides depending on which device ran the op.
///
/// The `DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>` bound is not redundant with
/// `DeviceRawAPI<MaybeUninit<TOut>, …>`: `full_impl`/`empty_impl`/`to_cpu_vec` all name the *plain*
/// projection, and a projection on a generic parameter has to be pinned explicitly or it stays
/// opaque. For the six same-type impls the compiler gets it from `DeviceRawAPI<T, Raw = CudaBuffer<T>>`
/// for free, which is why only the generic-`TOut` cases need it spelled out.
use crate::dtype::CudaDType;
fn reduce_axes_impl<TIn, TOut>(
    device: &DeviceCuda,
    op: UnaryOp,
    acc: AccSpec,
    a: &CudaBuffer<TIn>,
    la: &Layout<IxD>,
    axes: &[isize],
    empty_fill: TOut,
) -> Result<(Storage<DataOwned<CudaBuffer<TOut>>, TOut, DeviceCuda>, Layout<IxD>)>
where
    TIn: CudaDType,
    TOut: CudaDType,
    DeviceCuda: DeviceRawAPI<TIn, Raw = CudaBuffer<TIn>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>,
{
    reduce_axes_setup(device, a, la, axes, empty_fill, |c, lc, a, las, plan| {
        reduce_unary_launch(device, op, acc, c, lc, a, las, plan)
    })
}

/// Everything [`reduce_axes_impl`] does *except* the launch: split, output layout, plan, the
/// empty-input fill-versus-allocate decision.
///
/// Split out so the two-slot family ([`super::kernels::reduce_pair`]) reuses it verbatim. The plan
/// is the same object — a reduction's geometry does not care how many accumulators the body keeps —
/// and duplicating this prelude in a second helper is how the two would drift on the empty-input
/// rule.
fn reduce_axes_setup<TIn, TOut, F>(
    device: &DeviceCuda,
    a: &CudaBuffer<TIn>,
    la: &Layout<IxD>,
    axes: &[isize],
    empty_fill: TOut,
    launch: F,
) -> Result<(Storage<DataOwned<CudaBuffer<TOut>>, TOut, DeviceCuda>, Layout<IxD>)>
where
    TIn: CudaDType,
    TOut: CudaDType,
    DeviceCuda: DeviceRawAPI<TIn, Raw = CudaBuffer<TIn>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>,
    F: FnOnce(
        &mut CudaBuffer<MaybeUninit<TOut>>,
        &Layout<IxD>,
        &CudaBuffer<TIn>,
        &Layout<IxD>,
        &ReducePlan,
    ) -> Result<()>,
{
    let (las, lm) = la.dim_split_axes(axes)?;
    let lo = layout_for_array_copy(&lm, TensorIterOrder::K)?;
    let n_out = lo.size();

    // A unary reduction has no second operand, so the same layout goes in twice and the plan's `b`
    // strides come out as duplicates that the launcher ignores.
    let plan = classify_reduce(la, la, &lo, axes, axes, device.default_order())?;

    // An empty contraction writes nothing — the launcher returns before touching the buffer — so the
    // output is filled with the identity instead of being left uninitialised. `sum([]) == 0` must not
    // read device memory that happens to look like zero.
    let mut st: Storage<DataOwned<CudaBuffer<TOut>>, TOut, DeviceCuda> =
        if plan.n_contract == 0 || n_out == 0 {
            device.full_impl(n_out, empty_fill)?
        } else {
            // SAFETY: the kernel writes all `n_out` elements before the buffer is read, which is the
            // contract `empty_impl` asks the caller to uphold.
            unsafe { device.empty_impl(n_out)? }
        };

    if plan.n_contract > 0 && n_out > 0 {
        let buf: &mut CudaBuffer<TOut> = st.raw_mut();
        let buf_uninit: &mut CudaBuffer<MaybeUninit<TOut>> =
            unsafe { &mut *(buf as *mut CudaBuffer<TOut> as *mut CudaBuffer<MaybeUninit<TOut>>) };

        launch(buf_uninit, &lo, a, &las, &plan)?;
    }

    Ok((st, lo))
}

/// `*_all`: reduce every axis and take the single element back.
fn reduce_all_impl<TIn, TOut>(
    device: &DeviceCuda,
    op: UnaryOp,
    acc: AccSpec,
    a: &CudaBuffer<TIn>,
    la: &Layout<IxD>,
    empty_fill: TOut,
) -> Result<TOut>
where
    TIn: CudaDType,
    TOut: CudaDType,
    DeviceCuda: DeviceRawAPI<TIn, Raw = CudaBuffer<TIn>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>
        + DeviceToCpuAPI<TOut>,
{
    // Every axis, which the classifier turns into a rank-0 output of one element.
    let axes: Vec<isize> = (0..la.ndim() as isize).collect();
    let (st, _lo) = reduce_axes_impl(device, op, acc, a, la, &axes, empty_fill)?;

    let host: Vec<TOut> = DeviceToCpuAPI::<TOut>::to_cpu_vec::<DataOwned<CudaBuffer<TOut>>>(&st)?;
    match host.into_iter().next() {
        Some(v) => Ok(v),
        None => rstsr_raise!(InvalidValue, "cuda reduce: full reduction produced no element"),
    }
}

impl<T, D> OpSumAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    // Only `Zero` is needed, and only for the empty-input fill: the addition happens in the kernel.
    T: CudaDType + ReduceAcc + num::Zero,
    D: DimAPI,
{
    type TOut = T;

    fn sum_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        reduce_all_impl(self, UnaryOp::Sum, T::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
    }

    fn sum_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::Sum, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
    }
}

impl<T, D> OpProdAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    T: CudaDType + ReduceAcc + num::One,
    D: DimAPI,
{
    type TOut = T;

    fn prod_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        reduce_all_impl(self, UnaryOp::Prod, T::SPEC, a, &la.to_dim::<IxD>()?, T::one())
    }

    fn prod_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::Prod, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::one())
    }
}

impl<T, D> OpMeanAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    // `Float` covers the empty-input fill: an empty mean is `0/0`, and the CPU returns NaN too.
    T: CudaDType + ReduceAcc + num::Float,
    D: DimAPI,
{
    type TOut = T;

    fn mean_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        reduce_all_impl(self, UnaryOp::Mean, T::SPEC, a, &la.to_dim::<IxD>()?, T::nan())
    }

    fn mean_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::Mean, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::nan())
    }
}

impl<T, D> OpL2NormAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    // `Real = Self`, so the accumulator and the output are the operand's own type. A complex operand
    // needs `|x|²` in the fold and a real output buffer: not wired up, and refused at the call site
    // because `Complex` is not `Float`.
    T: CudaDType + ReduceAcc + num::Float,
    D: DimAPI,
{
    type TOut = T;

    fn l2_norm_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        reduce_all_impl(self, UnaryOp::L2Norm, T::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
    }

    fn l2_norm_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::L2Norm, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
    }
}

impl<T, D> OpMinAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    T: CudaDType + ReduceAcc + num::Float,
    D: DimAPI,
{
    type TOut = T;

    fn min_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for min")?;
        }
        // The fill is unreachable: a zero-size input raises above, so `n_contract` is never 0 here.
        reduce_all_impl(self, UnaryOp::Min, T::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
    }

    fn min_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for min")?;
        }
        reduce_axes_impl(self, UnaryOp::Min, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
    }
}

impl<T, D> OpMaxAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
        + DeviceCreationAnyAPI<T>
        + DeviceToCpuAPI<T>,
    T: CudaDType + ReduceAcc + num::Float,
    D: DimAPI,
{
    type TOut = T;

    fn max_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for max")?;
        }
        reduce_all_impl(self, UnaryOp::Max, T::SPEC, a, &la.to_dim::<IxD>()?, T::zero())
    }

    fn max_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        if la.size() == 0 {
            rstsr_raise!(InvalidValue, "zero-size array is not supported for max")?;
        }
        reduce_axes_impl(self, UnaryOp::Max, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::zero())
    }
}

// --- explicit-output-dtype variants -------------------------------------------------------------
//
// The same kernel with a wider accumulator: `@TACC@` is `TOut`, and the fold casts each element into
// it, which is what `into_cast` does on the CPU. The cast happens in the kernel, so the host never
// touches a value — hence no `Add`/`Mul`/`Clone` bound on `T`, and `T` need not implement
// `ReduceAcc`: only the accumulator type does.

impl<T, TOut, D> OpSumDtypeAPI<T, TOut, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>,
    T: CudaDType,
    TOut: CudaDType + ReduceAcc + num::Zero,
    D: DimAPI,
{
    fn sum_axes_dtype(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)> {
        reduce_axes_impl(self, UnaryOp::Sum, TOut::SPEC, a, &la.to_dim::<IxD>()?, axes, TOut::zero())
    }
}

impl<T, TOut, D> OpProdDtypeAPI<T, TOut, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>,
    T: CudaDType,
    TOut: CudaDType + ReduceAcc + num::One,
    D: DimAPI,
{
    fn prod_axes_dtype(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)> {
        reduce_axes_impl(self, UnaryOp::Prod, TOut::SPEC, a, &la.to_dim::<IxD>()?, axes, TOut::one())
    }
}

impl<T, TOut, D> OpMeanDtypeAPI<T, TOut, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>,
    T: CudaDType,
    TOut: CudaDType + ReduceAcc + num::Float,
    D: DimAPI,
{
    fn mean_axes_dtype(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)> {
        reduce_axes_impl(self, UnaryOp::Mean, TOut::SPEC, a, &la.to_dim::<IxD>()?, axes, TOut::nan())
    }
}

// --- predicates and counts ----------------------------------------------------------------------
//
// `all`/`any` accumulate in `int` and `count_nonzero` in `unsigned long long`, so these are the rows
// whose `AccSpec` is a constant rather than the operand's own type. The tree code is unchanged: only
// the fold differs, and it differs by a cast that the element type does not participate in.

impl<D> OpAllAPI<bool, D> for DeviceCuda
where
    Self: DeviceRawAPI<bool, Raw = CudaBuffer<bool>>
        + DeviceRawAPI<MaybeUninit<bool>, Raw = CudaBuffer<MaybeUninit<bool>>>
        + DeviceCreationAnyAPI<bool>
        + DeviceToCpuAPI<bool>,
    D: DimAPI,
{
    type TOut = bool;

    fn all_all(&self, a: &CudaBuffer<bool>, la: &Layout<D>) -> Result<bool> {
        // `all([]) == true`: the identity doubles as the empty-input fill, exactly as on the CPU.
        reduce_all_impl(self, UnaryOp::All, AccSpec::INT, a, &la.to_dim::<IxD>()?, true)
    }

    fn all_axes(
        &self,
        a: &CudaBuffer<bool>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::All, AccSpec::INT, a, &la.to_dim::<IxD>()?, axes, true)
    }
}

impl<D> OpAnyAPI<bool, D> for DeviceCuda
where
    Self: DeviceRawAPI<bool, Raw = CudaBuffer<bool>>
        + DeviceRawAPI<MaybeUninit<bool>, Raw = CudaBuffer<MaybeUninit<bool>>>
        + DeviceCreationAnyAPI<bool>
        + DeviceToCpuAPI<bool>,
    D: DimAPI,
{
    type TOut = bool;

    fn any_all(&self, a: &CudaBuffer<bool>, la: &Layout<D>) -> Result<bool> {
        reduce_all_impl(self, UnaryOp::Any, AccSpec::INT, a, &la.to_dim::<IxD>()?, false)
    }

    fn any_axes(
        &self,
        a: &CudaBuffer<bool>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(self, UnaryOp::Any, AccSpec::INT, a, &la.to_dim::<IxD>()?, axes, false)
    }
}

impl<T, D> OpCountNonZeroAPI<T, D> for DeviceCuda
where
    Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
        + DeviceRawAPI<MaybeUninit<usize>, Raw = CudaBuffer<MaybeUninit<usize>>>
        + DeviceCreationAnyAPI<usize>
        + DeviceToCpuAPI<usize>,
    // `CudaReal` is ints and floats: the fold compares the element against zero, which a complex
    // struct cannot do with the same expression.
    T: CudaDType + CudaReal,
    D: DimAPI,
{
    type TOut = usize;

    fn count_nonzero_all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<usize> {
        reduce_all_impl(self, UnaryOp::CountNonZero, AccSpec::U64, a, &la.to_dim::<IxD>()?, 0usize)
    }

    fn count_nonzero_axes(
        &self,
        a: &CudaBuffer<T>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(
            self,
            UnaryOp::CountNonZero,
            AccSpec::U64,
            a,
            &la.to_dim::<IxD>()?,
            axes,
            0usize,
        )
    }
}

impl<D> OpSumBoolAPI<D> for DeviceCuda
where
    Self: DeviceRawAPI<bool, Raw = CudaBuffer<bool>>
        + DeviceRawAPI<MaybeUninit<usize>, Raw = CudaBuffer<MaybeUninit<usize>>>
        + DeviceCreationAnyAPI<usize>
        + DeviceToCpuAPI<usize>,
    D: DimAPI,
{
    fn sum_all(&self, a: &CudaBuffer<bool>, la: &Layout<D>) -> Result<usize> {
        // Same row as `count_nonzero`: for a `bool`, "nonzero" is "true".
        reduce_all_impl(self, UnaryOp::CountNonZero, AccSpec::U64, a, &la.to_dim::<IxD>()?, 0usize)
    }

    fn sum_axes(
        &self,
        a: &CudaBuffer<bool>,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<
        (Storage<DataOwned<<Self as DeviceRawAPI<usize>>::Raw>, usize, Self>, Layout<IxD>),
    > {
        reduce_axes_impl(
            self,
            UnaryOp::CountNonZero,
            AccSpec::U64,
            a,
            &la.to_dim::<IxD>()?,
            axes,
            0usize,
        )
    }
}

/* #region var / std */

/// `var`/`std`: a two-slot fold, so they live in [`super::kernels::reduce_pair`] rather than in the
/// single-accumulator table above.
///
/// The pair is `(Σx, Σx²)` and the writer forms `Σx²/n − (Σx/n)²`, which is the CPU's one-pass
/// formula (`device_cpu_serial/reduction.rs`: `acc_2/size − mean*mean`). Both slots are the
/// accumulator's own type, so `TOut = T` and `Real = Self`.
///
/// **Real dtypes only**, for the same reason as `l2_norm`: a complex operand needs `Σx` in `T` beside
/// `Σ|x|²` in `T::Real`, i.e. two different accumulator types. `num::Float` is what excludes it.
///
/// `NaN` is the empty-input fill, matching the CPU: with `n = 0` the formula is `0/0 − 0`.
macro_rules! impl_var_std {
    ($tr:ident, $op:expr, $all:ident, $axes:ident) => {
        impl<T, D> $tr<T, D> for DeviceCuda
        where
            Self: DeviceRawAPI<T, Raw = CudaBuffer<T>>
                + DeviceRawAPI<MaybeUninit<T>, Raw = CudaBuffer<MaybeUninit<T>>>
                + DeviceCreationAnyAPI<T>
                + DeviceToCpuAPI<T>,
            T: CudaDType + ReduceAcc + num::Float,
            D: DimAPI,
        {
            type TOut = T;

            fn $all(&self, a: &CudaBuffer<T>, la: &Layout<D>) -> Result<T> {
                reduce_pair_all_impl(self, $op, T::SPEC, a, &la.to_dim::<IxD>()?, T::nan())
            }

            fn $axes(
                &self,
                a: &CudaBuffer<T>,
                la: &Layout<D>,
                axes: &[isize],
            ) -> Result<(
                Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>,
                Layout<IxD>,
            )> {
                reduce_pair_axes_impl(self, $op, T::SPEC, a, &la.to_dim::<IxD>()?, axes, T::nan())
            }
        }
    };
}

impl_var_std!(OpVarAPI, PairOp::Var, var_all, var_axes);
impl_var_std!(OpStdAPI, PairOp::Std, std_all, std_axes);

/// `*_axes` for the two-slot family: the shared prelude, then the pair launcher.
fn reduce_pair_axes_impl<TIn, TOut>(
    device: &DeviceCuda,
    op: PairOp,
    acc: AccSpec,
    a: &CudaBuffer<TIn>,
    la: &Layout<IxD>,
    axes: &[isize],
    empty_fill: TOut,
) -> Result<(Storage<DataOwned<CudaBuffer<TOut>>, TOut, DeviceCuda>, Layout<IxD>)>
where
    TIn: CudaDType,
    TOut: CudaDType,
    DeviceCuda: DeviceRawAPI<TIn, Raw = CudaBuffer<TIn>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>,
{
    reduce_axes_setup(device, a, la, axes, empty_fill, |c, lc, a, las, plan| {
        reduce_pair_launch(device, op, acc, c, lc, a, las, plan)
    })
}

/// `*_all` for the two-slot family.
fn reduce_pair_all_impl<TIn, TOut>(
    device: &DeviceCuda,
    op: PairOp,
    acc: AccSpec,
    a: &CudaBuffer<TIn>,
    la: &Layout<IxD>,
    empty_fill: TOut,
) -> Result<TOut>
where
    TIn: CudaDType,
    TOut: CudaDType,
    DeviceCuda: DeviceRawAPI<TIn, Raw = CudaBuffer<TIn>>
        + DeviceRawAPI<TOut, Raw = CudaBuffer<TOut>>
        + DeviceRawAPI<MaybeUninit<TOut>, Raw = CudaBuffer<MaybeUninit<TOut>>>
        + DeviceCreationAnyAPI<TOut>
        + DeviceToCpuAPI<TOut>,
{
    let axes: Vec<isize> = (0..la.ndim() as isize).collect();
    let (st, _lo) = reduce_pair_axes_impl(device, op, acc, a, la, &axes, empty_fill)?;
    let host: Vec<TOut> = DeviceToCpuAPI::<TOut>::to_cpu_vec::<DataOwned<CudaBuffer<TOut>>>(&st)?;
    match host.into_iter().next() {
        Some(v) => Ok(v),
        None => rstsr_raise!(InvalidValue, "cuda reduce: full reduction produced no element"),
    }
}

/* #endregion */
