use crate::prelude_dev::*;

#[allow(clippy::type_complexity)]
#[duplicate_item(
    OpReduceAPI   func           func_all    ;
   [OpSumAPI   ] [sum_axes    ] [sum_all    ];
   [OpMinAPI   ] [min_axes    ] [min_all    ];
   [OpMaxAPI   ] [max_axes    ] [max_all    ];
   [OpProdAPI  ] [prod_axes   ] [prod_all   ];
   [OpMeanAPI  ] [mean_axes   ] [mean_all   ];
   [OpL2NormAPI] [l2_norm_axes] [l2_norm_all];
   [OpArgMinAPI] [argmin_axes ] [argmin_all ];
   [OpArgMaxAPI] [argmax_axes ] [argmax_all ];
   [OpNanArgMinAPI] [nanargmin_axes ] [nanargmin_all ];
   [OpNanArgMaxAPI] [nanargmax_axes ] [nanargmax_all ];
   [OpAllAPI   ] [all_axes    ] [all_all    ];
   [OpAnyAPI   ] [any_axes    ] [any_all    ];
   [OpCountNonZeroAPI] [count_nonzero_axes] [count_nonzero_all];
)]
pub trait OpReduceAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<Self::TOut>,
{
    type TOut;
    fn func_all(&self, a: &<Self as DeviceRawAPI<T>>::Raw, la: &Layout<D>) -> Result<Self::TOut>;
    fn func(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>)>;
}

/// `var`/`std`: the same reduction, plus the Array-API `correction` (NumPy `ddof`). The device
/// applies it as part of the fold: it knows `M` from the layout and computes the factor with
/// [`var_correction_factor`]. `None` means population variance. Element-type requirements stay in
/// the device impls, so the tensor layer needs no `TOut` bound.
#[allow(clippy::type_complexity)]
#[duplicate_item(
    OpReduceCorrAPI  func       func_all  ;
   [OpVarAPI      ] [var_axes ] [var_all  ];
   [OpStdAPI      ] [std_axes ] [std_all  ];
)]
pub trait OpReduceCorrAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<Self::TOut>,
{
    type TOut;
    fn func_all(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        correction: Option<f64>,
    ) -> Result<Self::TOut>;
    fn func(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
        correction: Option<f64>,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>)>;
}

/// Rescale from population to corrected variance (NumPy `ddof` / Array-API `correction`):
/// `M / (M - correction)`, square-rooted for `std`; NaN when `M - correction <= 0`.
///
/// Device-layer helper: the `var`/`std` impls call it with the reduced size they already know, and
/// own the conversion into their element type.
pub fn var_correction_factor(size: usize, correction: f64, is_std: bool) -> f64 {
    let m = size as f64;
    let d = m - correction;
    if d <= 0.0 {
        f64::NAN
    } else {
        let factor = m / d;
        if is_std {
            factor.sqrt()
        } else {
            factor
        }
    }
}

#[allow(clippy::type_complexity)]
#[duplicate_item(
    OpReduceDtypeAPI  func_axes_dtype  ;
   [OpSumDtypeAPI   ] [sum_axes_dtype  ];
   [OpProdDtypeAPI  ] [prod_axes_dtype ];
   [OpMeanDtypeAPI  ] [mean_axes_dtype ];
)]
pub trait OpReduceDtypeAPI<T, TOut, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<TOut>,
{
    fn func_axes_dtype(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)>;
}

/// `var`/`std` with an explicit accumulator: same `correction` contract as [`OpVarAPI`] /
/// [`OpStdAPI`].
#[allow(clippy::type_complexity)]
#[duplicate_item(
    OpReduceDtypeCorrAPI  func_axes_dtype  ;
   [OpVarDtypeAPI      ] [var_axes_dtype  ];
   [OpStdDtypeAPI      ] [std_axes_dtype  ];
)]
pub trait OpReduceDtypeCorrAPI<T, TOut, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<TOut>,
{
    fn func_axes_dtype(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
        correction: Option<f64>,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)>;
}

#[allow(clippy::type_complexity)]
#[duplicate_item(
    OpReduceAPI            func                    func_all             ;
   [OpUnraveledArgMinAPI] [unraveled_argmin_axes] [unraveled_argmin_all];
   [OpUnraveledArgMaxAPI] [unraveled_argmax_axes] [unraveled_argmax_all];
)]
pub trait OpReduceAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T>,
{
    fn func_all(&self, a: &<Self as DeviceRawAPI<T>>::Raw, la: &Layout<D>) -> Result<D>;
    fn func(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<IxD>>::Raw>, IxD, Self>, Layout<IxD>)>
    where
        Self: DeviceAPI<IxD>;
}

#[allow(clippy::type_complexity)]
pub trait OpSumBoolAPI<D>
where
    D: DimAPI,
    Self: DeviceAPI<bool> + DeviceAPI<usize>,
{
    fn sum_all(&self, a: &<Self as DeviceRawAPI<bool>>::Raw, la: &Layout<D>) -> Result<usize>;
    fn sum_axes(
        &self,
        a: &<Self as DeviceRawAPI<bool>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<usize>>::Raw>, usize, Self>, Layout<IxD>)>;
}

#[allow(clippy::type_complexity)]
pub trait OpAllCloseAPI<TA, TB, TE, D>
where
    D: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<bool>,
{
    fn allclose_all(
        &self,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
        isclose_args: &IsCloseArgs<TE>,
    ) -> Result<bool>;
    fn allclose_axes(
        &self,
        a: &<Self as DeviceRawAPI<TA>>::Raw,
        la: &Layout<D>,
        b: &<Self as DeviceRawAPI<TB>>::Raw,
        lb: &Layout<D>,
        axes: &[isize],
        isclose_args: &IsCloseArgs<TE>,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<bool>>::Raw>, bool, Self>, Layout<IxD>)>;
}

/// Custom user reduction over generic closures (init / fold / combine /
/// finalize); the accumulator `TS` and output `TO` are both free.
///
/// `combine` must be associative; within one output cell the input order is
/// sequential (row-major traversal), but the tree shape of `combine` is
/// device-defined (e.g. parallel chunking on the rayon device).
#[allow(clippy::type_complexity)]
pub trait OpReduceCustomAPI<T, TS, TO, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<TO>,
{
    fn reduce_all_custom<FI, FF, FC, FO>(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        f_init: FI,
        f: FF,
        f_sum: FC,
        f_out: FO,
    ) -> Result<TO>
    where
        TS: Clone,
        FI: Fn() -> TS + Send + Sync,
        FF: Fn(TS, T) -> TS + Send + Sync,
        FC: Fn(TS, TS) -> TS + Send + Sync,
        FO: Fn(TS) -> TO + Send + Sync;

    fn reduce_axes_custom<FI, FF, FC, FO>(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axes: &[isize],
        f_init: FI,
        f: FF,
        f_sum: FC,
        f_out: FO,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TO>>::Raw>, TO, Self>, Layout<IxD>)>
    where
        TS: Clone,
        TO: Clone,
        FI: Fn() -> TS + Send + Sync,
        FF: Fn(TS, T) -> TS + Send + Sync,
        FC: Fn(TS, TS) -> TS + Send + Sync,
        FO: Fn(TS) -> TO + Send + Sync;
}

/// Cumulative sum (scan) along a single axis; `TOut` is the input dtype
/// (array-api `dtype=None` semantics).
#[allow(clippy::type_complexity)]
pub trait OpCumSumAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<Self::TOut>,
{
    type TOut;
    fn cumulative_sum(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axis: isize,
        include_initial: bool,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>)>;
}

/// Cumulative product (scan) along a single axis; `TOut` is the input dtype
/// (array-api `dtype=None` semantics).
#[allow(clippy::type_complexity)]
pub trait OpCumProdAPI<T, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<Self::TOut>,
{
    type TOut;
    fn cumulative_prod(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axis: isize,
        include_initial: bool,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<Self::TOut>>::Raw>, Self::TOut, Self>, Layout<IxD>)>;
}

/// Cumulative sum with an explicit output dtype: the scan accumulates in
/// `TOut` (elements are cast inside the fold, no cast copy of the input).
#[allow(clippy::type_complexity)]
pub trait OpCumSumDtypeAPI<T, TOut, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<TOut>,
{
    fn cumulative_sum_dtype(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axis: isize,
        include_initial: bool,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)>;
}

/// Cumulative product with an explicit output dtype: the scan accumulates in
/// `TOut` (elements are cast inside the fold, no cast copy of the input).
#[allow(clippy::type_complexity)]
pub trait OpCumProdDtypeAPI<T, TOut, D>
where
    D: DimAPI,
    Self: DeviceAPI<T> + DeviceAPI<TOut>,
{
    fn cumulative_prod_dtype(
        &self,
        a: &<Self as DeviceRawAPI<T>>::Raw,
        la: &Layout<D>,
        axis: isize,
        include_initial: bool,
    ) -> Result<(Storage<DataOwned<<Self as DeviceRawAPI<TOut>>::Raw>, TOut, Self>, Layout<IxD>)>;
}
