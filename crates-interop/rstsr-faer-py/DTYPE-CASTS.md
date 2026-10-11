# Dtype casting in the shim (shim-side "dtype-cheating")

`rstsr-faer-py` is a wrapper: it marshals, validates, and calls rstsr, and
carries no numeric algorithms in either layer (see `readme.md`). This note
records the one place where that boundary is deliberately stretched:
**pre-casts** — the Python layer casts operands *before* an rstsr call, so that a
multi-dtype array-API operation can be served by an rstsr **single-dtype**
kernel.

## Why a pre-cast is "cheating"

The array API derives the result of a multi-operand operation by type promotion
(`result_type(x1.dtype, x2.dtype)`). rstsr implements promotion as a
**compile-time** relation (`DTypePromoteAPI`), and a few of its kernels — the
set, join, and linear-solve kernels — are bound to a *single* dtype, so a mixed
operand pair never reaches them. The shim bridges the gap by casting each
operand to its promoted common dtype first and then calling the single-dtype
kernel.

The graded result is numerically what the standard asks for, but the promotion
happens *outside* rstsr, on the Python side: it is a workaround for a rust-side
gap, not a native capability. That is the sense in which it is "cheating" — the
conformance suite cannot tell, but the cast is not part of rstsr's contract, and
a rust-side promotion would make it unnecessary.

This is **not** the same as the casts the standard itself mandates — weak-scalar
marshalling (`_scalar_operand`), the accumulation-dtype rule for
`sum`/`prod`/`trace` (`_accumulated`), the `mean`/`std`/`var` int-to-float
default (`_float_input`), `clip` bounds, and `astype` itself. Those are the
spec's own semantics, recorded in the gap register as shim-side *binding* work,
not as a bridge over a missing rstsr feature. Only the four sites below are
"dtype-cheating".

## The promotion workarounds

| op | site (`python/rstsr_faer/api.py`) | cast | reach |
|---|---|---|---|
| `linalg.solve` | `solve` | both operands → `_promote_pair(x1.dtype, x2.dtype)` | input **and** result dtype |
| `isin` | `isin` | both operands → `_common_int_dtype(x1.dtype, x2.dtype)` | input only (result is always `bool`) |
| `concat` | `concat` via `_cast_parts` | every part → `_promote_pair` folded over the parts | input only |
| `stack` | `stack` via `_cast_parts` | every part → `_promote_pair` folded over the parts | input only |

Notes:

- `solve` is the only one where the cast also fixes the **result** dtype: rstsr's
  solve (`bin_fc!` in `src/linalg.rs`) is same-dtype, so the promoted pair both
  satisfies the kernel and yields `result_type(x1.dtype, x2.dtype)`.
- `isin` carries a **hand-written** integer promotion table (`_INT_PROMOTE` /
  `_common_int_dtype`) for the mixed-sign rule; the join and solve paths reuse
  `_promote_pair`, whose float/complex arms mirror `DTypePromoteAPI`. Integer
  pairs only for `isin`: a mixed non-integer pair raises rather than casting.
- The Rust half of the shim never casts for promotion. The join bindings decline
  a mismatched variant outright (`typed_part` in `src/manipulation.rs`), which is
  why the cast lives in the Python `concat` / `stack`.

Each of these four sites is recorded in the grading task's gap register; the
coded references live there, not in this repository.
