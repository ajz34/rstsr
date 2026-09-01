# rstsr-openblas

The OpenBLAS device crate of RSTSR: exposes BLAS/LAPACK capability for the
OpenBLAS backend. This file is the glossary for batched-GEMM work; it records
the language agreed in design sessions, nothing else.

## Language

### Matmul semantics

**Common matmul**:
Matrix multiplication where every operand is at most 2-d: 2d×2d, mat-vec
(2d×1d), or vec-mat (1d×2d). One GEMM/GEMV call, no batch.
_Avoid_: normal matmul, plain matmul

**Broadcasted matmul**:
Matrix multiplication where any operand has rank ≥ 3; batch dims broadcast as
in NumPy's `matmul`. The case batched GEMM accelerates.
_Avoid_: batched matmul (reserve that for the trait/function names)

**Batch dims**:
All dims of an operand except the trailing two (the matrix dims).

**Broadcasted operand**:
An operand whose batch dims are size-1/absent and are expanded against the
other operand; expressed as a stride-0 batch, never by copying.

### Batch layouts

**Uniformly-strided batch**:
A batch whose dims flatten (in one order) to a single constant stride between
consecutive matrices. Stride 0 — the broadcast case — is the degenerate
uniform case. Rank-3 tensors are always uniformly-strided (one batch dim).
_Avoid_: evenly strided, spread

**Scattered batch**:
A batch whose matrix offsets are not one constant stride; requires a pointer
array. Arises e.g. from strided slices of outer batch dims at rank ≥ 4.
_Avoid_: unevenly strided, gathered

### Backend interfaces (OpenBLAS extensions)

**Grouped batched GEMM** (`?gemm_batch`):
Pointer-array batched GEMM with per-group parameter arrays and
`group_count`/`group_size`. OpenBLAS's only pointer-based batched interface.

**Pointer-based batched GEMM**:
What RSTSR calls the grouped interface used with `group_count = 1`: uniform
M/N/K/ld for the whole batch, one pointer per matrix. The general case RSTSR's
`batched_gemm` exposes (full grouped surface).

**Strided batched GEMM** (`?gemm_batch_strided`):
Batched GEMM with single parameters plus one stride per operand (in elements)
and the batch count. Requires a uniformly-strided batch on every operand.
