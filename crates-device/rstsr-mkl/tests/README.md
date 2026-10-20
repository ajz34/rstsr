# rstsr-mkl tests

Entry binaries (ADR-0012, entry-binary test matrix):

| Target | Body | Notes |
|---|---|---|
| `entry_row_mkl` | `core_func/` + `test_utils/` + `test_issues/` (symlinked from rstsr-core) | shared device-agnostic body, row-major `TESTCFG` |
| `tests_driver_impl` | `driver_impl/` (symlinked from `rstsr-blas-traits/src/blas_tests_impl/`) | BLAS driver conformance, `DeviceBLAS` = `DeviceMKL` |
| `tests_linalg_func` | `linalg_func/` (same source, feature `linalg`) | LAPACK-backed linalg conformance |

## Divergences from the shared bodies

None at entry level. Per-file dedicated exceptions (ADR-0012 §5): none yet.

## Local build notes

Prefer dynamic linking over the `dynamic_loading` feature (debug builds
with runtime loading are extremely slow): set `RSTSR_DEV=1` and
`MKLROOT=<oneapi>/mkl/<ver>` and the dev `build.rs` links `mkl_rt` with
rpath baked in. If a threaded call dies with
`libmkl_intel_thread.so: undefined symbol: omp_get_num_procs`, set
`MKL_THREADING_LAYER=GNU` (or `SEQUENTIAL`).

General guidance: avoid `dynamic_loading` unless it is the only way to
build — runtime symbol resolution adds heavy per-call overhead in debug
builds. The dev `build.rs` path is the default for `cargo test`.
