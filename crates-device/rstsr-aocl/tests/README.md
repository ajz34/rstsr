# rstsr-aocl tests

Entry binaries (ADR-0012, entry-binary test matrix):

| Target | Body | Notes |
|---|---|---|
| `entry_row_aocl` | `core_func/` + `test_utils/` + `test_issues/` (symlinked from rstsr-core) | shared device-agnostic body, row-major `TESTCFG` |
| `tests_driver_impl` | `driver_impl/` (symlinked from `rstsr-blas-traits/src/blas_tests_impl/`) | BLAS driver conformance, `DeviceBLAS` = `DeviceAOCL` |
| `tests_linalg_func` | `linalg_func/` (same source, feature `linalg`) | LAPACK-backed linalg conformance |

## Divergences from the shared bodies

None at entry level. Per-file dedicated exceptions (ADR-0012 §5): none yet.

## Local build notes

Prefer dynamic linking over the `dynamic_loading` feature (debug builds
with runtime loading are extremely slow): set `RSTSR_DEV=1` and
`AOCL_ROOT=<aocl>/<ver>/gcc/MT` and the dev `build.rs` links
`blis-mt` + `flame` with rpath baked in.

General guidance: avoid `dynamic_loading` unless it is the only way to
build — runtime symbol resolution adds heavy per-call overhead in debug
builds. The dev `build.rs` path is the default for `cargo test`.
