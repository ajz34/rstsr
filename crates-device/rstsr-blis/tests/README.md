# rstsr-blis tests

Entry binaries (ADR-0012, entry-binary test matrix):

| Target | Body | Notes |
|---|---|---|
| `entry_row_blis` | `core_func/` + `test_utils/` + `test_issues/` (symlinked from rstsr-core) | shared device-agnostic body, row-major `TESTCFG` |
| `tests_driver_impl` | `driver_impl/` (symlinked from `rstsr-blas-traits/src/blas_tests_impl/`) | BLAS driver conformance, `DeviceBLAS` = `DeviceBLIS` |
| `tests_linalg_func` | `linalg_func/` (same source, feature `linalg`) | LAPACK-backed linalg conformance |

## Divergences from the shared bodies

None at entry level. Per-file dedicated exceptions (ADR-0012 §5): none yet.
