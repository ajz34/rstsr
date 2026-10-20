# rstsr-kml tests

Entry binaries (ADR-0012, entry-binary test matrix):

| Target | Body | Notes |
|---|---|---|
| `entry_row_kml` | `core_func/` + `test_utils/` + `test_issues/` (symlinked from rstsr-core) | shared device-agnostic body, row-major `TESTCFG` |
| `tests_driver_impl` | `driver_impl/` (symlinked from `rstsr-blas-traits/src/blas_tests_impl/`) | BLAS driver conformance, `DeviceBLAS` = `DeviceKML` |
| `tests_linalg_func` | `linalg_func/` (same source, feature `linalg`) | LAPACK-backed linalg conformance |

## Divergences from the shared bodies

Dedicated-file exception (ADR-0012 §5):

- `driver_impl/f64/gesvd.rs` - real file, not a symlink. KML's `DGESVD`
  returns a non-unique `U` at zero singular values, so the shared test's
  `U` fingerprint check cannot apply; this copy drops it (see
  https://github.com/ajz34/issue-kml/tree/master/issue-dgesvd-full).
