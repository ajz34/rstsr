# rstsr-faer-py

Python binding of the rstsr faer device, built as a **validation instrument**
for the Python array API standard: a pyo3 extension module exposing the graded
namespace `rstsr_faer.api` on top of `DeviceFaer`, so the data-apis
[`array-api-tests`] suite can score rstsr against the standard.

## Not for general use

- **Rust users**: use [`rstsr`] (or `rstsr-core`) directly — this crate adds
  nothing for Rust, and it is not published to crates.io (`publish = false`).
- **Python users**: do not depend on this package. It is distributed as a wheel
  on PyPI, but its surface tracks what the conformance suite grades, not what a
  Python tensor library should offer.
- The crate is a pure **wrapper**: marshalling, validation and rstsr calls
  only, with no numeric algorithms in either layer. A capability rstsr lacks
  becomes a rust-side fix request recorded in the grading task's gap register,
  never a shim-side reimplementation. One deliberate exception — pre-casting
  operands to their promoted dtype for the single-dtype set / join / solve
  kernels — is documented in [`DTYPE-CASTS.md`](DTYPE-CASTS.md).

## Layout

| layer | location | owns |
|---|---|---|
| Rust (pyo3, cdylib `rstsr_faer`) | `src/` | tensor construction, dtype dispatch, DLPack bridge, rstsr calls |
| Python | `python/rstsr_faer/` | the graded `api` namespace: signatures, protocol objects, marshalling |

The Rust side is split across several crates under `crates-interop/rstsr-faer-py/`
(`binding/` here, plus `core/` for the dtype-erased handle and dispatch macros and
one crate per surface domain). The split is a build-memory fix: one rustc
monomorphizing the whole dtype-dispatch surface needed ~45 GB.

## Build and grade locally

The workspace pins its own nightly toolchain (`rust-toolchain.toml`, at the
workspace root). [maturin] must run from the crate directory, not from the
workspace root:

```bash
cd crates-interop/rstsr-faer-py/binding
maturin build --release -i "$TEST_PY" -o /tmp/wheels
"$TEST_PY" -m pip install --force-reinstall --no-deps /tmp/wheels/rstsr_faer_py-*.whl
```

Grading (suite checkout and pins, gap files, red-map runs) is documented in
skill `rstsr-faer-py-tests` of the `rstsr-agents` repository; run reports land
in a scratch directory, never in this repository.

[`rstsr`]: https://crates.io/crates/rstsr
[`array-api-tests`]: https://github.com/data-apis/array-api-tests
[maturin]: https://www.maturin.rs/
