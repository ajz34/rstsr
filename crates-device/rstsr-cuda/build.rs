//! Tells the crate which cuBLAS bindings `cudarc` was built with.
//!
//! `cudarc` gates part of its `cublas` bindings on the CUDA version it is built for (the
//! `CUDARC_CUDA_VERSION` environment variable, else `nvcc --version` under
//! `cuda-version-from-build-system`): `cublasSetMathMode` from 12.9 on, the fixed-point FP64
//! emulation math mode from 13.0 on. A dependency's features are not visible to `cfg`, so the
//! version is resolved here the same way `cudarc` resolves it, and turned into:
//!
//! - `cublas_math_mode`: the handle math-mode entries are bound;
//! - `cublas_fixedpoint_math`: `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` is bound.
//!
//! A `cudarc` version feature pinned by another crate of the build is not seen here; in that case
//! set `CUDARC_CUDA_VERSION` to the version that feature names.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDARC_CUDA_VERSION");
    println!("cargo:rustc-check-cfg=cfg(cublas_math_mode)");
    println!("cargo:rustc-check-cfg=cfg(cublas_fixedpoint_math)");
    println!("cargo:rustc-check-cfg=cfg(cutensor_bindings)");
    println!("cargo:rustc-check-cfg=cfg(cutensor_dynamic_loading)");

    let (major, minor) = cudarc_cuda_version().unwrap_or((0, 0));
    if (major, minor) >= (12, 9) {
        println!("cargo:rustc-cfg=cublas_math_mode");
    }
    if (major, minor) >= (13, 0) {
        println!("cargo:rustc-cfg=cublas_fixedpoint_math");
    }
    if cutensor_bindings(Some((major, minor))) {
        println!("cargo:rustc-cfg=cutensor_bindings");
        // Under `dynamic-linking` cudarc links `-lcutensor` and drops `culib()`, so the compute
        // descriptor globals must come from `extern` statics instead of `libloading` lookups.
        if std::env::var_os("CARGO_FEATURE_DYNAMIC_LINKING").is_none() {
            println!("cargo:rustc-cfg=cutensor_dynamic_loading");
        }
    }
}

/// Whether cudarc compiles its `cutensor` module for this resolved CUDA version: every supported
/// pin from 12.0 up (cudarc compiles the module out for the 11.4-11.8 pins).
fn cutensor_bindings(version: Option<(u32, u32)>) -> bool {
    version.is_some_and(|(major, _)| major >= 12)
}

/// `(major, minor)` of the CUDA version `cudarc` binds, or `None` when it cannot be told (nvcc
/// absent: `cudarc`'s own build script then fails unless it pins a version feature).
fn cudarc_cuda_version() -> Option<(u32, u32)> {
    if let Ok(v) = std::env::var("CUDARC_CUDA_VERSION") {
        // cudarc spells a version `format!("{major}0{minor}0")` and matches the whole value
        // against that pattern (`12060` is 12.6, `13000` is 13.0), panicking on anything else in
        // its own build script; so this parse sees every value that builds.
        if let Some((major, minor)) = parse_cudarc_version(&v) {
            return Some((major, minor));
        }
        // an unparsable value is cudarc's build script's error to report
        return None;
    }
    let Ok(output) = Command::new("nvcc").arg("--version").output() else {
        return None;
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    // line 4: `Cuda compilation tools, release 13.2, V13.2.78`
    let release = stdout.lines().nth(3).and_then(|l| l.split(", ").nth(1))?;
    let number = release.strip_prefix("release ")?;
    let (major, minor) = number.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Parse the `{major}0{minor}0` spelling `cudarc` matches `CUDARC_CUDA_VERSION` against: `12060`
/// is (12, 6), `13000` is (13, 0). The zeros are literal separators, and `cudarc` 0.19's supported
/// majors are two-digit and its minors one-digit, which pins the lengths here.
fn parse_cudarc_version(v: &str) -> Option<(u32, u32)> {
    let rest = v.strip_suffix('0')?;
    let (major, minor) = rest.split_once('0')?;
    if major.len() != 2 || minor.len() != 1 {
        return None;
    }
    Some((major.parse().ok()?, minor.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::parse_cudarc_version as parse;

    /// The spellings `cudarc`'s supported versions produce all parse to their version.
    #[test]
    fn cudarc_spelling_parses() {
        assert_eq!(parse("12060"), Some((12, 6)));
        assert_eq!(parse("12090"), Some((12, 9)));
        assert_eq!(parse("13000"), Some((13, 0)));
        assert_eq!(parse("13040"), Some((13, 4)));
        assert_eq!(parse("11080"), Some((11, 8)));
        assert_eq!(parse("12000"), Some((12, 0)));
    }

    /// Anything `cudarc` would panic on stays `None` here, including the six-character spelling
    /// this file once assumed.
    #[test]
    fn other_spellings_are_refused() {
        assert_eq!(parse("120600"), None);
        assert_eq!(parse("1206"), None);
        assert_eq!(parse("13.0"), None);
        assert_eq!(parse("1206x"), None);
        assert_eq!(parse(""), None);
    }

    /// The `cutensor_bindings` cfg follows the cudarc pin: on from 12.0, off for the 11.x pins
    /// cudarc compiles its `cutensor` module out for, and off when the version is unresolvable.
    #[test]
    fn cutensor_bindings_track_the_cudarc_pin() {
        assert!(super::cutensor_bindings(Some((13, 2))));
        assert!(super::cutensor_bindings(Some((12, 0))));
        assert!(!super::cutensor_bindings(Some((11, 8))));
        assert!(!super::cutensor_bindings(Some((11, 4))));
        assert!(!super::cutensor_bindings(Some((0, 0))));
        assert!(!super::cutensor_bindings(None));
    }
}
