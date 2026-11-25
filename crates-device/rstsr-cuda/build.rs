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

    let (major, minor) = cudarc_cuda_version().unwrap_or((0, 0));
    if (major, minor) >= (12, 9) {
        println!("cargo:rustc-cfg=cublas_math_mode");
    }
    if (major, minor) >= (13, 0) {
        println!("cargo:rustc-cfg=cublas_fixedpoint_math");
    }
}

/// `(major, minor)` of the CUDA version `cudarc` binds, or `None` when it cannot be told (nvcc
/// absent: `cudarc`'s own build script then fails unless it pins a version feature).
fn cudarc_cuda_version() -> Option<(u32, u32)> {
    // cudarc's own spelling of 12.6 is `12060`: `major`, `0`, `minor`, `0`.
    if let Ok(v) = std::env::var("CUDARC_CUDA_VERSION") {
        // cudarc's own spelling of 12.6 is `12060`: `major`, `0`, `minor`, `0`.
        if v.len() == 6 && v.as_bytes()[1] == b'0' && v.as_bytes()[5] == b'0' {
            if let (Ok(major), Ok(minor)) = (v[0..1].parse::<u32>(), v[2..4].parse::<u32>()) {
                return Some((major, minor));
            }
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
