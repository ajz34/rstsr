//! Development-only build script for `rstsr-aocl`.
//!
//! When the `RSTSR_DEV` environment variable is set, this script wires up
//! dynamic linking against an installed AOCL so that `cargo test` works out
//! of the box. It is a no-op for downstream users, who are expected to link
//! AOCL themselves (and to leave `RSTSR_DEV` unset).
//!
//! What it does — only under `RSTSR_DEV`, and only when the `dynamic_loading`
//! feature is *not* enabled (that feature resolves the library at runtime via
//! `libloading` instead, so link-time wiring would be redundant and wrong):
//!
//! 1. Collects library search directories from `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`,
//!    `LIBRARY_PATH`, `PATH` and `AOCL_ROOT` (its `lib` / `lib_LP64` subdirs), emitting
//!    `cargo:rustc-link-search` for each.
//! 2. Bakes an `-rpath` for the loader-style dirs so the test binary finds the dylibs at runtime
//!    without re-exporting the environment.
//! 3. Links `blis-mt` (CBLAS) and `flame` (LAPACK) — the AOCL libraries `rstsr-aocl-ffi`'s externs
//!    resolve against.

use std::path::PathBuf;

/// Library-path environment variables, in priority order.
const LIB_PATH_VARS: &[&str] = &["REST_EXT_DIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LIBRARY_PATH", "PATH"];

/// Vars whose directories are also baked into `-rpath` (runtime lookup).
const RPATH_VARS: &[&str] = &["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "REST_EXT_DIR"];

/// True only when we should perform link-time wiring.
fn is_dev_link() -> bool {
    std::env::var("RSTSR_DEV").is_ok() && !cfg!(feature = "dynamic_loading")
}

/// Collect existing directories from the given env vars, plus AOCL_ROOT-derived ones.
fn collect_dirs(vars: &[&str]) -> Vec<PathBuf> {
    let sep = if cfg!(windows) { ';' } else { ':' };
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for var in vars {
        let Ok(val) = std::env::var(var) else {
            continue;
        };
        for p in val.split(sep) {
            if p.is_empty() {
                continue;
            }
            let pb = PathBuf::from(p);
            if pb.is_dir() && seen.insert(pb.clone()) {
                out.push(pb);
            }
        }
    }
    if let Ok(aocl_root) = std::env::var("AOCL_ROOT") {
        let aocl_root = PathBuf::from(aocl_root);
        // source-tree layout (<root>/lib) and installed layout (<root>/lib_LP64)
        for sub in ["lib", "lib_LP64"] {
            let pb = aocl_root.join(sub);
            if pb.is_dir() && seen.insert(pb.clone()) {
                out.push(pb);
            }
        }
    }
    out
}

fn main() {
    println!("cargo:rerun-if-env-changed=RSTSR_DEV");
    println!("cargo:rerun-if-env-changed=AOCL_ROOT");
    for v in LIB_PATH_VARS {
        println!("cargo:rerun-if-env-changed={v}");
    }

    if !is_dev_link() {
        return;
    }

    let dirs = collect_dirs(LIB_PATH_VARS);
    for d in &dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }
    // embed rpath so the test binary resolves the dylibs at runtime without
    // re-exporting DYLD_LIBRARY_PATH / LD_LIBRARY_PATH.
    for d in collect_dirs(RPATH_VARS) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
    }

    println!("cargo:rustc-link-lib=blis-mt");
    println!("cargo:rustc-link-lib=flame");
}
