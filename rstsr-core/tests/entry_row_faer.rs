// Entry binary: row-major, DeviceFaer.
// See ADR-0012 (entry-binary test matrix) and skill `test-conventions`.
// Gated to `row_major` + `faer` via [[test]] required-features in rstsr-core/Cargo.toml.
mod core_func;
mod doc_draft;
mod test_issues;
mod test_utils;

pub use rstsr::prelude::*;
pub use std::sync::LazyLock;
pub use test_utils::TestCfg;

pub use DeviceFaer as DeviceType;

pub static TESTCFG: LazyLock<TestCfg<DeviceType>> = LazyLock::new(|| {
    let mut device = DeviceType::default();
    device.set_default_order(RowMajor);
    TestCfg::init(device, vec![], None)
});
