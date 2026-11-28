//! A safe wrapper over the cuTENSOR library: RAII objects plus one function per operation.
//!
//! This is the machine layer of the CUDA contractions — no device traits, no high-level tensor
//! API; the einsum/tensordot entries are built on it, and it can be used directly for evaluation.
//! The C pipeline of every cuTENSOR 2.x call is
//! *tensor descriptors* (extent, stride, dtype, alignment) -> *operation descriptor* (the
//! operands, their modes and unary operators, the compute descriptor) -> *plan preference*
//! (algorithm, JIT mode) -> *workspace estimate* -> *plan* -> one execute call on a stream. The
//! object layer here owns each of those as an RAII type (`Drop` calls the matching
//! `cutensorDestroy*`, raw pointers never public); the one-shot functions build the objects and
//! execute in one call, for callers that do not reuse a problem (planning costs 20-50 us per new
//! plan, plus a one-time ~28 ms lazy cuBLAS initialization inside the first cuBLAS-mapped
//! estimate; execute is ~2 us host-side).
//!
//! # Safety policy
//!
//! Memory safety comes from RAII, checked lengths and the borrows. Beyond that the wrapper
//! **refuses before calling cuTENSOR** (a named `InvalidValue`/`InvalidLayout` error) every case
//! the library accepts with wrong results or leaves undocumented — cuTENSOR validates very little
//! itself: stride-0 outputs and extent mismatches produce garbage on every version probed
//! (2.4.1-2.8.0), `opC != IDENTITY` is silently ignored through 2.7.0, and the scalar pointer type
//! is never checked. The wrapper never copies; materializing an operand is the caller's job.
//! Operands are dynamic-rank (`IxD`) only: convert a static-rank tensor with `.to_dim::<IxD>()`
//! before calling. Aliasing follows cuTENSOR's rule and is **not** checked: `A` / `B` must not
//! overlap the written output; `C` may alias `D` (that is the in-place accumulate form, `c =
//! None`).
//!
//! # Raw FFI at the boundary
//!
//! Modes are `&[i32]`, operators and algorithm choices are the `cudarc::cutensor::sys` enums, and
//! the compute descriptor is the raw `cutensorComputeDescriptor_t` (an opaque pointer; its nine
//! legal values come from [`compute_desc`], because they are exported `const` globals that the
//! bindings do not carry). Signatures name `cudarc` types directly; every `unsafe` stays inside
//! this module.
//!
//! # Errors
//!
//! Every function returns [`rstsr_core::prelude_dev::Result`]; a non-SUCCESS status is raised with
//! the pipeline stage and the library's own error string in the message, and the variant carries
//! the meaning (`NOT_SUPPORTED` -> `UnImplemented`, ... — see [`error`]). [`is_decline`] tells a
//! caller with a fallback whether the call was *not served* (try the next tier) rather than broken.
//! Unlike the rest of the crate there are no `_f` twins here: nothing panics to twin — every
//! function is already fallible.
//!
//! # Coverage
//!
//! Every tensor-semantics entry of the header is wrapped: the five operation families
//! (contraction, **trinary contraction** — gated on libcutensor >= 2.1.1, reduction,
//! permutation, elementwise binary/trinary) with create/execute, the RAII handle, plan
//! preference and plan (estimate, required-workspace), the descriptor's scalar-type
//! verification, and the handle-level plan/kernel-cache management
//! (resize, read/write). Deliberately out of scope: the `cutensorLogger*` family (no tensor
//! semantics) and block-sparse support; `cutensorOperationDescriptorSetAttribute` is refused
//! by design — setting `SCALAR_TYPE` would defeat the wrapper's derived-type verification; the
//! `cutensorPlanPreferenceGetAttribute`/`SetAttribute` pair (the 2.6+ autotuning knobs, future);
//! the `MOVED_BYTES` attribute is block-sparse-only on the probed versions and is not exposed.
//!
//! # Loading
//!
//! `libcutensor` is dlopened at first use (no link-time dependency). The pip wheel ships only the
//! versioned `libcutensor.so.2`, which is not among the names cudarc searches: point
//! `LD_LIBRARY_PATH` at a directory containing a `libcutensor.so` symlink to it. The Rust tests
//! run against any installed version by pointing at a different directory.

pub mod compute_desc;
pub mod descriptor;
pub mod dtype;
pub mod error;
pub mod handle;
pub mod one_shot;
pub mod operation;
pub mod plan;
pub mod workspace;

pub use descriptor::TensorDescriptor;
pub use dtype::{CutensorDType, CutensorScalar};
pub use error::{is_decline, Stage};
pub use handle::CutensorHandle;
pub use one_shot::{contract, contract_trinary, elementwise_binary, elementwise_trinary, permute, reduce};
pub use operation::OperationDescriptor;
pub use plan::{Plan, PlanPreference};
pub use workspace::CutensorWorkspace;
