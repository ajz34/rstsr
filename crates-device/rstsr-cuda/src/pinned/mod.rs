//! Pinned (page-locked) host memory tensors.
//!
//! - [`PinnedBuffer<T>`]: owning page-locked host buffer with one ordering event;
//!   `DeviceRawAPI::Raw` of [`DeviceCudaPinned`].
//! - [`DeviceCudaPinned`]: the device type (`{ ordinal, base: DeviceCpuSerial }`); storage traits
//!   and the inherent constructors only — no operation traits.
//! - [`TensorPinnedHostViewAPI`] / [`TensorPinnedHostViewMutAPI`]: zero-copy `DeviceCpuSerial` (or
//!   any `Raw = Vec<T>` device) views over the bytes.
//! - [`transfer`]: the non-blocking pinned <-> device copies.
//! - [`conversion`]: `DeviceChangeAPI` edges (pinned <-> CUDA non-blocking, pinned <-> CPU devices
//!   by copy).
//!
//! Ordering model: every copy makes its stream wait for the buffer's event,
//! enqueues, then records the event; every host access synchronizes the event
//! first; drop synchronizes then frees. Sequential use across streams is safe,
//! simultaneous use of one buffer by two streams is not.

pub mod buffer;
pub mod conversion;
pub mod device;
pub mod transfer;
pub mod view;

pub use buffer::{pinned_flags, PinnedBuffer};
pub use device::DeviceCudaPinned;
pub use view::{TensorPinnedHostViewAPI, TensorPinnedHostViewMutAPI};
