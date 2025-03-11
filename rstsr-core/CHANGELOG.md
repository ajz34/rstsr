# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/ajz34/rstsr/compare/rstsr-core-v0.1.0...rstsr-core-v0.1.1) - 2025-03-11

### Other

- bug fix: avoid the case that length of shape (each dimension > 1) can be zero, causes error in `check_strides`
- add function fill
- Reduction operation enhancement and refactor ([#9](https://github.com/ajz34/rstsr/pull/9))
- loosen trait bounds on Clone ([#8](https://github.com/ajz34/rstsr/pull/8))
- disable clone_from_slice currently
- add argmin, argmax for serial CPU
- diagonal (as slice method)
- use general assert instead of debug_assert
- Revert "refactor: use general assert instead of debug_assert"
- use general assert instead of debug_assert
- cargo fmt/clippy fix
- make many traits to be dyn compatible, and move `Self: Sized` trait bound to associated functions that not dispatchable dynamically
- cargo fmt fix
- add DeviceRayonAPI
- l2_norm
- fix complex reduce
- extend cpu_rayon capability for different output/intermediate type reduction
- changed rayon implementation by soft-link
- extend cpu_serial capability for different output/intermediate type reduction
- cholesky (add tril/triu support)
- tril, triu
- wrap vector of ints to 1-D tensor
- syev
- first scratch to implement linalg::inv for openblas
- add field derive_builder::UninitializedFieldError
- add some simple assoc fn for TensorMutable
- add ErrorCode type of error
- trsm
- add DataReference and TensorReference for handling both ref and mut tensor
- add DeviceComplexFloatAPI
- add flag to char
- change name TensorOrder to FlagOrder, previous name still usable
- simplify trait relations
- allow testing by manifests
- add fingerprint
- add alias
- add leading dimension associated methods to Ix2 tensor
- add c/f_contig/prefer to tensor associated methods
- from char (instead of try_into)
- add method to_prefer
- add char try_into and flip
- update rule for c/f-prefer, that last/first shape dim = 1 is considered as contiguous
- add enum TensorMutable
- add to_contig
- add uninitialized
