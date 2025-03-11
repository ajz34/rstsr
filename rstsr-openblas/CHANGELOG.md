# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/ajz34/rstsr/compare/rstsr-openblas-v0.1.0...rstsr-openblas-v0.1.1) - 2025-03-11

### Other

- Allow linalg function accept any dimension instead of only 2-D ([#10](https://github.com/ajz34/rstsr/pull/10))
- Reduction operation enhancement and refactor ([#9](https://github.com/ajz34/rstsr/pull/9))
- loosen trait bounds on Clone ([#8](https://github.com/ajz34/rstsr/pull/8))
- use general assert instead of debug_assert
- Revert "refactor: use general assert instead of debug_assert"
- use general assert instead of debug_assert
- cargo fmt fix
- changed rayon implementation by soft-link
- add to rstsr main crate by feature
- solve (general, symmetric, triangular)
- cholesky (add tril/triu support)
- tril, triu
- [WIP] cholesky
- cargo fmt fix
- linalg::eigh (simple cases)
- wrap vector of ints to 1-D tensor
- sygv, sygvd
- syevd
- syev
- update prelude
- change naming convention inside crate definition
- add BlasThreadAPI for control thread numbers
- first scratch to implement linalg::inv for openblas
- getri
- getrf
- trsm
- add DeviceComplexFloatAPI
- update linting
- syhemm
- change name TensorOrder to FlagOrder, previous name still usable
- simplify trait relations
- allow testing by manifests
- add fingerprint
- init and gemm
