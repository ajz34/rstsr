# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/ajz34/rstsr/compare/rstsr-blas-traits-v0.1.0...rstsr-blas-traits-v0.1.1) - 2025-03-11

### Other

- solve (general, symmetric, triangular)
- [WIP] cholesky
- linalg::eigh (simple cases)
- wrap vector of ints to 1-D tensor
- sygv, sygvd
- syevd
- syev
- update prelude
- add BlasThreadAPI for control thread numbers
- add field derive_builder::UninitializedFieldError
- getri
- getrf
- trsm
- add DataReference and TensorReference for handling both ref and mut tensor
- add DeviceComplexFloatAPI
- syhemm
- change name TensorOrder to FlagOrder, previous name still usable
- simplify trait relations
- init and gemm
