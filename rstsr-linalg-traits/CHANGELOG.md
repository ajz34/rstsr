# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/ajz34/rstsr/compare/rstsr-linalg-traits-v0.1.0...rstsr-linalg-traits-v0.1.1) - 2025-03-11

### Other

- update prelude
- cargo fmt/clippy fix
- make many traits to be dyn compatible, and move `Self: Sized` trait bound to associated functions that not dispatchable dynamically
- add to rstsr main crate by feature
- cargo fmt fix
- solve (general, symmetric, triangular)
- cholesky (add tril/triu support)
- [WIP] cholesky
- linalg::eigh (simple cases)
- wrap vector of ints to 1-D tensor
- update prelude
- add BlasThreadAPI for control thread numbers
- first scratch to implement linalg::inv for openblas
