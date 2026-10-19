//! Zero-size operands at the BLAS/LAPACK boundary.
//!
//! These tests are deliberately OpenBLAS-only: the shared `blas_tests_impl`
//! sources are symlinked into every BLAS device. The `xerbla`-handler harness
//! that would turn an illegal parameter into a test failure is a future task;
//! the assertions here cover the returned shapes and the wrapper's own
//! no-op semantics.

mod blas3;
