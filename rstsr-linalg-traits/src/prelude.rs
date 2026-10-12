pub mod rstsr_traits {
    pub use crate::traits_def::{
        CholeskyAPI, DetAPI, EighAPI, EigvalshAPI, InvAPI, MatrixNormAPI, MatrixPowerAPI, MatrixRankAPI, PinvAPI,
        SLogDetAPI, SVDvalsAPI, SolveGeneralAPI, SolveSymmetricAPI, SolveTriangularAPI, VectorNormAPI, SVDAPI,
    };
}

pub mod rstsr_funcs {
    pub use crate::traits_def::{
        cholesky, cholesky_f, det, det_f, eigh, eigh_f, eigvalsh, eigvalsh_f, inv, inv_f, matrix_norm, matrix_norm_f,
        matrix_power, matrix_power_f, matrix_rank, matrix_rank_f, pinv, pinv_f, slogdet, slogdet_f, solve_general,
        solve_general_f, solve_symmetric, solve_symmetric_f, solve_triangular, solve_triangular_f, svd, svd_f, svdvals,
        svdvals_f, vector_norm, vector_norm_f,
    };
}

pub mod rstsr_structs {
    pub use crate::traits_def::{
        EighArgs, EighArgs_, EighArgs_Builder, EighResult, MatrixNormArgs, MatrixRankRtol, NormOrd, SLogDetResult,
        SVDArgs, SVDArgs_, SVDArgs_Builder, SVDResult, VectorNormArgs,
    };
}
