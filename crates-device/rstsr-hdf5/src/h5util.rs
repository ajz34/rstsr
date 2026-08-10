use crate::prelude_dev::*;

/// Verify the on-disk dtype matches the requested Rust type `T`, so that same-size but
/// different-kind reads (e.g. int64 dataset as float64) are rejected instead of silently
/// reinterpreting the raw bytes.
pub fn verify_dataset_type<T: H5Type>(dataset: &H5Dataset) -> Result<()> {
    let dtype = dataset
        .dtype()
        .map_err(|e| rstsr_error!(IOError, "Failed to read HDF5 datatype: {e}"))?
        .to_descriptor()
        .map_err(|e| rstsr_error!(IOError, "Failed to convert HDF5 datatype: {e}"))?;
    let expected_dtype = T::type_descriptor();
    if dtype != expected_dtype {
        return Err(rstsr_error!(IOError, "Dataset type mismatch: expected {expected_dtype:?}, got {dtype:?}"));
    }
    Ok(())
}
