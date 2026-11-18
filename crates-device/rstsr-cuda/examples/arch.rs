//! Print this card's compute capability as bare digits (`86`), for scripts that need to know which
//! archs it can host.
//!
//! Why this exists: `nvidia-smi --query-gpu=compute_cap` is not available on every driver, and the
//! alternative — mapping a product name to an arch — is a guess. This asks the same query the launch
//! path uses (`DeviceCuda::compute_capability`), so the number cannot disagree with what the kernels
//! actually compile for.
//!
//! It reports the **device**, not `RSTSR_CUDA_ARCH`: the point is the ceiling, and an override can
//! only lower the target.
//!
//! It also has to behave on a machine with no driver, because that is where it will most often be
//! run: `cudarc` *panics* when the driver library is missing instead of reporting zero devices, so
//! the probe is wrapped and the panic hook silenced — one line of diagnosis beats a dlopen list.

use rstsr_cuda::device::DeviceCuda;

fn main() {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let probe = std::panic::catch_unwind(|| {
        let device = DeviceCuda::new_f(0).map_err(|e| e.to_string())?;
        device.compute_capability().map_err(|e| e.to_string())
    });
    std::panic::set_hook(hook);

    match probe {
        Ok(Ok((maj, min))) => println!("{maj}{min}"),
        Ok(Err(e)) => {
            eprintln!("arch: {e}");
            std::process::exit(1);
        },
        Err(_) => {
            eprintln!("arch: no CUDA driver visible");
            std::process::exit(1);
        },
    }
}
