// SPDX-License-Identifier: AGPL-3.0-only
//! Timing harness for cutile RMSNorm against `rms_norm_vanilla`.
//! Split out of `rmsnorm_vanilla_microtest.rs` to keep that file under
//! the 500 line cap.
//!
//!   ATLAS_NO_RDMA=1 cargo run -p spark-model --release \
//!     --example rmsnorm_cutile_bench --features cuda,gpu-examples,cutile
use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const EPS: f32 = 1e-6;

fn ub(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn launch(
    g: &dyn GpuBackend,
    x: DevicePtr,
    w: DevicePtr,
    o: DevicePtr,
    tokens: u32,
    hidden: u32,
) -> Result<()> {
    let k = g.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
    KernelLaunch::new(g, k)
        .grid([tokens, 1, 1])
        .block([hidden.min(1024), 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(o)
        .arg_u32(hidden)
        .arg_f32(EPS)
        .launch(0)?;
    g.synchronize(0)?;
    Ok(())
}

/// Wall time per launch, including the sync both paths already do.
fn bench_both(g: &dyn GpuBackend) -> Result<()> {
    #[cfg(feature = "cutile")]
    use spark_model::layers::ops::cutile_rms_norm::{RmsNormOp, rms_norm_op};
    use std::time::Instant;
    println!("=== bench — cutile vs rms_norm_vanilla (sync each launch) ===");
    println!(
        "{:>8} {:>8} {:>14} {:>14} {:>8}",
        "tokens", "hidden", "cuda us", "cutile us", "delta"
    );
    for (tokens, hidden) in [(1u32, 4096u32), (8192, 4096), (8192, 7168)] {
        let n = tokens as usize * hidden as usize;
        let x = vec![bf16::from_f32(0.5); n];
        let w = vec![bf16::from_f32(0.7); hidden as usize];
        let xp = ub(g, &x)?;
        let wp = ub(g, &w)?;
        let op = g.alloc(n * 2)?;
        let reps = 30u32;
        for _ in 0..5 {
            launch(g, xp, wp, op, tokens, hidden)?;
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            launch(g, xp, wp, op, tokens, hidden)?;
        }
        let cuda_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
        #[cfg(feature = "cutile")]
        let cutile_us = {
            let dw = spark_model::weight_map::DenseWeight { weight: wp };
            for _ in 0..3 {
                rms_norm_op(g, &RmsNormOp::Cutile, xp, &dw, op, tokens, hidden, EPS, 0)?;
                g.synchronize(0)?;
            }
            let t1 = Instant::now();
            for _ in 0..reps {
                rms_norm_op(g, &RmsNormOp::Cutile, xp, &dw, op, tokens, hidden, EPS, 0)?;
                g.synchronize(0)?;
            }
            t1.elapsed().as_secs_f64() * 1e6 / f64::from(reps)
        };
        #[cfg(not(feature = "cutile"))]
        let cutile_us = f64::NAN;
        let delta = (cutile_us - cuda_us) / cuda_us * 100.0;
        println!("{tokens:>8} {hidden:>8} {cuda_us:>14.2} {cutile_us:>14.2} {delta:>+7.1}%");
        for p in [xp, wp, op] {
            g.free(p)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    bench_both(&gpu)
}
