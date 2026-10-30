// SPDX-License-Identifier: AGPL-3.0-only
//! Numerics check for the cutile fused K-norm+RoPE kernel against an F64
//! reference of the CUDA kernel's math. Position is fixed at 0 in the kernel
//! for now, so the rotated value equals the normalized value and this only
//! proves the RMS and the pair split. The RoPE angle itself is checked once
//! per-token positions are wired.
//!
//!   cargo run -p spark-model --release --example k_norm_rope_cutile_check \
//!     --features cuda,gpu-examples,cutile
use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const HD: usize = 128;
const EPS: f32 = 1e-6;
const THETA: f32 = 1_000_000.0;

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &[])?;
    let g: &dyn GpuBackend = &gpu;
    let rows = 4usize;
    let mut x = vec![bf16::ZERO; rows * HD];
    let mut w = vec![bf16::ZERO; HD];
    for (i, v) in x.iter_mut().enumerate() {
        *v = bf16::from_f32(((i * 17 % 100) as f32 / 50.0) - 1.0);
    }
    for (i, v) in w.iter_mut().enumerate() {
        *v = bf16::from_f32((i as f32 / HD as f32) * 0.5);
    }
    let xp = upload(g, &x)?;
    let wp = upload(g, &w)?;
    let op = g.alloc(rows * HD * 2)?;
    spark_model::layers::ops::cutile_k_norm_rope::fused_k_norm_rope_cache_write_bf16_op(
        g,
        &spark_model::layers::ops::cutile_k_norm_rope::FusedKNormRopeOp::Cutile,
        spark_runtime::gpu::KernelHandle(0),
        xp, wp, xp, op, xp,
        rows as u32, 1, HD as u32, HD as u32, 16, EPS, THETA, 0,
    )?;
    g.synchronize(0)?;
    let got = download(g, op, rows * HD)?;

    let mut worst = 0.0f64;
    for r in 0..rows {
        let row = &x[r * HD..(r + 1) * HD];
        let mean: f64 = row.iter().map(|v| v.to_f64().powi(2)).sum::<f64>() / HD as f64;
        let rms = (mean + EPS as f64).sqrt().recip();
        for d in 0..HD {
            let normed = row[d].to_f64() * rms * (1.0 + w[d].to_f64());
            let expect = bf16::from_f32(normed as f32).to_f64();
            worst = worst.max((got[r * HD + d] - expect).abs());
        }
    }
    println!("rows={rows} hd={HD} pos=0 max|abs|={worst:.3e}");
    if worst > 0.02 {
        anyhow::bail!("K-norm+RoPE cutile check failed: {worst:.3e}");
    }
    println!("PASS");
    Ok(())
}

fn upload(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f64>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
        .collect())
}
