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
    if std::env::var("ATLAS_KNR_BENCH").as_deref() == Ok("1") {
        return bench();
    }
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
    let positions = [3i32, 10, 17, 100_000];
    let xp = upload(g, &x)?;
    let wp = upload(g, &w)?;
    let pp = upload_i32(g, &positions)?;
    let op = g.alloc(rows * HD * 2)?;
    spark_model::layers::ops::cutile_k_norm_rope::fused_k_norm_rope_cache_write_bf16_op(
        g,
        &spark_model::layers::ops::cutile_k_norm_rope::FusedKNormRopeOp::Cutile,
        spark_runtime::gpu::KernelHandle(0),
        xp, wp, pp, op, xp,
        rows as u32, 1, HD as u32, HD as u32, 16, EPS, THETA, 0,
    )?;
    g.synchronize(0)?;
    let got = download(g, op, rows * HD)?;

    let half = HD / 2;
    let mut worst = 0.0f64;
    let mut worst_ulp = 0u32;
    for r in 0..rows {
        let row = &x[r * HD..(r + 1) * HD];
        let xf: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
        let sum: f32 = xf.iter().map(|v| v * v).sum();
        let rms = (sum / HD as f32 + EPS).sqrt().recip();
        let n: Vec<f32> = xf.iter().zip(&w).map(|(v, ww)| v * rms * (1.0 + ww.to_f32())).collect();
        let pos = positions[r] as f32;
        for d in 0..half {
            let freq = (THETA as f64).powf(-2.0 * d as f64 / HD as f64) as f32;
            let ang = pos * freq;
            let (c, s) = (ang.cos(), ang.sin());
            for (col, y) in [(d, n[d] * c - n[d + half] * s), (d + half, n[d + half] * c + n[d] * s)] {
                let expect = bf16::from_f32(y);
                let got_b = bf16::from_f64(got[r * HD + col]);
                let ulp = (expect.to_bits() as i32 - got_b.to_bits() as i32).unsigned_abs();
                worst_ulp = worst_ulp.max(ulp);
                worst = worst.max((expect.to_f64() - got_b.to_f64()).abs());
            }
        }
    }
    println!(
        "rows={rows} hd={HD} pos={positions:?} max|abs|={worst:.3e} max_ulp={worst_ulp}"
    );
    if worst_ulp > 1 {
        anyhow::bail!("K-norm+RoPE cutile check failed: {worst_ulp} ulp");
    }
    println!("PASS");
    Ok(())
}

fn upload_i32(g: &dyn GpuBackend, d: &[i32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn upload(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn bench() -> Result<()> {
    use std::time::Instant;
    use spark_model::layers::ops::cutile_k_norm_rope::{
        fused_k_norm_rope_cache_write_bf16_op, FusedKNormRopeOp,
    };
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let k = g.kernel("fused_k_norm_rope_cache", "fused_k_norm_rope_cache_write_bf16")?;
    println!(
        "{:>8} {:>6} {:>12} {:>12} {:>8}",
        "tokens", "hd", "cuda us", "cutile us", "delta"
    );
    for tokens in [512u32, 8192] {
        let n = tokens as usize * HD;
        let x = vec![bf16::from_f32(0.5); n];
        let w = vec![bf16::from_f32(0.1); HD];
        let pos: Vec<i32> = (0..tokens).map(|t| t as i32).collect();
        let slots: Vec<i64> = (0..tokens as i64).collect();
        let xp = upload(g, &x)?;
        let wp = upload(g, &w)?;
        let pp = upload_i32(g, &pos)?;
        let sb: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        let sp = g.alloc(sb.len())?;
        g.copy_h2d(&sb, sp)?;
        let op = g.alloc(n * 2)?;
        let reps = 30u32;
        let once = |kind: &FusedKNormRopeOp| -> Result<()> {
            fused_k_norm_rope_cache_write_bf16_op(
                g, kind, k, xp, wp, pp, op, sp, tokens, 1, HD as u32, HD as u32, tokens, EPS,
                THETA, 0,
            )?;
            g.synchronize(0)
        };
        for _ in 0..3 {
            once(&FusedKNormRopeOp::Registry(k))?;
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            once(&FusedKNormRopeOp::Registry(k))?;
        }
        let cuda = t0.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
        for _ in 0..3 {
            once(&FusedKNormRopeOp::Cutile)?;
        }
        let t1 = Instant::now();
        for _ in 0..reps {
            once(&FusedKNormRopeOp::Cutile)?;
        }
        let cut = t1.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
        println!(
            "{tokens:>8} {HD:>6} {cuda:>12.2} {cut:>12.2} {:>+7.1}%",
            (cut - cuda) / cuda * 100.0
        );
        for p in [xp, wp, pp, sp, op] {
            g.free(p)?;
        }
    }
    Ok(())
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f64>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
        .collect())
}
