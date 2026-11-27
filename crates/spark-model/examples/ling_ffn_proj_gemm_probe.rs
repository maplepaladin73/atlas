// SPDX-License-Identifier: AGPL-3.0-only
//! Exact Ling FFN projection shapes through the split-K-free zero-workspace
//! cuBLASLt lane versus the scalar production GEMM fallback. Fails if the
//! lane is unavailable or numerically divergent at a tested shape.

use anyhow::{Result, bail};
use spark_model::layers::ops;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

fn bf16s(seed: &mut u64, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            (((f * scale).to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn bf16f(b: &[u8], i: usize) -> f32 {
    f32::from_bits((u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32) << 16)
}

fn compare(a: &[u8], b: &[u8]) -> (usize, f64, f32) {
    let bytes = a.iter().zip(b).filter(|(x, y)| x != y).count();
    let n = a.len() / 2;
    let (mut dot, mut aa, mut bb, mut maxd) = (0f64, 0f64, 0f64, 0f32);
    for i in 0..n {
        let (x, y) = (bf16f(a, i), bf16f(b, i));
        dot += f64::from(x * y);
        aa += f64::from(x * x);
        bb += f64::from(y * y);
        maxd = maxd.max((x - y).abs());
    }
    (bytes, dot / (aa.sqrt() * bb.sqrt()).max(1e-30), maxd)
}

fn time_ms(g: &dyn GpuBackend, stream: u64, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    g.synchronize(stream)?;
    let t0 = std::time::Instant::now();
    for _ in 0..5 {
        f()?;
    }
    g.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / 5.0)
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let gemm_k = g.kernel("gemm", "dense_gemm_bf16")?;
    let mut seed = 0x51a7u64;
    for (name, n, k) in [
        ("dense_gate_up", 6144usize, 2560usize),
        ("dense_down", 2560, 6144),
        ("shared_gate_up", 768, 2560),
        ("shared_down", 2560, 768),
    ] {
        let w = g.alloc(n * k * 2)?;
        g.copy_h2d(&bf16s(&mut seed, n * k, 0.02), w)?;
        let weight = DenseWeight { weight: w };
        for m in [16usize, 64, 511, 512, 1024] {
            let a = g.alloc(m * k * 2)?;
            g.copy_h2d(&bf16s(&mut seed, m * k, 1.0), a)?;
            let scalar = g.alloc(m * n * 2)?;
            let chain = g.alloc(m * n * 2)?;
            ops::dense_gemm(
                g, gemm_k, a, &weight, scalar, m as u32, n as u32, k as u32, stream,
            )?;
            let ran = spark_runtime::cublaslt::bf16_gemm_act_weight_t_in_order_no_workspace(
                a.0,
                w.0,
                chain.0,
                [m as u32, n as u32, k as u32],
                stream,
            )?;
            g.synchronize(stream)?;
            let (mut vs, mut vc) = (vec![0u8; m * n * 2], vec![0u8; m * n * 2]);
            g.copy_d2h(scalar, &mut vs)?;
            g.copy_d2h(chain, &mut vc)?;
            let (bytes, cosine, maxd) = compare(&vs, &vc);
            if !ran || cosine < 0.999 {
                bail!("{name} M={m}: lane ran={ran} cosine={cosine}");
            }
            let scalar_ms = time_ms(g, stream, || {
                ops::dense_gemm(
                    g, gemm_k, a, &weight, scalar, m as u32, n as u32, k as u32, stream,
                )
            })?;
            let lt_ms = if ran {
                time_ms(g, stream, || {
                    spark_runtime::cublaslt::bf16_gemm_act_weight_t_in_order_no_workspace(
                        a.0,
                        w.0,
                        chain.0,
                        [m as u32, n as u32, k as u32],
                        stream,
                    )
                    .map(|_| ())
                })?
            } else {
                f64::NAN
            };
            println!(
                "{name} M={m} N={n} K={k}: ran={ran} byte_diff={bytes} cosine={cosine:.9} max_abs={maxd:.6} scalar_ms={scalar_ms:.3} lt_ms={lt_ms:.3}"
            );
            for p in [a, scalar, chain] {
                g.free(p)?;
            }
        }
        g.free(w)?;
    }
    Ok(())
}
