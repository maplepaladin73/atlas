// SPDX-License-Identifier: AGPL-3.0-only
//! Numerics + timing check for the cutile `gated_rms_norm_f32_input_sigmoid`
//! port against the precompiled CUDA kernel (same inputs, same shapes).
//!
//!   cargo run -p spark-model --release --example gated_rms_norm_sigmoid_cutile_check \
//!     --features cuda,gpu-examples,cutile
//!   ATLAS_GRN_BENCH=1 cargo run -p spark-model --release --example gated_rms_norm_sigmoid_cutile_check \
//!     --features cuda,gpu-examples,cutile
use anyhow::Result;
use half::bf16;
use spark_model::layers::ops::cutile_gated_rms_norm::{GatedRmsNormOp, gated_rms_norm_op};
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const H: usize = 128; // value head dim used by the qwen3.8-flash GDN norm
const GS: usize = 128; // gate stride (== hidden here; a padded stride is covered by shape (rows, H, GS=256) below)
const EPS: f32 = 1e-6;

fn main() -> Result<()> {
    if std::env::var("ATLAS_GRN_BENCH").as_deref() == Ok("1") {
        return bench();
    }
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let k = g.kernel("norm", "gated_rms_norm_f32_input_sigmoid")?;

    let mut worst = 0.0f64;
    let mut worst_ulp = 0u32;
    for (rows, gs) in [(8usize, GS), (64, GS), (512, GS), (64, 256)] {
        let n = rows * H;
        // f32 input, bf16 gate/weight — matches gated_rms_norm_f32_input_sigmoid
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 97) as f32 / 48.5) - 1.0)
            .collect();
        let gate: Vec<bf16> = (0..rows * gs)
            .map(|i| bf16::from_f32(((i * 11 % 89) as f32 / 44.5) - 1.0))
            .collect();
        let w: Vec<bf16> = (0..H)
            .map(|i| bf16::from_f32(i as f32 / H as f32))
            .collect();

        let xp = upload_f32(g, &x)?;
        let gp = upload_bf16(g, &gate)?;
        let wp = upload_bf16(g, &w)?;
        let out_c = g.alloc(n * 2)?;
        let out_r = g.alloc(n * 2)?;
        let dw = DenseWeight { weight: wp };

        gated_rms_norm_op(
            g,
            &GatedRmsNormOp::Cutile,
            k,
            xp,
            gp,
            &dw,
            out_c,
            rows as u32,
            H as u32,
            gs as u32,
            EPS,
            0,
        )?;
        gated_rms_norm_op(
            g,
            &GatedRmsNormOp::Registry(k),
            k,
            xp,
            gp,
            &dw,
            out_r,
            rows as u32,
            H as u32,
            gs as u32,
            EPS,
            0,
        )?;
        g.synchronize(0)?;
        let got = download_bf16(g, out_c, n)?;
        let want = download_bf16(g, out_r, n)?;

        for i in 0..n {
            let e = bf16::from_f64(want[i]);
            let b = bf16::from_f64(got[i]);
            let ulp = (e.to_bits() as i32 - b.to_bits() as i32).unsigned_abs();
            worst_ulp = worst_ulp.max(ulp);
            worst = worst.max((e.to_f64() - b.to_f64()).abs());
        }
        // host reference of the CUDA math, as a second opinion
        for r in 0..rows {
            let xf = &x[r * H..(r + 1) * H];
            let sum: f32 = xf.iter().map(|v| v * v).sum();
            let rms = (sum / H as f32 + EPS).sqrt().recip();
            for d in 0..H {
                let gv = gate[r * gs + d].to_f32();
                let sig = 1.0 / (1.0 + (-gv).exp());
                let expect = bf16::from_f32(xf[d] * rms * w[d].to_f32() * sig);
                let b = bf16::from_f64(got[r * H + d]);
                let ulp = (expect.to_bits() as i32 - b.to_bits() as i32).unsigned_abs();
                if ulp > 1 {
                    anyhow::bail!(
                        "rows={rows} gs={gs} row={r} col={d}: {ulp} ulp vs f32 reference"
                    );
                }
            }
        }
        for p in [xp, gp, wp, out_c, out_r] {
            g.free(p)?;
        }
        println!("rows={rows:>4} gs={gs:>3} checked vs registry + f32 reference");
    }
    println!("max|abs| vs registry = {worst:.3e}  max_ulp = {worst_ulp}");
    if worst_ulp > 1 {
        anyhow::bail!("gated_rms_norm cutile check failed: {worst_ulp} ulp vs registry");
    }
    println!("PASS");
    Ok(())
}

fn upload_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn upload_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn download_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f64>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
        .collect())
}

fn bench() -> Result<()> {
    use std::time::Instant;
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let k = g.kernel("norm", "gated_rms_norm_f32_input_sigmoid")?;
    println!(
        "{:>8} {:>6} {:>12} {:>12} {:>8}",
        "tokens", "hd", "cuda us", "cutile us", "delta"
    );
    for tokens in [8u32, 64, 512, 8192] {
        let n = tokens as usize * H;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 97) as f32 / 48.5) - 1.0)
            .collect();
        let gate = vec![bf16::from_f32(0.3); tokens as usize * GS];
        let w = vec![bf16::from_f32(0.1); H];
        let xp = upload_f32(g, &x)?;
        let gp = upload_bf16(g, &gate)?;
        let wp = upload_bf16(g, &w)?;
        let op = g.alloc(n * 2)?;
        let dw = DenseWeight { weight: wp };
        let reps = 30u32;
        let once = |kind: &GatedRmsNormOp| -> Result<()> {
            gated_rms_norm_op(
                g, kind, k, xp, gp, &dw, op, tokens, H as u32, GS as u32, EPS, 0,
            )?;
            g.synchronize(0)
        };
        for _ in 0..3 {
            once(&GatedRmsNormOp::Registry(k))?;
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            once(&GatedRmsNormOp::Registry(k))?;
        }
        let cuda = t0.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
        for _ in 0..3 {
            once(&GatedRmsNormOp::Cutile)?;
        }
        let t1 = Instant::now();
        for _ in 0..reps {
            once(&GatedRmsNormOp::Cutile)?;
        }
        let cut = t1.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
        println!(
            "{tokens:>8} {H:>6} {cuda:>12.2} {cut:>12.2} {:>+7.1}%",
            (cut - cuda) / cuda * 100.0
        );
        for p in [xp, gp, wp, op] {
            g.free(p)?;
        }
    }
    Ok(())
}
