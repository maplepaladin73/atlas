// SPDX-License-Identifier: AGPL-3.0-only
//! Cutile Mamba-2 decode against the CUDA kernel and its f32 math.
//!
//!   ATLAS_MAMBA2_PTX=/path/mamba2_decode.ptx ATLAS_CUTILE_MAMBA2_DECODE=1 \
//!   ATLAS_CUTILE_MIN_TOKENS=1 cargo run -p spark-model --release \
//!   --example mamba2_decode_cutile_check --features cuda,gpu-examples,cutile
use anyhow::{Result, bail};
use half::bf16;
use spark_model::layers::ops::cutile_mamba2_decode::{Mamba2DecodeOp, mamba2_ssm_decode_op};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|v| v.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn launch(
    g: &dyn GpuBackend,
    op: &Mamba2DecodeOp,
    k: KernelHandle,
    h: DevicePtr,
    x: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    dt: DevicePtr,
    a: DevicePtr,
    d: DevicePtr,
    bias: DevicePtr,
    y: DevicePtr,
    batch: u32,
    heads: u32,
    hd: u32,
    ss: u32,
    groups: u32,
) -> Result<()> {
    mamba2_ssm_decode_op(
        g, op, k, h, x, b, c, dt, a, d, bias, y, batch, heads, hd, ss, groups, 1e-6, 1e3, 0,
    )
}

fn case(
    g: &dyn GpuBackend,
    k: KernelHandle,
    batch: usize,
    heads: usize,
    hd: usize,
    ss: usize,
    groups: usize,
    seed: u64,
) -> Result<()> {
    let mut r = Lcg(seed);
    let n = batch * heads;
    let rows = n * hd;
    let h: Vec<f32> = (0..rows * ss).map(|_| r.f() * 0.5).collect();
    let x: Vec<f32> = (0..rows).map(|_| r.f()).collect();
    let b: Vec<f32> = (0..batch * groups * ss).map(|_| r.f()).collect();
    let c: Vec<f32> = (0..batch * groups * ss).map(|_| r.f()).collect();
    let dt: Vec<f32> = (0..n).map(|_| r.f().abs()).collect();
    let a: Vec<f32> = (0..heads).map(|_| -1.0 + 0.1 * r.f()).collect();
    let d: Vec<f32> = (0..heads).map(|_| 0.5 + 0.1 * r.f()).collect();
    let bias: Vec<f32> = (0..heads).map(|_| 0.1 * r.f()).collect();
    let xp = up_bf16(g, &x)?;
    let bp = up_bf16(g, &b)?;
    let cp = up_bf16(g, &c)?;
    let dtp = up_bf16(g, &dt)?;
    let ap = up_f32(g, &a)?;
    let dp = up_f32(g, &d)?;
    let biasp = up_f32(g, &bias)?;
    let h_cu = up_f32(g, &h)?;
    let h_ct = up_f32(g, &h)?;
    let y_cu = g.alloc(rows * 2)?;
    let y_ct = g.alloc(rows * 2)?;
    launch(
        g,
        &Mamba2DecodeOp::Registry(k),
        k,
        h_cu,
        xp,
        bp,
        cp,
        dtp,
        ap,
        dp,
        biasp,
        y_cu,
        batch as u32,
        heads as u32,
        hd as u32,
        ss as u32,
        groups as u32,
    )?;
    launch(
        g,
        &Mamba2DecodeOp::Cutile,
        k,
        h_ct,
        xp,
        bp,
        cp,
        dtp,
        ap,
        dp,
        biasp,
        y_ct,
        batch as u32,
        heads as u32,
        hd as u32,
        ss as u32,
        groups as u32,
    )?;
    g.synchronize(0)?;
    let cu = down_bf16(g, y_cu, rows)?;
    let ct = down_bf16(g, y_ct, rows)?;
    let hu = down_f32(g, h_cu, h.len())?;
    let ht = down_f32(g, h_ct, h.len())?;
    let mut y_abs = 0.0f32;
    let mut y_ulp = 0u32;
    for (a, w) in ct.iter().zip(&cu) {
        y_abs = y_abs.max((a - w).abs());
        let ulp = (bf16::from_f32(*a).to_bits() as i32 - bf16::from_f32(*w).to_bits() as i32)
            .unsigned_abs();
        y_ulp = y_ulp.max(ulp);
    }
    let h_abs = ht
        .iter()
        .zip(&hu)
        .map(|(a, w)| (a - w).abs())
        .fold(0.0f32, f32::max);
    println!(
        "b{batch} h{heads} hd{hd} s{ss} g{groups}  y|abs|={y_abs:.3e} ulp={y_ulp}  H|abs|={h_abs:.3e}"
    );
    if y_ulp > 1 || h_abs > 1e-4 {
        bail!("numerics failed");
    }
    Ok(())
}

fn bench(g: &dyn GpuBackend, k: KernelHandle, batch: usize) -> Result<()> {
    let (heads, hd, ss, groups) = (8usize, 64usize, 128usize, 4usize);
    let rows = batch * heads * hd;
    let mut r = Lcg(3);
    let h = up_f32(g, &(0..rows * ss).map(|_| r.f()).collect::<Vec<_>>())?;
    let x = up_bf16(g, &(0..rows).map(|_| r.f()).collect::<Vec<_>>())?;
    let b = up_bf16(
        g,
        &(0..batch * groups * ss).map(|_| r.f()).collect::<Vec<_>>(),
    )?;
    let c = up_bf16(
        g,
        &(0..batch * groups * ss).map(|_| r.f()).collect::<Vec<_>>(),
    )?;
    let dt = up_bf16(
        g,
        &(0..batch * heads).map(|_| r.f().abs()).collect::<Vec<_>>(),
    )?;
    let a = up_f32(g, &(0..heads).map(|_| -1.0).collect::<Vec<_>>())?;
    let d = up_f32(g, &(0..heads).map(|_| 0.5).collect::<Vec<_>>())?;
    let bias = up_f32(g, &vec![0.0f32; heads])?;
    let y = g.alloc(rows * 2)?;
    let once = |op: &Mamba2DecodeOp| {
        launch(
            g,
            op,
            k,
            h,
            x,
            b,
            c,
            dt,
            a,
            d,
            bias,
            y,
            batch as u32,
            heads as u32,
            hd as u32,
            ss as u32,
            groups as u32,
        )
    };
    once(&Mamba2DecodeOp::Registry(k))?;
    once(&Mamba2DecodeOp::Cutile)?;
    let time = |op: Mamba2DecodeOp| -> Result<f64> {
        let t = Instant::now();
        for _ in 0..30 {
            once(&op)?;
            g.synchronize(0)?;
        }
        Ok(t.elapsed().as_secs_f64() * 1e6 / 30.0)
    };
    let cuda = time(Mamba2DecodeOp::Registry(k))?;
    let cutile = time(Mamba2DecodeOp::Cutile)?;
    println!(
        "bench b{batch}  cuda {cuda:.1} us  cutile {cutile:.1} us  {:+.1}%",
        (cutile - cuda) / cuda * 100.0
    );
    Ok(())
}

fn main() -> Result<()> {
    let path = std::env::var("ATLAS_MAMBA2_PTX")?;
    let ptx = std::fs::read(&path)?;
    let ptx: &'static [u8] = Box::leak(ptx.into_boxed_slice());
    let gpu = AtlasCudaBackend::new(0, &[("mamba2_ssm", ptx)])?;
    let g: &dyn GpuBackend = &gpu;
    let k = g.kernel("mamba2_ssm", "mamba2_ssm_decode")?;
    case(g, k, 2, 4, 16, 16, 2, 7)?;
    case(g, k, 8, 8, 64, 128, 4, 11)?;
    case(g, k, 16, 8, 64, 128, 4, 13)?;
    case(g, k, 64, 8, 64, 128, 4, 17)?;
    case(g, k, 8, 64, 64, 128, 8, 19)?;
    for b in [8usize, 64, 512] {
        bench(g, k, b)?;
    }
    println!("PASS");
    Ok(())
}
