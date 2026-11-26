// SPDX-License-Identifier: AGPL-3.0-only
//! Wall-clock benchmark of the Ling-3.2-air route kernel (ling_moe_route:
//! fp32 router GEMV + NVFP4 input quant + group top-8 selection + counting
//! sort + worklist build) at the prefill chunk shape the serve path runs
//! (M = rows per sub-chunk, default 512).
//!
//! Times the kernel with CUDA events (median of reps) and prints a content
//! checksum of the routing outputs so a before/after kernel diff can be
//! checked without running the full parity harness.
//!
//! Exit: 0 ok.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=ling-3.2-air \
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example moe_route_bench -- [M] [reps]

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::{AtlasCudaBackend, configure_pdl};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const HIDDEN: usize = 2560;
const EXPERTS: usize = 512;
const TOP_K: usize = 8;
const N_GROUP: usize = 8;
const TOPK_GROUP: usize = 4;
const SCALING: f32 = 2.5;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let m: u32 = args.first().map(|s| s.parse()).unwrap_or(Ok(512u32))?;
    let reps: usize = args.get(1).map(|s| s.parse()).unwrap_or(Ok(30usize))?;
    configure_pdl("ling-3.2-air");
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let h = g.kernel("moe_decode", "ling_moe_route")?;
    let stream = g.create_stream()?;

    let k = HIDDEN as u32;
    let e = EXPERTS as u32;
    let tk = TOP_K as u32;

    // Deterministic host data (small LCG; the values only feed the kernel's
    // timing, not a correctness contract).
    let mut rng = 0x2545F491u64;
    let mut next = || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let z = rng ^ (rng >> 33);
        (z & 0xffff_ffff) as u32
    };
    let mut a_bytes = Vec::with_capacity(m as usize * HIDDEN * 2);
    for _ in 0..(m as usize) * HIDDEN {
        let v = ((next() % 1000) as f32 / 1000.0 - 0.5) * 4.0;
        let bf = (v.to_bits() >> 16) as u16;
        a_bytes.extend_from_slice(&bf.to_le_bytes());
    }
    let mut w_bytes = Vec::with_capacity(e as usize * HIDDEN * 4);
    for _ in 0..(e as usize) * HIDDEN {
        let v = ((next() % 1000) as f32 / 1000.0 - 0.5) * 0.5;
        w_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut bias_bytes = Vec::with_capacity(e as usize * 4);
    for _ in 0..e as usize {
        let v = ((next() % 1000) as f32 / 1000.0 - 0.5) * 0.1;
        bias_bytes.extend_from_slice(&v.to_le_bytes());
    }

    let a = gpu_alloc(g, &a_bytes)?;
    let gate_w = gpu_alloc(g, &w_bytes)?;
    let bias = gpu_alloc(g, &bias_bytes)?;

    let logits = g.alloc((m as usize) * (e as usize) * 4)?;
    let a_packed = g.alloc((m as usize) * ((k / 2) as usize))?;
    let a_scale = g.alloc((m as usize) * ((k / 16) as usize))?;
    let topk_ids = g.alloc((m as usize) * (tk as usize) * 4)?;
    let topk_weights = g.alloc((m as usize) * (tk as usize) * 4)?;
    let sorted_ids = g.alloc((m as usize) * (tk as usize) * 4)?;
    let token_to_perm = g.alloc((m as usize) * (tk as usize) * 4)?;
    let expert_offsets = g.alloc((e as usize + 1) * 4)?;
    let max_items = EXPERTS + m as usize * TOP_K / 16;
    let worklist = g.alloc((1 + 2 * max_items) * 4)?;
    let done = g.alloc(4)?;

    let router_ctas = div_ceil(e, 8);
    let quant_ctas = div_ceil(m * (k / 16), 256);
    let grid = router_ctas + quant_ctas;

    let launch = |g: &dyn GpuBackend| -> Result<()> {
        g.memset_async(done, 0, 4, stream)?;
        KernelLaunch::new(g, h)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(gate_w)
            .arg_ptr(bias)
            .arg_f32(1.0 / 2.0)
            .arg_f32(2.0)
            .arg_ptr(logits)
            .arg_ptr(a_packed)
            .arg_ptr(a_scale)
            .arg_ptr(topk_ids)
            .arg_ptr(topk_weights)
            .arg_ptr(sorted_ids)
            .arg_ptr(token_to_perm)
            .arg_ptr(expert_offsets)
            .arg_ptr(worklist)
            .arg_u32(max_items as u32)
            .arg_u32(16)
            .arg_u32(m)
            .arg_u32(e)
            .arg_u32(k)
            .arg_u32(tk)
            .arg_u32(N_GROUP as u32)
            .arg_u32(TOPK_GROUP as u32)
            .arg_f32(SCALING)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(done)
            .launch(stream)
    };

    // Warm-up (the checksum source; `launch` zeroes `done` before every rep).
    launch(g)?;
    g.synchronize(stream)?;

    let mut ev = vec![0u64; 2 * reps];
    for e0 in ev.iter_mut() {
        if unsafe { cuEventCreate(e0, 0) } != 0 {
            bail!("cuEventCreate");
        }
    }
    for i in 0..reps {
        if unsafe { cuEventRecord(ev[2 * i], stream) } != 0 {
            bail!("cuEventRecord");
        }
        launch(g)?;
        if unsafe { cuEventRecord(ev[2 * i + 1], stream) } != 0 {
            bail!("cuEventRecord");
        }
    }
    if unsafe { cuEventSynchronize(ev[2 * reps - 1]) } != 0 {
        bail!("cuEventSynchronize");
    }
    let mut us = Vec::with_capacity(reps);
    for i in 0..reps {
        let mut ms = 0f32;
        if unsafe { cuEventElapsedTime(&mut ms, ev[2 * i], ev[2 * i + 1]) } != 0 {
            bail!("cuEventElapsedTime");
        }
        us.push(ms * 1000.0);
        unsafe { cuEventDestroy_v2(ev[2 * i]) };
        unsafe { cuEventDestroy_v2(ev[2 * i + 1]) };
    }
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = us[reps / 2];
    let min = us[0];
    let max = us[reps - 1];

    // Checksum of the routing outputs (ids, weights, sorted ids, perm,
    // offsets, worklist) so two builds can be compared head-to-head.
    let digest = routing_checksum(
        g,
        m,
        tk,
        topk_ids,
        topk_weights,
        sorted_ids,
        token_to_perm,
        expert_offsets,
        worklist,
    )?;

    println!(
        "=== moe_route_bench: M={m} E={e} K={k} top_k={tk} n_group={N_GROUP} topk_group={TOPK_GROUP} reps={reps} ==="
    );
    println!("median: {median:.1} us   min: {min:.1} us   max: {max:.1} us");
    println!("checksum: {digest:#018x}");

    Ok(())
}

fn gpu_alloc(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn routing_checksum(
    g: &dyn GpuBackend,
    m: u32,
    tk: u32,
    topk_ids: DevicePtr,
    topk_weights: DevicePtr,
    sorted_ids: DevicePtr,
    token_to_perm: DevicePtr,
    expert_offsets: DevicePtr,
    worklist: DevicePtr,
) -> Result<u64> {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    let mut slot_buf = vec![0u8; (m * tk) as usize * 4];
    for p in [topk_ids, topk_weights, sorted_ids, token_to_perm] {
        g.copy_d2h(p, &mut slot_buf)?;
        mix(&slot_buf);
    }
    let mut offsets = vec![0u8; (512usize + 1) * 4];
    g.copy_d2h(expert_offsets, &mut offsets)?;
    mix(&offsets);
    let max_items = EXPERTS + m as usize * TOP_K / 16;
    let mut wl = vec![0u8; (1 + 2 * max_items) * 4];
    g.copy_d2h(worklist, &mut wl)?;
    mix(&wl);
    Ok(h)
}
