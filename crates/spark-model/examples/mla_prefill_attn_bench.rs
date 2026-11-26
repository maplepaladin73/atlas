// SPDX-License-Identifier: AGPL-3.0-only
//! Wall-clock benchmark of the Ling-3.2-air prefill attention lane
//! (ling_mla_prefill_attn scalar vs ling_mla_prefill_attn_mma) at the
//! production chunk shape.
//!
//! Attention runs inside the packed prefill: the chunk's `seq_len` query
//! rows against the context rows [0, kv_base + seq_len), causal inside the
//! chunk. Q: [batch, seq_len, heads, 576] (bf16), K: [batch, kv_run, 576],
//! V: [batch, kv_run, 512] (bf16), O: [batch, seq_len, heads, 512].
//!
//! Times the kernel with CUDA events (median of reps) for the scalar and
//! mma lanes on the same synthetic data.
//!
//! Run (dev box):
//!   PATH=/home/spark/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
//!   CUDARC_CUDA_VERSION=13000 ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=ling-3.2-air \
//!   LIBRARY_PATH=/home/spark/nccl-lib LD_LIBRARY_PATH=/home/spark/nccl-lib \
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!     --example mla_prefill_attn_bench -- [seq_len] [reps]

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::{AtlasCudaBackend, configure_pdl};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuDeviceGetAttribute(value: *mut i32, attr: i32, dev: i32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const HEADS: u32 = 32;
const KV_DIM: u32 = 576; // kv_lora 512 | k rope raw 64
const LAT: u32 = 512;   // latent dims per head

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seq_len: u32 = args.first().map(|s| s.parse()).unwrap_or(Ok(4096u32))?;
    let reps: usize = args.get(1).map(|s| s.parse()).unwrap_or(Ok(30usize))?;
    let kv_base: u32 = 0;
    let causal: u32 = 1;
    let inv_sqrt_d: f32 = 1.0 / (192.0f32).sqrt();

    configure_pdl("ling-3.2-air");
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let h_scalar = g.kernel("gated_mla", "ling_mla_prefill_attn")?;
    let h_mma = g.kernel("gated_mla", "ling_mla_prefill_attn_mma")?;
    let stream = g.create_stream()?;

    let kv_run = kv_base + seq_len;

    // Deterministic host data (small LCG; the values only feed kernels'
    // timing, not a correctness contract).
    let mut rng = 0x9E3779B9u64;
    let mut next = || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let z = rng ^ (rng >> 33);
        (z & 0xffff_ffff) as u32
    };

    let mut q = vec![0u8; (seq_len as usize) * (HEADS as usize) * (KV_DIM as usize) * 2];
    for b in q.iter_mut() {
        *b = (next() & 0xff) as u8;
    }
    let mut k = vec![0u8; (kv_run as usize) * (KV_DIM as usize) * 2];
    for b in k.iter_mut() {
        *b = (next() & 0xff) as u8;
    }
    let mut v = vec![0u8; (kv_run as usize) * (LAT as usize) * 2];
    for b in v.iter_mut() {
        *b = (next() & 0xff) as u8;
    }

    let q_p = gpu_alloc(g, &q)?;
    let k_p = gpu_alloc(g, &k)?;
    let v_p = gpu_alloc(g, &v)?;
    let o_p = g.alloc((seq_len as usize) * (HEADS as usize) * (LAT as usize) * 2)?;

    let grid = [HEADS, div_ceil(seq_len, 16), 1];

    let launch = |h: KernelHandle, block: u32| -> Result<()> {
        KernelLaunch::new(g, h)
            .grid(grid)
            .block([block, 1, 1])
            .arg_ptr(q_p)
            .arg_ptr(k_p)
            .arg_ptr(v_p)
            .arg_ptr(o_p)
            .arg_u32(seq_len)
            .arg_u32(causal)
            .arg_f32(inv_sqrt_d)
            .arg_u32(kv_base)
            .launch(stream)
    };

    let mut out: Vec<String> = Vec::new();
    for (name, h, block) in [("scalar", h_scalar, 256u32), ("mma", h_mma, 128u32)] {
        launch(h, block)?;
        g.synchronize(stream)?;

        let mut ev = vec![0u64; 2 * reps];
        for e in ev.iter_mut() {
            if unsafe { cuEventCreate(e, 0) } != 0 {
                bail!("cuEventCreate");
            }
        }
        for i in 0..reps {
            if unsafe { cuEventRecord(ev[2 * i], stream) } != 0 {
                bail!("cuEventRecord");
            }
            launch(h, block)?;
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
        out.push(format!(
            "{name}: median {median:.1} us   min {min:.1} us   max {max:.1} us  (grid {grid:?})"
        ));
    }
    if std::env::var("MMA_BENCH_PROPS").as_deref() == Ok("1") {
        unsafe extern "C" { fn cuFuncGetAttribute(value: *mut i32, attr: i32, func: u64) -> i32; }
        for (name, h) in [("scalar", h_scalar.0), ("mma", h_mma.0)] {
            let mut v: i32 = -1;
            let r = unsafe { cuFuncGetAttribute(&mut v, 1, h) };
            println!("{} static smem: {} bytes (status={})", name, v, r);
        }

        let attrs: [(&str, i32); 5] = [
            ("maxSharedMemoryPerMultiprocessor", 81),
            ("maxSharedMemoryPerBlock", 8),
            ("maxSharedMemoryPerBlockOptin", 97),
            ("multiProcessorCount", 16),
            ("maxThreadsPerMultiprocessor", 39),
        ];
        for (name, attr) in attrs {
            let mut v: i32 = 0;
            let r = unsafe { cuDeviceGetAttribute(&mut v, attr, 0) };
            println!("{name}: {v}  (status={r})");
        }
    }
    println!("=== mla_prefill_attn_bench: seq_len={seq_len} heads={HEADS} causal={causal} kv_base={kv_base} reps={reps} ===");
    for line in &out {
        println!("{line}");
    }

    Ok(())
}

fn gpu_alloc(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}
