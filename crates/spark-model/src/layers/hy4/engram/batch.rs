// SPDX-License-Identifier: AGPL-3.0-only

use super::{EngramScratch, Hy4Engram};
use crate::layers::hy4::{attention::memory::Owned, batch};
use anyhow::{Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

pub(super) struct BatchScratch<'a> {
    _memory: Owned<'a>,
    tokens: DevicePtr,
    hashes: DevicePtr,
    hashes_verify: DevicePtr,
    lookup: DevicePtr,
    keys: DevicePtr,
    norm_keys: DevicePtr,
    queries: DevicePtr,
    value: DevicePtr,
}
impl<'a> BatchScratch<'a> {
    fn new(e: &Hy4Engram<'a>) -> Result<Self> {
        let mut m = Owned::new(e.mem.gpu);
        let rows = batch::prefill_rows();
        Ok(Self {
            tokens: m.alloc(rows * 3 * 4)?,
            hashes: m.alloc(rows * e.layout.sizes.len() * 4)?,
            hashes_verify: m.alloc(rows * e.layout.sizes.len() * 4)?,
            lookup: m.alloc(rows * e.heads.len() * e.dim * 2)?,
            keys: m.alloc(rows * 4 * e.h * 2)?,
            norm_keys: m.alloc(rows * 4 * e.h * 2)?,
            queries: m.alloc(rows * 4 * e.h * 2)?,
            value: m.alloc(rows * e.h * 2)?,
            _memory: m,
        })
    }
}

impl<'a> Hy4Engram<'a> {
    pub fn forward_batch(
        &self,
        hidden: DevicePtr,
        history: &[u32],
        start: usize,
        scratch: &mut EngramScratch<'a>,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let rows = history
            .len()
            .checked_sub(start)
            .ok_or_else(|| anyhow::anyhow!("Hy4 Engram batch start"))?;
        ensure!(
            rows > 0
                && rows <= batch::prefill_rows()
                && !hidden.is_null()
                && self.mem.is_live()
                && scratch.mem.is_live()
                && scratch.signature == [self.h, self.heads.start, self.heads.end],
            "Hy4 Engram batch geometry"
        );
        if self.topology[1] > 1 {
            ensure!(
                comm.is_some_and(|c| [c.rank(), c.world_size()] == self.topology),
                "Hy4 Engram batch communicator"
            );
        }
        let b = match scratch.batch.take() {
            Some(b) => b,
            None => BatchScratch::new(self)?,
        };
        let result = self.batch_rows(hidden, history, start, &b, comm, stream);
        scratch.batch = Some(b);
        result
    }

    fn batch_rows(
        &self,
        hidden: DevicePtr,
        history: &[u32],
        start: usize,
        b: &BatchScratch<'_>,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let gpu = self.mem.gpu;
        let rows = history.len() - start;
        let heads = self.layout.sizes.len();
        let local = self.heads.len() * self.dim;
        {
            let _lookup = super::super::timing::span("engram_lookup", stream)?;
            let rowcache_guard = if self.rowcache_slots > 0 {
                let mut matrix = vec![-1i32; rows * heads];
                for r in 0..rows {
                    let p = start + r;
                    let mut context = Vec::with_capacity(2);
                    if p >= 2 {
                        context.push(history[p - 2]);
                    }
                    if p >= 1 {
                        context.push(history[p - 1]);
                    }
                    let row_hashes = self.host_hashes(&context, history[p])?;
                    matrix[r * heads..(r + 1) * heads].copy_from_slice(&row_hashes);
                }
                let preslot = verify_hashes().then(|| matrix.clone());
                let guard = self.resolve_hash_matrix(&mut matrix)?;
                let bytes: Vec<u8> = matrix.iter().flat_map(|v| v.to_le_bytes()).collect();
                gpu.copy_h2d_async(&bytes, b.hashes, stream)?;
                if let Some(preslot) = preslot {
                    verify_kernel_hashes(
                        self, history, start, rows, heads, &preslot, b, gpu, stream,
                    )?;
                }
                guard
            } else {
                // One H2D of the (token, previous, older) window plus one
                // hash launch for the whole chunk, replacing per-row launches.
                let mut window = Vec::with_capacity(rows * 3);
                for r in 0..rows {
                    let p = start + r;
                    window.push(history[p] as i32);
                    window.push(p.checked_sub(1).map_or(-1, |i| history[i] as i32));
                    window.push(p.checked_sub(2).map_or(-1, |i| history[i] as i32));
                }
                gpu.copy_h2d_async(
                    &window
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                    b.tokens,
                    stream,
                )?;
                KernelLaunch::new(gpu, self.kernels[5])
                    .grid([rows as u32, 1, 1])
                    .block([32, 1, 1])
                    .arg_ptr(b.tokens)
                    .arg_ptr(self.map)
                    .arg_ptr(self.multipliers)
                    .arg_ptr(self.sizes)
                    .arg_ptr(self.offsets)
                    .arg_ptr(b.hashes)
                    .arg_u32(self.layout.map.len() as u32)
                    .arg_u32(heads as u32)
                    .arg_u32(self.layout.heads_per_order as u32)
                    .arg_u32(self.layout.pad)
                    .arg_u32(rows as u32)
                    .launch(stream)?;
                None
            };
            let lookup = KernelLaunch::new(gpu, self.kernels[6])
                .grid([self.heads.len() as u32, rows as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(b.hashes)
                .arg_ptr(self.table);
            let lookup = if self.fp8_table {
                lookup.arg_ptr(self.scales)
            } else {
                lookup
            };
            let (base_row, local_rows) = self.lookup_bounds();
            lookup
                .arg_ptr(self.lookup_norm)
                .arg_ptr(b.lookup)
                .arg_u32(self.heads.start as u32)
                .arg_u32(base_row)
                .arg_u32(local_rows)
                .arg_u32(self.dim as u32)
                .arg_u32(heads as u32)
                .arg_u32(local as u32)
                .arg_u32(rows as u32)
                .launch(stream)?;
            drop(rowcache_guard);
        }
        // Tensor-core path for the projections (same GEMM as the scalar
        // kernel: bias iff topology[0]==0, weight column window starting at
        // heads.start*dim) when the shared linear-mma flag is on.
        let mma = std::env::var("ATLAS_HY4_LINEAR_MMA").ok().as_deref() == Some("1")
            && local.is_multiple_of(32)
            && rows >= 32;
        for i in 0..5 {
            let _project = super::super::timing::span("engram_project", stream)?;
            let (out, stride) = if i == 4 {
                (b.value, self.h)
            } else {
                (b.keys.offset(i * self.h * 2), 4 * self.h)
            };
            if mma {
                batch::gemm(
                    gpu,
                    [
                        b.lookup,
                        self.projections[i].offset(self.heads.start * self.dim * 2),
                        if self.topology[0] == 0 {
                            self.biases[i]
                        } else {
                            DevicePtr::NULL
                        },
                        out,
                    ],
                    [
                        self.h,
                        local,
                        rows,
                        1,
                        0,
                        0,
                        local,
                        stride,
                        heads * self.dim,
                    ],
                    [0, 0],
                    stream,
                )?;
                continue;
            }
            KernelLaunch::new(
                gpu,
                gpu.op_cache()
                    .kernel(gpu, "hy4_engram", "hy4_engram_batch_project")?,
            )
            .grid([self.h.div_ceil(128) as u32, rows.div_ceil(8) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(b.lookup)
            .arg_ptr(self.projections[i])
            .arg_ptr(self.biases[i])
            .arg_ptr(out)
            .arg_u32(self.h as u32)
            .arg_u32((heads * self.dim) as u32)
            .arg_u32((self.heads.start * self.dim) as u32)
            .arg_u32(local as u32)
            .arg_u32(u32::from(self.topology[0] == 0))
            .arg_u32(rows as u32)
            .arg_u32(stride as u32)
            .launch(stream)?;
        }
        if let Some(comm) = comm.filter(|_| self.topology[1] > 1) {
            let _reduce = super::super::timing::span("all_reduce", stream)?;
            let bytes = rows * self.h * 2;
            for part in 0..4 {
                comm.all_reduce_async(b.keys.offset(part * bytes).0, bytes, stream)?;
            }
            comm.all_reduce_async(b.value.0, rows * self.h * 2, stream)?;
        }
        {
            let _finish = super::super::timing::span("engram_norm_gate", stream)?;
            for (input, weight, out) in [
                (b.keys, self.key_norm, b.norm_keys),
                (hidden, self.query_norm, b.queries),
            ] {
                KernelLaunch::new(gpu, self.kernels[7])
                    .grid([4, rows as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(input)
                    .arg_ptr(weight)
                    .arg_ptr(out)
                    .arg_u32(self.h as u32)
                    .arg_u32((4 * self.h) as u32)
                    .arg_u32(rows as u32)
                    .launch(stream)?;
            }
            KernelLaunch::new(gpu, self.kernels[8])
                .grid([4, rows as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(hidden)
                .arg_ptr(b.norm_keys)
                .arg_ptr(b.queries)
                .arg_ptr(b.value)
                .arg_ptr(self.gamma)
                .arg_ptr(self.hc_scales)
                .arg_ptr(b.hashes)
                .arg_u32(self.h as u32)
                .arg_u32((4 * self.h) as u32)
                .arg_u32(heads as u32)
                .arg_u32(rows as u32)
                .launch(stream)?;
        }
        Ok(())
    }
}

/// `ATLAS_HY4_ENGRAM_ROW_CACHE_VERIFY=1`: in the exact-row cache path, ALSO
/// run the GPU hash kernel and bit-compare its absolute row ids against the
/// host-computed pre-resolve matrix. Opt-in debug validation; costs one
/// synchronizing copyback per layer-chunk.
fn verify_hashes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_HY4_ENGRAM_ROW_CACHE_VERIFY")
            .ok()
            .as_deref()
            == Some("1")
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_kernel_hashes(
    e: &Hy4Engram<'_>,
    history: &[u32],
    start: usize,
    rows: usize,
    heads: usize,
    preslot: &[i32],
    b: &BatchScratch<'_>,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    stream: u64,
) -> Result<()> {
    let mut window = Vec::with_capacity(rows * 3);
    for r in 0..rows {
        let p = start + r;
        window.push(history[p] as i32);
        window.push(p.checked_sub(1).map_or(-1, |i| history[i] as i32));
        window.push(p.checked_sub(2).map_or(-1, |i| history[i] as i32));
    }
    gpu.copy_h2d_async(
        &window
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        b.tokens,
        stream,
    )?;
    KernelLaunch::new(gpu, e.kernels[5])
        .grid([rows as u32, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(b.tokens)
        .arg_ptr(e.map)
        .arg_ptr(e.multipliers)
        .arg_ptr(e.sizes)
        .arg_ptr(e.offsets)
        .arg_ptr(b.hashes_verify)
        .arg_u32(e.layout.map.len() as u32)
        .arg_u32(heads as u32)
        .arg_u32(e.layout.heads_per_order as u32)
        .arg_u32(e.layout.pad)
        .arg_u32(rows as u32)
        .launch(stream)?;
    let mut kernel_out = vec![0u8; rows * heads * 4];
    gpu.copy_d2h(b.hashes_verify, &mut kernel_out)?;
    let mut mismatches = 0usize;
    let mut first = None;
    for (i, b4) in kernel_out.chunks_exact(4).enumerate() {
        let k = i32::from_le_bytes(b4.try_into().unwrap());
        if k != preslot[i] {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, preslot[i], k));
            }
        }
    }
    if mismatches > 0 {
        let (i, host, kern) = first.unwrap();
        tracing::error!(
            "rowcache hash verify: {mismatches}/{} MISMATCHES, first at flat {i} (row {} head {}): host {host} vs kernel {kern}",
            rows * heads,
            i / heads,
            i % heads
        );
    } else {
        tracing::info!(
            "rowcache hash verify: 0/{} mismatches (rows={rows} heads={heads})",
            rows * heads
        );
    }
    Ok(())
}
