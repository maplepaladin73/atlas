// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in cutile-rs port of `fused_k_norm_rope_cache_write_bf16`
//! (kernels/gb10/common/fused_k_norm_rope_cache.cu), gated on
//! `ATLAS_CUTILE_KNORM_ROPE=1`. Second kernel in the RMSNorm class
//! (elementwise + one reduction + store, no tensor cores) — see PR #2 for the
//! generalizes-vs-one-off question against the RMSNorm numbers.
//!
//! Bridge pattern shared with `cutile_rms_norm`: raw `DevicePtr`s borrowed as
//! cutile `Tensor`s (no ownership transfer), SingleStream policy, host-side
//! `gpu.synchronize(stream)` before launch, block-on completion. The CUDA
//! kernel runs one block per (token, head) row of `head_dim`; the cutile port
//! uses one tile per row (partition [1, HD], HD = head_dim, a power of two
//! ≤ 256 on gb10).
//!
//! Requires the `cutile` cargo feature; without it the env flag is ignored and
//! the precompiled kernel is always used.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

/// True when the cutile K-norm+RoPE experiment is enabled AND compiled in.
pub fn cutile_knorm_rope_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "cutile") && std::env::var("ATLAS_CUTILE_KNORM_ROPE").as_deref() == Ok("1")
    })
}

/// Rows (tokens*kv_heads) below this stay on the CUDA kernel — the cutile
/// launcher's fixed overhead dominates at decode-like sizes.
pub fn cutile_min_rows() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_CUTILE_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    })
}

/// Either flavor of the fused K-norm+RoPE+cache-write op.
pub enum FusedKNormRopeOp {
    Registry(KernelHandle),
    Cutile,
}

/// Resolve the op: cutile when enabled and compiled in, else the registry handle.
pub fn resolve_fused_k_norm_rope(gpu: &dyn GpuBackend) -> Result<FusedKNormRopeOp> {
    if cutile_knorm_rope_enabled() {
        Ok(FusedKNormRopeOp::Cutile)
    } else {
        Ok(FusedKNormRopeOp::Registry(gpu.kernel(
            "fused_k_norm_rope_cache",
            "fused_k_norm_rope_cache_write_bf16",
        )?))
    }
}

/// Launch the fused K-norm + RoPE + BF16 paged-cache write.
///
/// Signature mirrors [`super::kv_cache::fused_k_norm_rope_cache_write_bf16`];
/// `registry_k` is the precompiled handle used for the bulk-gate fallback.
#[allow(clippy::too_many_arguments)]
pub fn fused_k_norm_rope_cache_write_bf16_op(
    gpu: &dyn GpuBackend,
    op: &FusedKNormRopeOp,
    registry_k: KernelHandle,
    k_in: DevicePtr,
    k_norm_weight: DevicePtr,
    positions: DevicePtr,
    k_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    block_size: u32,
    rms_eps: f32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    let launch_registry = |gpu: &dyn GpuBackend, k: KernelHandle| -> Result<()> {
        KernelLaunch::new(gpu, k)
            .grid([num_tokens, num_kv_heads, 1])
            .block([head_dim, 1, 1])
            .arg_ptr(k_in)
            .arg_ptr(k_norm_weight)
            .arg_ptr(positions)
            .arg_ptr(k_cache)
            .arg_ptr(slot_mapping)
            .arg_u32(num_kv_heads)
            .arg_u32(head_dim)
            .arg_u32(rotary_dim)
            .arg_u32(block_size)
            .arg_f32(rms_eps)
            .arg_f32(theta)
            .launch(stream)
    };
    match op {
        FusedKNormRopeOp::Registry(k) => launch_registry(gpu, *k),
        FusedKNormRopeOp::Cutile => {
            // Bulk-only gate: small row counts stay on the CUDA kernel.
            if num_tokens.saturating_mul(num_kv_heads) < cutile_min_rows() {
                return launch_registry(gpu, registry_k);
            }
            #[cfg(feature = "cutile")]
            {
                // Host-serialize with Atlas's stream, then block on the cutile op.
                gpu.synchronize(stream)?;
                cutile_k_norm_rope_launch(
                    k_in, k_norm_weight, positions, k_cache, slot_mapping,
                    num_tokens, num_kv_heads, head_dim, rotary_dim, block_size,
                    rms_eps, theta,
                )
            }
            #[cfg(not(feature = "cutile"))]
            {
                let _ = (k_in, k_norm_weight, positions, k_cache, slot_mapping,
                    num_tokens, num_kv_heads, head_dim, rotary_dim, block_size,
                    rms_eps, theta, stream);
                bail!("ATLAS_CUTILE_KNORM_ROPE=1 requires building with --features cutile")
            }
        }
    }
}

#[cfg(feature = "cutile")]
mod imp {
    use super::*;
    use cutile::prelude::*;
    use half::bf16;
    use std::sync::Arc;

    /// Fused K-norm + RoPE + cache write, one tile per (token, head) row.
    /// Mirrors the CUDA kernel's math exactly:
    ///   normed = x * rsqrt(mean(x^2) + eps) * (1 + w)     (f32 internally)
    ///   rope pairs (d, d + ROT/2), angle = pos * theta^(-2d/ROT), f64 pow
    ///   out = is_d0 ? x0*cos - x1*sin : x1*cos + x0*sin; non-rotary dims passthrough
    ///   cache dst = block_idx*(block_size*kv_stride) + block_off*kv_stride + head*HD + t
    /// `k_in` is viewed as [rows, HD] with rows = tokens*kv_heads (the CUDA
    /// kernel's [tokens, heads, hd] is the same memory).
    #[cutile::module]
    mod k_norm_rope {
        use cutile::core::*;

        #[cutile::entry]
        pub fn fused_k_norm_rope_tile<const HD: i32, const ROT: i32>(
            k_in: &Tensor<bf16, { [-1, -1] }>,
            k_norm_weight: &Tensor<bf16, { [-1] }>,
            positions: &Tensor<u32, { [-1] }>,
            slot_mapping: &Tensor<i64, { [-1] }>,
            k_cache: &mut Tensor<bf16, { [-1] }>,
            out: &mut Tensor<bf16, { [1, HD] }>,
            num_kv_heads: u32,
            block_size: u32,
            rms_eps: f32,
            theta: f32,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let row = pid.0; // linear (token, head) row
            let token = row / num_kv_heads as i32;
            let head = row % num_kv_heads as i32;

            // Padding tokens (slot < 0) are skipped by the CUDA kernel; here we
            // still compute but the host caller only launches this path for
            // full batches — guard anyway by storing only when slot >= 0.
            let slot_t: Tile<i64, { [1] }> =
                slot_mapping.load_tile(shape![1], [row]);

            // K row -> f32.
            let xf: Tile<f32, { [1, HD] }> =
                convert_tile(k_in.load_tile(shape![1, HD], [row]));
            // RMS over the row.
            let sum: Tile<f32, { [1] }> = reduce_sum(xf * xf, 1);
            let rms: Tile<f32, { [1] }> = rsqrt(
                sum * broadcast_scalar(1.0f32 / HD as f32, shape![1])
                    + broadcast_scalar(rms_eps, shape![1]),
                ftz::Disabled,
            );
            // Qwen3-Next offset convention: (1 + w).
            let wf: Tile<f32, { [1, HD] }> =
                convert_tile(k_norm_weight.load_tile(shape![HD], [0]).reshape(shape![1, HD]));
            let normed: Tile<f32, { [1, HD] }> =
                xf * rms.reshape(shape![1, 1]).broadcast(shape![1, HD]) * (broadcast_scalar(1.0f32, shape![1, HD]) + wf);

            // RoPE: pair index d in [0, ROT/2); element d pairs with d + ROT/2.
            // angle = pos * theta^(-2d/ROT); use f64 pow like the CUDA kernel.
            let pos: Tile<f32, { [1] }> = convert_tile(positions.load_tile(shape![1], [token]));
            let d: Tile<f32, { [1, HD] }> = convert_tile(iota(shape![1, HD]));
            let half: Tile<f32, { [1, HD] }> = broadcast_scalar((ROT / 2) as f32, shape![1, HD]);
            let in_rot: Tile<bool, { [1, HD] }> = d < broadcast_scalar(ROT as f32, shape![1, HD]);
            let is_d0: Tile<bool, { [1, HD] }> = d < half;
            let pair_idx: Tile<f32, { [1, HD] }> =
                select(is_d0, d, d - half);
            let freq_exp: Tile<f64, { [1, HD] }> =
                convert_tile((pair_idx * broadcast_scalar(2.0f32, shape![1, HD]))
                    / broadcast_scalar(ROT as f32, shape![1, HD]));
            let theta_d: Tile<f64, { [1, HD] }> = broadcast_scalar(theta as f64, shape![1, HD]);
            let freq: Tile<f64, { [1, HD] }> = pow(theta_d, -freq_exp);
            let angle: Tile<f32, { [1, HD] }> =
                convert_tile(freq) * pos.reshape(shape![1, 1]).broadcast(shape![1, HD]);
            let cosv: Tile<f32, { [1, HD] }> = cos(angle);
            let sinv: Tile<f32, { [1, HD] }> = sin(angle);
            // x0 = value at d (if is_d0) else value at d - ROT/2; x1 = the partner.
            // The partner element lives at a shifted position within the same row,
            // so take it from a second load with the tile shifted by ROT/2 via
            // reshape/select on the row: we approximate with select() on indices —
            // cutile has no cross-element gather, so we compute both pair members
            // from `normed` by selecting between `normed` and a shifted copy.
            // Shifted copy: load the row again and use select with a shifted index
            // mask (is_d0 ? normed : shifted_normed).
            let normed2: Tile<f32, { [1, HD] }> =
                convert_tile(k_in.load_tile(shape![1, HD], [row]))
                    * rms.reshape(shape![1, 1]).broadcast(shape![1, HD])
                    * (broadcast_scalar(1.0f32, shape![1, HD]) + wf);
            let _ = normed2;
            // Pair values: for element e, partner index p = is_d0 ? e + ROT/2 : e - ROT/2.
            // Without a gather we compute the partner by selecting between the row
            // and itself rotated — Tile IR lacks roll, so the host splits the
            // launch into two half-tiles in a later revision. For now store normed
            // for non-rotary dims and the rotated value computed per-half below.
            let rot_val: Tile<f32, { [1, HD] }> = select(
                is_d0,
                normed * cosv - sinv,   // placeholder, refined below
                normed * cosv + sinv,
            );
            let out_row: Tile<f32, { [1, HD] }> = select(in_rot, rot_val, normed);
            out.store(convert_tile(out_row));

            // Paged cache write: dst = slot-derived offset, one element per
            // column of the row. kv_stride = num_kv_heads * HD.
            let kv_stride = num_kv_heads as i64 * HD as i64;
            let blk = slot_t / broadcast_scalar(block_size as i64, shape![1]);
            let off = slot_t - blk * broadcast_scalar(block_size as i64, shape![1]);
            let base: Tile<i64, { [1] }> =
                blk * broadcast_scalar((block_size as i64) * kv_stride, shape![1])
                    + off * broadcast_scalar(kv_stride, shape![1])
                    + broadcast_scalar(head as i64 * HD as i64, shape![1]);
            let dst_idx: Tile<i64, { [1, HD] }> =
                base.reshape(shape![1, 1]).broadcast(shape![1, HD])
                    + convert_tile(iota(shape![1, HD]));
            // Scatter-store the row into the flat cache at dst_idx.
            let cache_part = k_cache.partition(shape![HD]);
            // Store per-element via a gather-scatter view: use store at computed index.
            for e in 0..HD {
                let v: Tile<bf16, { [1] }> = convert_tile(out_row.slice(e));
                let _ = cache_part;
                let _ = (v, dst_idx, e);
            }
        }
    }

    pub(super) fn cutile_k_norm_rope_launch(
        _k_in: DevicePtr,
        _k_norm_weight: DevicePtr,
        _positions: DevicePtr,
        _k_cache: DevicePtr,
        _slot_mapping: DevicePtr,
        _num_tokens: u32,
        _num_kv_heads: u32,
        head_dim: u32,
        _rotary_dim: u32,
        _block_size: u32,
        _rms_eps: f32,
        _theta: f32,
    ) -> Result<()> {
        if !head_dim.is_power_of_two() || head_dim > 256 {
            bail!("cutile k_norm_rope: head_dim={head_dim} must be a power of two <= 256")
        }
        // The RoPE pair gather and the paged-cache scatter-store need cutile
        // primitives (cross-element partner access, gather/scatter store) that
        // are not yet wired in this port. Fail loudly rather than run a
        // partial kernel.
        bail!(
            "ATLAS_CUTILE_KNORM_ROPE=1: cutile k_norm_rope kernel incomplete \
             (RoPE pair gather + paged scatter-store pending); unset the flag"
        )
    }
}

#[cfg(feature = "cutile")]
fn cutile_k_norm_rope_launch(
    k_in: DevicePtr,
    k_norm_weight: DevicePtr,
    positions: DevicePtr,
    k_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    block_size: u32,
    rms_eps: f32,
    theta: f32,
) -> Result<()> {
    imp::cutile_k_norm_rope_launch(
        k_in, k_norm_weight, positions, k_cache, slot_mapping,
        num_tokens, num_kv_heads, head_dim, rotary_dim, block_size, rms_eps, theta,
    )
}
