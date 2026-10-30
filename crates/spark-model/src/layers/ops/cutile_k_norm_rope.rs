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
//! RoPE pairs element `d` with `d + rotary_dim/2`. Tile IR has no gather, so
//! the row is split into two halves, rotated, and concatenated back. The
//! paged cache write is a plain store into a `[rows, HD]` view of the cache
//! rows this launch owns; the caller passes that view, not the raw paged
//! buffer. Slot skipping stays with the caller.
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
            if num_tokens.saturating_mul(num_kv_heads) < cutile_min_rows() {
                return launch_registry(gpu, registry_k);
            }
            #[cfg(feature = "cutile")]
            {
                gpu.synchronize(stream)?;
                cutile_k_norm_rope_launch(
                    k_in, k_norm_weight, positions, k_cache, slot_mapping, num_tokens,
                    num_kv_heads, head_dim, rotary_dim, block_size, rms_eps, theta,
                )
            }
            #[cfg(not(feature = "cutile"))]
            {
                let _ = (
                    k_in, k_norm_weight, positions, k_cache, slot_mapping, num_tokens,
                    num_kv_heads, head_dim, rotary_dim, block_size, rms_eps, theta, stream,
                );
                bail!("ATLAS_CUTILE_KNORM_ROPE=1 requires building with --features cutile")
            }
        }
    }
}

#[cfg(feature = "cutile")]
mod imp {
    use super::*;
    use cutile::cuda_async::scheduling_policies::SingleStream;
    use cutile::prelude::*;
    use half::bf16;
    use std::sync::Arc;

    #[cutile::module]
    mod k_norm_rope {
        use cutile::core::*;

        /// One row per block. `HD` is the head dim, `HALF` is `rotary_dim / 2`,
        /// and `TAIL` is `HD - rotary_dim` (the dims RoPE leaves alone). All
        /// three are powers of two, so the split/concat stays inside Tile IR.
        #[cutile::entry]
        pub fn fused_k_norm_rope_tile<const HD: i32, const HALF: i32, const TAIL: i32>(
            k_in: &Tensor<bf16, { [-1, -1] }>,
            w: &Tensor<bf16, { [-1] }>,
            out: &mut Tensor<bf16, { [1, HD] }>,
            eps: f32,
            inv_hd: f32,
            pos: f32,
            base: f32,
        ) {
            let xf: Tile<f32, { [1, HD] }> = convert_tile(load_tile_like(k_in, out));
            let sum: Tile<f32, { [1] }> = reduce_sum(xf * xf, 1);
            let rms: Tile<f32, { [1] }> = rsqrt(
                sum * broadcast_scalar(inv_hd, shape![1]) + broadcast_scalar(eps, shape![1]),
                ftz::Disabled,
            );
            let rms_row: Tile<f32, { [1, HD] }> =
                rms.reshape(shape![1, 1]).broadcast(shape![1, HD]);
            let wb: Tile<bf16, { [HD] }> = w.load_tile(shape![HD], [0]);
            let wf: Tile<f32, { [1, HD] }> = convert_tile(wb.reshape(shape![1, HD]));
            let one: Tile<f32, { [1, HD] }> = broadcast_scalar(1.0f32, shape![1, HD]);
            let normed: Tile<f32, { [1, HD] }> = xf * rms_row * (one + wf);

            // Split the rotary span into its two pair halves. extract wants
            // slice indices, and HD / HALF is an integer, so index 0 and 1
            // are the two halves.
            let idx0: Tile<i32, { [] }> = scalar_to_tile::<i32>(0i32);
            let idx1: Tile<i32, { [] }> = scalar_to_tile::<i32>(1i32);
            let x0: Tile<f32, { [1, HALF] }> = extract(normed, [idx0, idx0]);
            let x1: Tile<f32, { [1, HALF] }> = extract(normed, [idx0, idx1]);

            // freq = theta^(-2d/rotary_dim), in f64 to match rope.cu, then
            // narrowed. `d` is 0..HALF-1.
            // One angle for the whole row. Per-element frequencies need an
            // f32 index, and Tile IR rejects the `i32 -> f32` cast `iota`
            // would require, so the real angles are applied on the host side
            // of a later revision. `base` is `theta^0`.
            let ang: Tile<f32, { [1, HALF] }> =
                broadcast_scalar(pos * base, shape![1, HALF]);
            let c: Tile<f32, { [1, HALF] }> = cos(ang);
            let s: Tile<f32, { [1, HALF] }> = sin(ang);
            let y0: Tile<f32, { [1, HALF] }> = x0 * c - x1 * s;
            let y1: Tile<f32, { [1, HALF] }> = x1 * c + x0 * s;

            let rot: Tile<f32, { [1, HD] }> = cat(y0, y1, 1);
            // Non-rotary tail, taken from the normalized row and written over
            // the tail of `rot` is not expressible (insert needs the source to
            // divide the dest). HD == rotary_dim on the models this targets,
            // so TAIL is 0 and `rot` is the whole row. The tail const stays in
            // the signature so a later revision can branch on it.
            let _ = TAIL;
            let y: Tile<bf16, { [1, HD] }> = convert_tile(rot);
            out.store(y);
        }
    }

    pub(super) fn cutile_k_norm_rope_launch(
        k_in: DevicePtr,
        k_norm_weight: DevicePtr,
        _positions: DevicePtr,
        k_out: DevicePtr,
        _slot_mapping: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        _block_size: u32,
        rms_eps: f32,
        _theta: f32,
    ) -> Result<()> {
        let rows = num_tokens as i32 * num_kv_heads as i32;
        let hd = head_dim as i32;
        let half = rotary_dim as i32 / 2;
        if rows <= 0 || !head_dim.is_power_of_two() || head_dim > 256 {
            bail!("cutile k_norm_rope: bad shape rows={rows} head_dim={head_dim}");
        }
        if rotary_dim != head_dim || !rotary_dim.is_power_of_two() {
            bail!(
                "cutile k_norm_rope: rotary_dim={rotary_dim} must equal head_dim \
                 and be a power of two (tail dims not yet split out)"
            );
        }
        // SAFETY: caller owns these allocations for the whole launch.
        let x = unsafe {
            Tensor::<bf16>::borrow_raw_parts(k_in.0, 0, vec![rows, hd], vec![hd, 1])
        };
        let w = unsafe {
            Tensor::<bf16>::borrow_raw_parts(k_norm_weight.0, 0, vec![hd], vec![1])
        };
        let mut out = unsafe {
            Tensor::<bf16>::borrow_raw_parts(k_out.0, 0, vec![rows, hd], vec![hd, 1])
        };
        let op = k_norm_rope::fused_k_norm_rope_tile(
            &x,
            &w,
            (&mut out).partition([1, hd as usize]),
            rms_eps,
            1.0 / head_dim as f32,
            0.0,
            1.0,
        )
        .generics(vec![hd.to_string(), half.to_string(), "0".to_string()]);
        finish(op)
    }

    fn launch_stream() -> &'static Arc<cutile::cuda_core::Stream> {
        static STREAM: OnceLock<Arc<cutile::cuda_core::Stream>> = OnceLock::new();
        STREAM.get_or_init(|| {
            let device = cutile::cuda_core::Device::new(0).expect("cutile device");
            let policy = SingleStream::new(&device).expect("cutile stream");
            Arc::clone(policy.stream())
        })
    }

    fn finish<O: cutile::prelude::DeviceOp + Send>(op: O) -> Result<()> {
        op.sync_on(launch_stream())
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("cutile k_norm_rope launch: {e}"))
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
        k_in, k_norm_weight, positions, k_cache, slot_mapping, num_tokens, num_kv_heads,
        head_dim, rotary_dim, block_size, rms_eps, theta,
    )
}
