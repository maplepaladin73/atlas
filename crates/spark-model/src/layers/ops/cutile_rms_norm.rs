// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in cutile-rs (CUDA Tile) RMSNorm, gated on `ATLAS_CUTILE_RMSNORM=1`.
//! See Atlas-Inf/atlas issue #263 for the benchmark writeup (parity at decode,
//! 6-8% faster at bulk shapes on sm_120 and sm_121).
//!
//! cutile 0.4 JIT-compiles Rust to a CUBIN via the external `tileiras` tool but
//! never hands the CUBIN back to the caller, so this path cannot produce a
//! `KernelHandle` for Atlas's PTX registry. Instead the cutile kernel is
//! launched through cutile's own async `DeviceOp` API: the raw `DevicePtr`s are
//! borrowed as cutile `Tensor`s (`borrow_raw_parts`, no ownership transfer) and
//! the launch is driven on a dedicated current-thread tokio runtime, so the
//! sync `ops::rms_norm` call sites stay unchanged. cutile's JIT store disk-caches
//! the compile, so the first launch pays it once per process.
//!
//! Tile IR requires every tile dimension to be a positive power of two, so a
//! row wider than one legal tile (Qwen's 7168, for one) cannot be a single
//! partition. Each block owns one row and walks it in `C`-wide chunks
//! (`C` = 256): sum of squares, then `out = x * w * rsqrt(mean + eps)`. One
//! compile covers every hidden size that is a multiple of 256.
//!
//! Requires the `cutile` cargo feature (`--features cutile`); without it the
//! env flag is ignored and the precompiled kernel is always used.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

use crate::weight_map::DenseWeight;

/// True when the cutile RMSNorm experiment is enabled AND compiled in.
pub fn cutile_rmsnorm_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "cutile") && std::env::var("ATLAS_CUTILE_RMSNORM").as_deref() == Ok("1")
    })
}

/// Resolve the vanilla-convention RMS norm op: cutile when
/// `ATLAS_CUTILE_RMSNORM=1` (and the `cutile` feature is on), else the
/// precompiled `rms_norm_vanilla`.
pub fn resolve_rms_norm_vanilla(gpu: &dyn GpuBackend) -> Result<RmsNormOp> {
    if cutile_rmsnorm_enabled() {
        Ok(RmsNormOp::Cutile)
    } else {
        Ok(RmsNormOp::Registry(
            gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
        ))
    }
}

/// Either kernel flavor for `out = x * w / rms` (HF-vanilla convention).
pub enum RmsNormOp {
    Registry(KernelHandle),
    Cutile,
}

/// Launch `out = x * w / rms` with whichever flavor was resolved.
///
/// Signature mirrors [`super::norm::rms_norm`] exactly so call sites swap only
/// the handle argument.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_op(
    gpu: &dyn GpuBackend,
    op: &RmsNormOp,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    match op {
        RmsNormOp::Registry(k) => {
            KernelLaunch::new(gpu, *k)
                .grid([num_tokens, 1, 1])
                .block([hidden_size.min(1024), 1, 1])
                .arg_ptr(input)
                .arg_ptr(weight.weight)
                .arg_ptr(output)
                .arg_u32(hidden_size)
                .arg_f32(eps)
                .launch(stream)
        }
        RmsNormOp::Cutile => {
            #[cfg(feature = "cutile")]
            {
                // cutile 0.4 launches on its own stream. Wait for `stream`
                // first so the inputs are finished, and the launch below does
                // not return until the cutile op itself has finished, so the
                // next kernel on `stream` cannot overtake it.
                gpu.synchronize(stream)?;
                cutile_rms_norm_launch(input, weight.weight, output, num_tokens, hidden_size, eps)
            }
            #[cfg(not(feature = "cutile"))]
            {
                let _ = (input, weight, output, num_tokens, hidden_size, eps, stream);
                bail!("ATLAS_CUTILE_RMSNORM=1 requires building with --features cutile")
            }
        }
    }
}

#[cfg(feature = "cutile")]
mod imp {
    use super::*;
    use cutile::prelude::*;
    use half::bf16;

    #[cutile::module]
    mod rms_norm {
        use cutile::core::*;

        /// One tile per row. `H` must be a power of two (Tile IR rule).
        /// This is the kernel from the 2026-10-29 bench: one load, one
        /// reduce, one store. No chunk loop.
        #[cutile::entry]
        pub fn rms_norm_pow2<const H: i32>(
            x: &Tensor<bf16, { [-1, -1] }>,
            w: &Tensor<bf16, { [-1] }>,
            out: &mut Tensor<bf16, { [1, H] }>,
            eps: f32,
            inv_h: f32,
        ) {
            let xf: Tile<f32, { [1, H] }> = convert_tile(load_tile_like(x, out));
            let sum: Tile<f32, { [1] }> = reduce_sum(xf * xf, 1);
            let mean = sum * broadcast_scalar(inv_h, shape![1]);
            let rms: Tile<f32, { [1] }> =
                rsqrt(mean + broadcast_scalar(eps, shape![1]), ftz::Disabled);
            let rms_row: Tile<f32, { [1, H] }> =
                rms.reshape(shape![1, 1]).broadcast(out.shape());
            let wb: Tile<bf16, { [H] }> = w.load_tile(shape![H], [0]);
            let wf: Tile<f32, { [1, H] }> = convert_tile(wb.reshape(shape![1, H]));
            let y: Tile<bf16, { [1, H] }> = convert_tile(xf * rms_row * wf);
            out.store(y);
        }

        /// One block per row; the block owns every chunk of that row.
        /// `MAP_SHAPE` is `[1, OWNED]`, so `iter_indices` yields one item per
        /// row and the column axis is walked by counted loops. See cutile's
        /// `owned_axis_composite_store` test for the same shape.
        #[cutile::entry]
        pub fn rms_norm_row<const C: i32, const MAP_SHAPE: [i32; 2]>(
            x: &Tensor<bf16, { [-1, -1] }>,
            w: &Tensor<bf16, { [-1] }>,
            mut out: MappedPartitionMut<bf16, { [1, C] }, MAP_SHAPE>,
            eps: f32,
            inv_h: f32,
        ) {
            let rows = num_tiles(&out, 0);
            let cols = num_tiles(&out, 1);
            let _ = (rows, cols);
            let xp = x.partition(shape![1, C]);
            let wp = w.partition(shape![C]);
            let cols = cols.into_dim();
            for index in out.iter_indices() {
                let (row, _col) = index.components();
                let mut acc: Tile<f32, { [1] }> = broadcast_scalar(0.0f32, shape![1]);
                for k in cols {
                    let xf: Tile<f32, { [1, C] }> = convert_tile(xp.load([row, k]));
                    acc = acc + reduce_sum(xf * xf, 1);
                }
                let mean = acc * broadcast_scalar(inv_h, shape![1]);
                let rms: Tile<f32, { [1] }> =
                    rsqrt(mean + broadcast_scalar(eps, shape![1]), ftz::Disabled);
                let rms_row: Tile<f32, { [1, C] }> =
                    rms.reshape(shape![1, 1]).broadcast(shape![1, C]);
                for k in cols {
                    let xf: Tile<f32, { [1, C] }> = convert_tile(xp.load([row, k]));
                    let wf: Tile<f32, { [1, C] }> =
                        convert_tile(wp.load([k]).reshape(shape![1, C]));
                    let y: Tile<bf16, { [1, C] }> = convert_tile(xf * rms_row * wf);
                    out.store(y, coord((row, k)));
                }
            }
        }
    }

    pub(super) fn cutile_rms_norm_launch(
        input: DevicePtr,
        weight: DevicePtr,
        output: DevicePtr,
        num_tokens: u32,
        hidden_size: u32,
        eps: f32,
    ) -> Result<()> {
        let hidden = hidden_size as i32;
        let tokens = num_tokens as i32;
        if tokens <= 0 || hidden <= 0 {
            bail!("cutile rms_norm: non-positive shape tokens={tokens} hidden={hidden}");
        }
        // SAFETY: caller owns these allocations for the whole launch; we block
        // before returning, so nothing else touches `output` meanwhile.
        let x = unsafe {
            Tensor::<bf16>::borrow_raw_parts(input.0, 0, vec![tokens, hidden], vec![hidden, 1])
        };
        let w = unsafe { Tensor::<bf16>::borrow_raw_parts(weight.0, 0, vec![hidden], vec![1]) };
        let mut out = unsafe {
            Tensor::<bf16>::borrow_raw_parts(output.0, 0, vec![tokens, hidden], vec![hidden, 1])
        };
        let inv_h = 1.0 / (hidden as f32);
        // A power-of-two row is one tile. Anything else (7168) is cut into
        // the largest power-of-two chunk that divides it.
        if hidden > 0 && (hidden as u32).is_power_of_two() && hidden <= 8192 {
            let op = rms_norm::rms_norm_pow2(
                &x,
                &w,
                (&mut out).partition([1, hidden as usize]),
                eps,
                inv_h,
            )
            .generics(vec![hidden.to_string()]);
            runtime().block_on(async move {
                op.await
                    .map(|_| ())
                    .map_err(|e| anyhow::anyhow!("cutile rms_norm launch: {e}"))
            })
        } else {
            let mut c = 4096i32;
            while c > 1 && hidden % c != 0 {
                c /= 2;
            }
            if hidden % c != 0 {
                bail!("cutile rms_norm: hidden={hidden} has no power-of-two chunk");
            }
            let op = rms_norm::rms_norm_row(
                &x,
                &w,
                (&mut out).partition([1, c as usize]).map([1, OWNED], tokens as u32),
                eps,
                inv_h,
            )
            .generics(vec![c.to_string(), "1".to_string(), "0".to_string()]);
            runtime().block_on(async move {
                op.await
                    .map(|_| ())
                    .map_err(|e| anyhow::anyhow!("cutile rms_norm launch: {e}"))
            })
        }
    }

    fn runtime() -> &'static tokio::runtime::Runtime {
        static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
        RT.get_or_init(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("cutile tokio runtime")
        })
    }
}

#[cfg(feature = "cutile")]
fn cutile_rms_norm_launch(
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
) -> Result<()> {
    imp::cutile_rms_norm_launch(input, weight, output, num_tokens, hidden_size, eps)
}
