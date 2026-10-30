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
                let _ = stream;
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

    /// The cutile module holding the RMSNorm kernel. One row (token) per tile:
    /// the partition shape `[1, H]` gives each block one full `hidden_size`
    /// row, so the block reduces that row and writes it back. `H` is a const
    /// generic supplied via `.generics()` at the launch site, so cutile
    /// specializes (and disk-caches) one compile per hidden size seen.
    #[cutile::module]
    mod rms_norm {
        use cutile::core::*;

        /// HF-vanilla RMSNorm: `out = x * w / sqrt(mean(x^2) + eps)`.
        /// `x`: `[tokens, hidden]` bf16, `w`: `[hidden]` bf16, `out`: `[1, H]`
        /// bf16 slab per program. Math in f32, stored back as bf16.
        #[cutile::entry]
        pub fn rms_norm_vanilla_tile<const H: i32>(
            x: &Tensor<bf16, { [-1, -1] }>,
            w: &Tensor<bf16, { [-1] }>,
            out: &mut Tensor<bf16, { [1, H] }>,
            eps: f32,
        ) {
            // This block's row of x, widened to f32 for the math.
            let xf: Tile<f32, { [1, H] }> = convert_tile(load_tile_like(x, out));
            // Row sum of squares -> [1], then rms = rsqrt(mean + eps).
            let sum: Tile<f32, { [1] }> = reduce_sum(xf * xf, 1);
            let inv_h: Tile<f32, { [1] }> = broadcast_scalar(1.0f32 / (H as f32), shape![1]);
            let eps_t: Tile<f32, { [1] }> = broadcast_scalar(eps, shape![1]);
            let rms: Tile<f32, { [1] }> = rsqrt(sum * inv_h + eps_t, ftz::Disabled);
            let rms_row: Tile<f32, { [1, H] }> = rms.reshape(shape![1, H]).broadcast(out.shape());
            // Weight row: load at the partition origin. NOTE (issue #263):
            // load_tile_like on a broadcast input indexes by partition — use
            // the explicit tile load against the output's slab shape instead.
            let w_tile: Tile<bf16, { [H] }> = w.load_tile(shape![H], [0]);
            let wf: Tile<f32, { [H] }> = convert_tile(w_tile);
            let w_row: Tile<f32, { [1, H] }> = wf.reshape(shape![1, H]).broadcast(out.shape());
            out.store(convert_tile(xf * rms_row * w_row));
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
        if hidden <= 0 || tokens <= 0 {
            bail!("cutile rms_norm: non-positive shape tokens={tokens} hidden={hidden}");
        }
        // SAFETY: input/weight/output are live device allocations owned by the
        // caller (the GpuBackend's pool), valid for tokens*hidden*2,
        // hidden*2, and tokens*hidden*2 bytes respectively. They stay mapped
        // for the whole launch because we block on the DeviceOp before
        // returning; no other path touches output while this runs.
        let x = unsafe {
            Tensor::<bf16>::borrow_raw_parts(
                input.0,
                0,
                vec![tokens, hidden],
                vec![hidden, 1],
            )
        };
        let w = unsafe {
            Tensor::<bf16>::borrow_raw_parts(weight.0, 0, vec![hidden], vec![1])
        };
        let mut out = unsafe {
            Tensor::<bf16>::borrow_raw_parts(
                output.0,
                0,
                vec![tokens, hidden],
                vec![hidden, 1],
            )
        };
        let op = rms_norm::rms_norm_vanilla_tile(
            &x,
            &w,
            (&mut out).partition([1, hidden as usize]),
            eps,
        )
        .generics(vec![hidden.to_string()]);
        runtime().block_on(async move {
            op.await.map(|_| ()).map_err(|e| anyhow::anyhow!("cutile rms_norm launch: {e}"))
        })
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