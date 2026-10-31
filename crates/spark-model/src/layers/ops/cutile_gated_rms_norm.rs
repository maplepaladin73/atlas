// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in cutile-rs port of `gated_rms_norm_f32_input_sigmoid`
//! (kernels/gb10/common/rms_norm.cu), gated on
//! `ATLAS_CUTILE_GATED_RMSNORM=1`. Fourth cutile kernel; target model is
//! qwen4_exp (Qwen3.8-Flash-Next), which declares
//! `output_gate_type = "sigmoid"` — the SiLU twin is the WRONG gate for
//! this model and must not be ported.
//!
//! Math per row (f32 input `x` [rows, H], bf16 gate `g` [rows, gate_stride],
//! bf16 weight `w` [H], bf16 out [rows, H]):
//!   rms = rsqrt(mean(x^2) + eps)
//!   out = bf16(x * rms * w * sigmoid(g)),  sigmoid(g) = 1 / (1 + exp(-g))
//!
//! Same class as the merged RMSNorm port (PR #1) and K-norm+RoPE port
//! (PR #2): one row per tile, one reduction, elementwise math, one store.
//! The f32 input means no bf16→f32 `convert_tile` on `x` (unlike PR #1/#2).
//!
//! Bridge pattern shared with the other cutile modules: raw `DevicePtr`s
//! borrowed as cutile `Tensor`s (no ownership transfer), SingleStream
//! policy, host-side `gpu.synchronize(stream)` before launch, block on
//! completion.
//!
//! Requires the `cutile` cargo feature; without it the env flag is ignored
//! and the precompiled kernel is always used.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

use crate::weight_map::DenseWeight;

/// True when the cutile gated-RMSNorm experiment is enabled AND compiled in.
pub fn cutile_gated_rmsnorm_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "cutile")
            && std::env::var("ATLAS_CUTILE_GATED_RMSNORM").as_deref() == Ok("1")
    })
}

/// Rows below this stay on the CUDA kernel (the cutile launcher's fixed
/// overhead dominates at decode sizes).
pub fn cutile_min_rows() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_CUTILE_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    })
}

/// Either flavor of the gated RMS norm op.
pub enum GatedRmsNormOp {
    Registry(KernelHandle),
    Cutile,
}

impl GatedRmsNormOp {
    /// True when the cutile flavor was resolved (flag on + feature compiled).
    pub fn is_cutile(&self) -> bool {
        matches!(self, GatedRmsNormOp::Cutile)
    }
}

/// Resolve the op: cutile when enabled and compiled in, else the registry
/// handle for `kernel_name` (e.g. `gated_rms_norm_f32_input_sigmoid`).
pub fn resolve_gated_rms_norm(gpu: &dyn GpuBackend, kernel_name: &str) -> Result<GatedRmsNormOp> {
    if cutile_gated_rmsnorm_enabled() {
        Ok(GatedRmsNormOp::Cutile)
    } else {
        Ok(GatedRmsNormOp::Registry(gpu.kernel("norm", kernel_name)?))
    }
}

/// Launch the gated RMS norm with whichever flavor was resolved.
///
/// Signature mirrors [`super::norm::gated_rms_norm`]; `registry_k` is the
/// precompiled handle used for the bulk-gate fallback.
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm_op(
    gpu: &dyn GpuBackend,
    op: &GatedRmsNormOp,
    registry_k: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    gate_stride: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let launch_registry = |gpu: &dyn GpuBackend, k: KernelHandle| -> Result<()> {
        KernelLaunch::new(gpu, k)
            .grid([num_tokens, 1, 1])
            .block([hidden_size.min(1024), 1, 1])
            .arg_ptr(input)
            .arg_ptr(gate)
            .arg_ptr(weight.weight)
            .arg_ptr(output)
            .arg_u32(hidden_size)
            .arg_f32(eps)
            .arg_u32(gate_stride)
            .arg_u32(hidden_size) // group_size, unused by these kernels
            .launch(stream)
    };
    match op {
        GatedRmsNormOp::Registry(k) => launch_registry(gpu, *k),
        GatedRmsNormOp::Cutile => {
            if num_tokens < cutile_min_rows() {
                return launch_registry(gpu, registry_k);
            }
            #[cfg(feature = "cutile")]
            {
                gpu.synchronize(stream)?;
                cutile_gated_rms_norm_launch(
                    input,
                    gate,
                    weight.weight,
                    output,
                    num_tokens,
                    hidden_size,
                    gate_stride,
                    eps,
                )
            }
            #[cfg(not(feature = "cutile"))]
            {
                let _ = (
                    input,
                    gate,
                    weight,
                    output,
                    num_tokens,
                    hidden_size,
                    gate_stride,
                    eps,
                    stream,
                );
                bail!("ATLAS_CUTILE_GATED_RMSNORM=1 requires building with --features cutile")
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
    mod gated_rms_norm {
        use cutile::core::*;

        /// One row per tile. `H` must be a power of two (Tile IR rule).
        /// Input is f32; gate and weight are bf16 and convert up.
        /// `GS` is the gate stride (elements between gate rows); the gate
        /// row for `row` starts at `[row, 0]` of a `[rows, GS]` view.
        #[cutile::entry]
        pub fn gated_rms_norm_sigmoid_tile<const H: i32>(
            x: &Tensor<f32, { [-1, -1] }>,
            g: &Tensor<bf16, { [-1, -1] }>,
            w: &Tensor<bf16, { [-1] }>,
            out: &mut Tensor<bf16, { [1, H] }>,
            gate_stride: i32,
            eps: f32,
            inv_h: f32,
        ) {
            let (row, _, _) = get_tile_block_id();
            let xf: Tile<f32, { [1, H] }> = x.load_tile(shape![1, H], [row, 0]);
            let sum: Tile<f32, { [1] }> = reduce_sum(xf * xf, 1);
            let mean = sum * broadcast_scalar(inv_h, shape![1]);
            let rms: Tile<f32, { [1] }> =
                rsqrt(mean + broadcast_scalar(eps, shape![1]), ftz::Disabled);
            let rms_row: Tile<f32, { [1, H] }> = rms.reshape(shape![1, 1]).broadcast(out.shape());

            let gf: Tile<f32, { [1, H] }> =
                convert_tile(g.load_tile(shape![1, H], [row * gate_stride, 0]));
            // sigmoid(g) = 1 / (1 + exp(-g))
            let one: Tile<f32, { [1, H] }> = broadcast_scalar(1.0f32, shape![1, H]);
            let sig: Tile<f32, { [1, H] }> = one / (one + exp(-gf));

            let wb: Tile<bf16, { [H] }> = w.load_tile(shape![H], [0]);
            let wf: Tile<f32, { [1, H] }> = convert_tile(wb.reshape(shape![1, H]));

            let y: Tile<bf16, { [1, H] }> = convert_tile(xf * rms_row * wf * sig);
            out.store(y);
        }
    }

    pub(super) fn cutile_gated_rms_norm_launch(
        input: DevicePtr,
        gate: DevicePtr,
        weight: DevicePtr,
        output: DevicePtr,
        num_tokens: u32,
        hidden_size: u32,
        gate_stride: u32,
        eps: f32,
    ) -> Result<()> {
        let hidden = hidden_size as i32;
        let tokens = num_tokens as i32;
        let gstride = gate_stride as i32;
        if tokens <= 0 || hidden <= 0 {
            bail!("cutile gated_rms_norm: non-positive shape tokens={tokens} hidden={hidden}");
        }
        if gstride < hidden {
            bail!("cutile gated_rms_norm: gate_stride={gstride} < hidden={hidden}");
        }
        if !(hidden_size.is_power_of_two() && hidden_size <= 8192) {
            bail!(
                "cutile gated_rms_norm: hidden={hidden} must be a power of two <= 8192 \
                 (chunked path not implemented; PR #1's rms_norm_row pattern applies if needed)"
            );
        }
        // SAFETY: caller owns these allocations for the whole launch; we
        // block before returning, so nothing else touches `output` meanwhile.
        // The gate is viewed as [tokens, gate_stride] and the kernel indexes
        // it flat ([row * gate_stride, 0]), matching the CUDA layout.
        let x = unsafe {
            Tensor::<f32>::borrow_raw_parts(input.0, 0, vec![tokens, hidden], vec![hidden, 1])
        };
        let g =
            unsafe { Tensor::<bf16>::borrow_raw_parts(gate.0, 0, vec![tokens * gstride], vec![1]) };
        let w = unsafe { Tensor::<bf16>::borrow_raw_parts(weight.0, 0, vec![hidden], vec![1]) };
        let mut out = unsafe {
            Tensor::<bf16>::borrow_raw_parts(output.0, 0, vec![tokens, hidden], vec![hidden, 1])
        };
        let op = gated_rms_norm::gated_rms_norm_sigmoid_tile(
            &x,
            &g,
            &w,
            (&mut out).partition([1, hidden as usize]),
            gstride,
            eps,
            1.0 / (hidden as f32),
        )
        .generics(vec![hidden.to_string()]);
        finish(op)
    }

    /// One stream for every cutile launch (see cutile_rms_norm.rs: a stream
    /// pool would turn `sync_on` into a device-wide sync).
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
            .map_err(|e| anyhow::anyhow!("cutile gated_rms_norm launch: {e}"))
    }
}

#[cfg(feature = "cutile")]
fn cutile_gated_rms_norm_launch(
    input: DevicePtr,
    gate: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    gate_stride: u32,
    eps: f32,
) -> Result<()> {
    imp::cutile_gated_rms_norm_launch(
        input,
        gate,
        weight,
        output,
        num_tokens,
        hidden_size,
        gate_stride,
        eps,
    )
}
