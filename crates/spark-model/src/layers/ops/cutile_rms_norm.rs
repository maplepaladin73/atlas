// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in cutile-rs (CUDA Tile) RMSNorm, gated on `ATLAS_CUTILE_RMSNORM=1`.
//! See Atlas-Inf/atlas issue #263 for the benchmark writeup (parity at decode,
//! 6-8% faster at bulk shapes on sm_120 and sm_121).
//!
//! cutile 0.4 JIT-compiles Rust to a CUBIN via the external `tileiras` tool but
//! does not hand the CUBIN back to the caller, so this path cannot produce a
//! `KernelHandle` for Atlas's PTX registry. Instead the cutile kernel is
//! launched through cutile's own async `DeviceOp` API, driven from a dedicated
//! current-thread tokio runtime so the sync `ops::rms_norm` call sites are
//! unchanged. The compile itself is disk-cached by cutile's JIT store.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

use crate::weight_map::DenseWeight;

/// True when the cutile RMSNorm experiment is enabled.
pub fn cutile_rmsnorm_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_CUTILE_RMSNORM").as_deref() == Ok("1"))
}

/// Resolve the vanilla-convention RMS norm op: cutile when
/// `ATLAS_CUTILE_RMSNORM=1`, else the precompiled `rms_norm_vanilla`.
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
        RmsNormOp::Registry(k) => KernelLaunch::new(gpu, *k)
            .grid([num_tokens, 1, 1])
            .block([hidden_size.min(1024), 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight.weight)
            .arg_ptr(output)
            .arg_u32(hidden_size)
            .arg_f32(eps)
            .launch(stream),
        RmsNormOp::Cutile => cutile_rms_norm_launch(
            input,
            weight.weight,
            output,
            num_tokens,
            hidden_size,
            eps,
        ),
    }
}

fn cutile_rms_norm_launch(
    _input: DevicePtr,
    _weight: DevicePtr,
    _output: DevicePtr,
    _num_tokens: u32,
    _hidden_size: u32,
    _eps: f32,
) -> Result<()> {
    // cutile 0.4's public API is async DeviceOp over its own tensor/partition
    // types (cuda-async), not raw DevicePtr + cuLaunchKernel, so it cannot be
    // bridged onto Atlas's KernelLaunch path without either (a) an async
    // runtime in spark-model plus tensor wrappers around DevicePtr, or
    // (b) extracting the CUBIN — which cutile 0.4 does not expose.
    // That wiring is the next step; this stub keeps the env gate compiling
    // and fails loudly rather than silently running the CUDA kernel.
    bail!(
        "ATLAS_CUTILE_RMSNORM=1: cutile launch not yet wired into the sync op path \
         (cutile 0.4 is async-only); unset the flag to use rms_norm_vanilla"
    )
}