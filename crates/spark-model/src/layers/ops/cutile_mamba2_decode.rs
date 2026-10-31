// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in cutile-rs port of `mamba2_ssm_decode`
//! (kernels/gb10/common/mamba2_ssm_decode.cu), gated on
//! `ATLAS_CUTILE_MAMBA2_DECODE=1`. Third cutile kernel — and a different class
//! from the first two: per (batch, head) it does a state-sized output reduction
//! (y = C·H + D·x) AND a rank-1 outer-product state update
//! (H[hd, s] = dA*H[hd,s] + dt*x[hd]*B[g,s]) over a [head_dim, state_size]
//! tile. PR #3 answers whether tile loops beat mma-shaped tiles at decode sizes.
//!
//! Bridge pattern shared with `cutile_rms_norm` / `cutile_k_norm_rope`: raw
//! `DevicePtr`s borrowed as cutile `Tensor`s (no ownership transfer),
//! SingleStream policy, host-side `gpu.synchronize(stream)` before launch,
//! block-on completion. The CUDA kernel runs one block per (head, batch) with
//! `state_size` threads; the cutile port uses one tile per (head, batch) with
//! partition `[HD, S]`-shaped tiles (both powers of two on gb10: HD=64, S=128).
//!
//! Requires the `cutile` cargo feature; without it the env flag is ignored and
//! the precompiled kernel is always used.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::sync::OnceLock;

/// True when the cutile Mamba-2 decode experiment is enabled AND compiled in.
pub fn cutile_mamba2_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "cutile")
            && std::env::var("ATLAS_CUTILE_MAMBA2_DECODE").as_deref() == Ok("1")
    })
}

/// Batch rows below this stay on the CUDA kernel (decode is batch=1; the
/// cutile launcher's fixed overhead dominates there).
pub fn cutile_min_batch() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_CUTILE_MIN_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    })
}

/// Either flavor of the mamba2 decode op.
pub enum Mamba2DecodeOp {
    Registry(KernelHandle),
    Cutile,
}

/// Resolve the op: cutile when enabled and compiled in, else the registry handle.
pub fn resolve_mamba2_decode(gpu: &dyn GpuBackend) -> Result<Mamba2DecodeOp> {
    if cutile_mamba2_enabled() {
        Ok(Mamba2DecodeOp::Cutile)
    } else {
        Ok(Mamba2DecodeOp::Registry(gpu.kernel(
            "mamba2_ssm",
            "mamba2_ssm_decode",
        )?))
    }
}

/// Launch one Mamba-2 SSM decode step.
///
/// Signature mirrors the `ssm_decode` method in `nemotron_mamba2.rs`
/// (grid heads x batch, block state_size); `registry_k` is the precompiled
/// handle used for the batch-gate fallback.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssm_decode_op(
    gpu: &dyn GpuBackend,
    op: &Mamba2DecodeOp,
    registry_k: KernelHandle,
    h_state: DevicePtr,
    x: DevicePtr,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    dt_raw: DevicePtr,
    a_log: DevicePtr,
    d_param: DevicePtr,
    dt_bias: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    dt_min: f32,
    dt_max: f32,
    stream: u64,
) -> Result<()> {
    let launch_registry = |gpu: &dyn GpuBackend, k: KernelHandle| -> Result<()> {
        KernelLaunch::new(gpu, k)
            .grid([num_heads, batch_size, 1])
            .block([state_size, 1, 1])
            .arg_ptr(h_state)
            .arg_ptr(x)
            .arg_ptr(b_proj)
            .arg_ptr(c_proj)
            .arg_ptr(dt_raw)
            .arg_ptr(a_log)
            .arg_ptr(d_param)
            .arg_ptr(dt_bias)
            .arg_ptr(output)
            .arg_u32(batch_size)
            .arg_u32(num_heads)
            .arg_u32(head_dim)
            .arg_u32(state_size)
            .arg_u32(n_groups)
            .arg_f32(dt_min)
            .arg_f32(dt_max)
            .launch(stream)
    };
    match op {
        Mamba2DecodeOp::Registry(k) => launch_registry(gpu, *k),
        Mamba2DecodeOp::Cutile => {
            if batch_size < cutile_min_batch() {
                return launch_registry(gpu, registry_k);
            }
            #[cfg(feature = "cutile")]
            {
                gpu.synchronize(stream)?;
                cutile_mamba2_launch(
                    h_state, x, b_proj, c_proj, dt_raw, a_log, d_param, dt_bias, output,
                    batch_size, num_heads, head_dim, state_size, n_groups, dt_min, dt_max,
                )
            }
            #[cfg(not(feature = "cutile"))]
            {
                let _ = (h_state, x, b_proj, c_proj, dt_raw, a_log, d_param, dt_bias,
                    output, batch_size, num_heads, head_dim, state_size, n_groups,
                    dt_min, dt_max, stream);
                bail!("ATLAS_CUTILE_MAMBA2_DECODE=1 requires building with --features cutile")
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

    /// One tile per (head, batch): partition `[HD, S]` over the state tile.
    /// dt = clamp(softplus(dt_raw + dt_bias), dt_min, dt_max); dA = exp(-exp(A_log[h])*dt);
    /// H[hd, s] = dA*H[hd, s] + dt*x[hd]*B[g, s); y[hd] = sum_s(H*C) + D[h]*x[hd].
    #[cutile::module]
    mod mamba2 {
        use cutile::core::*;

        #[cutile::entry]
        pub fn mamba2_decode_tile<const HD: i32, const S: i32>(
            h_state: &mut Tensor<f32, { [-1, -1] }>,
            x: &Tensor<bf16, { [-1] }>,
            b_proj: &Tensor<bf16, { [-1] }>,
            c_proj: &Tensor<bf16, { [-1] }>,
            dt_raw: &Tensor<bf16, { [-1] }>,
            a_log: &Tensor<f32, { [-1] }>,
            d_param: &Tensor<f32, { [-1] }>,
            dt_bias: &Tensor<f32, { [-1] }>,
            out: &mut Tensor<bf16, { [HD, 1] }>,
            state: &mut Tensor<f32, { [HD, S] }>,
            num_heads: u32,
            n_groups: u32,
            dt_min: f32,
            dt_max: f32,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let head = pid.0;
            let b = pid.1;
            let heads_per_group = (num_heads / n_groups) as i32;
            let group = head / heads_per_group;

            // dt = clamp(softplus(dt_raw[b, head] + dt_bias[head]), dt_min, dt_max)
            let dt_bf: Tile<bf16, { [1] }> = dt_raw.load_tile(shape![1], [b * num_heads as i32 + head]);
            let bias: Tile<f32, { [1] }> = a_log.load_tile(shape![1], [head]); // placeholder shape, fixed below
            let _ = bias;
            let dtab: Tile<f32, { [1] }> = convert_tile(dt_bf);
            let dtb: Tile<f32, { [1] }> = dt_bias.load_tile(shape![1], [head]);
            let raw: Tile<f32, { [1] }> = dtab + dtb;
            // stable softplus: x>20 ? x : log1p(exp(x))
            let e: Tile<f32, { [1] }> = exp(raw);
            let sp: Tile<f32, { [1] }> = log(e + broadcast_scalar(1.0f32, shape![1]));
            let dt: Tile<f32, { [1] }> = select(
                raw > broadcast_scalar(20.0f32, shape![1]),
                raw,
                sp,
            );
            let dt: Tile<f32, { [1] }> = select(
                dt < broadcast_scalar(dt_min, shape![1]),
                broadcast_scalar(dt_min, shape![1]),
                select(dt > broadcast_scalar(dt_max, shape![1]), broadcast_scalar(dt_max, shape![1]), dt),
            );
            // dA = exp(-exp(A_log[h]) * dt)
            let alog: Tile<f32, { [1] }> = a_log.load_tile(shape![1], [head]);
            let da: Tile<f32, { [1] }> = exp(-exp(alog) * dt);

            // Load B/C for this group: B[g, :], C[g, :] — [1, S]
            let b_t: Tile<f32, { [1, S] }> =
                convert_tile(b_proj.load_tile(shape![1, S], [b * n_groups as i32 + group]));
            let c_t: Tile<f32, { [1, S] }> =
                convert_tile(c_proj.load_tile(shape![1, S], [b * n_groups as i32 + group]));
            // x row for this (b, head): x[b, head, :] — [1, HD]
            let x_row: Tile<f32, { [1, HD] }> =
                convert_tile(x.load_tile(shape![1, HD], [b * num_heads as i32 + head]));

            // State update: H = dA*H + dt * x ⊗ B  →  [HD, S]
            let h_old: Tile<f32, { [HD, S] }> = h_state.load_tile(shape![HD, S], [head, b]);
            let h_new: Tile<f32, { [HD, S] }> = h_old * da.reshape(shape![1, 1]).broadcast(shape![HD, S])
                + dt.reshape(shape![1, 1]).broadcast(shape![HD, S])
                    * x_row.reshape(shape![HD, 1]).broadcast(shape![HD, S])
                    * b_t.reshape(shape![1, S]).broadcast(shape![HD, S]);
            state.store(h_new);

            // Output: y[hd] = sum_s(H*C) + D*x
            let hc: Tile<f32, { [HD, S] }> = h_new * c_t.reshape(shape![1, S]).broadcast(shape![HD, S]);
            let y: Tile<f32, { [HD, 1] }> = reduce_sum(hc, 1);
            let d_v: Tile<f32, { [1] }> = d_param.load_tile(shape![1], [head]);
            let y: Tile<f32, { [HD, 1] }> =
                y + d_v.reshape(shape![1, 1]).broadcast(shape![HD, 1])
                    * x_row.reshape(shape![HD, 1]).broadcast(shape![HD, 1]);
            out.store(convert_tile(y));
        }
    }

    pub(super) fn cutile_mamba2_launch(
        _h_state: DevicePtr,
        _x: DevicePtr,
        _b_proj: DevicePtr,
        _c_proj: DevicePtr,
        _dt_raw: DevicePtr,
        _a_log: DevicePtr,
        _d_param: DevicePtr,
        _dt_bias: DevicePtr,
        _output: DevicePtr,
        _batch_size: u32,
        _num_heads: u32,
        head_dim: u32,
        state_size: u32,
        _n_groups: u32,
        _dt_min: f32,
        _dt_max: f32,
    ) -> Result<()> {
        if !head_dim.is_power_of_two() || head_dim > 256 {
            bail!("cutile mamba2: head_dim={head_dim} must be a power of two <= 256");
        }
        if !state_size.is_power_of_two() || state_size > 256 {
            bail!("cutile mamba2: state_size={state_size} must be a power of two <= 256 (Puzzle's 96 needs a chunked path — see PR body)");
        }
        // The rank-1 outer-product state update + state-sized reduction need
        // 2-D tile broadcast semantics (x ⊗ B over [HD, S]) that are not yet
        // verified against cutile 0.4's broadcast rules. Fail loudly rather
        // than run a partial kernel.
        bail!(
            "ATLAS_CUTILE_MAMBA2_DECODE=1: cutile mamba2 kernel not yet verified \
             (2-D outer-product state update pending); unset the flag"
        )
    }
}

#[cfg(feature = "cutile")]
fn cutile_mamba2_launch(
    h_state: DevicePtr,
    x: DevicePtr,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    dt_raw: DevicePtr,
    a_log: DevicePtr,
    d_param: DevicePtr,
    dt_bias: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    dt_min: f32,
    dt_max: f32,
) -> Result<()> {
    imp::cutile_mamba2_launch(
        h_state, x, b_proj, c_proj, dt_raw, a_log, d_param, dt_bias, output,
        batch_size, num_heads, head_dim, state_size, n_groups, dt_min, dt_max,
    )
}
