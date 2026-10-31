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
        Ok(Mamba2DecodeOp::Registry(
            gpu.kernel("mamba2_ssm", "mamba2_ssm_decode")?,
        ))
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
                    gpu, h_state, x, b_proj, c_proj, dt_raw, a_log, d_param, dt_bias, output,
                    batch_size, num_heads, head_dim, state_size, n_groups, dt_min, dt_max,
                )
            }
            #[cfg(not(feature = "cutile"))]
            {
                let _ = (
                    h_state, x, b_proj, c_proj, dt_raw, a_log, d_param, dt_bias, output,
                    batch_size, num_heads, head_dim, state_size, n_groups, dt_min, dt_max, stream,
                );
                bail!("ATLAS_CUTILE_MAMBA2_DECODE=1 requires building with --features cutile")
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

    /// One tile per state row. A whole [HD, S] tile mis-broadcast x across rows
    /// on cutile 0.4, so each block owns one row of length S.
    #[cutile::module]
    mod mamba2 {
        use cutile::core::*;

        #[cutile::entry]
        pub fn mamba2_row<const S: i32>(
            h_in: &Tensor<f32, { [-1, -1] }>,
            x: &Tensor<bf16, { [-1] }>,
            b_proj: &Tensor<bf16, { [-1, -1] }>,
            c_proj: &Tensor<bf16, { [-1, -1] }>,
            dt_raw: &Tensor<bf16, { [-1] }>,
            a_log: &Tensor<f32, { [-1] }>,
            d_param: &Tensor<f32, { [-1] }>,
            dt_bias: &Tensor<f32, { [-1] }>,
            h_out: &mut Tensor<f32, { [1, S] }>,
            y_out: &mut Tensor<bf16, { [1] }>,
            hd: i32,
            heads: i32,
            groups: i32,
            dt_min: f32,
            dt_max: f32,
        ) {
            let pid: (i32, i32, i32) = get_tile_block_id();
            let row = pid.0;
            let token_head = row / hd;
            let head = token_head % heads;
            let b = token_head / heads;
            let group = head / (heads / groups);

            let h_old: Tile<f32, { [1, S] }> = load_tile_like(h_in, h_out);
            let xs: Tile<f32, { [1] }> = convert_tile(x.load_tile(shape![1], [row]));
            let b_row: Tile<f32, { [1, S] }> =
                convert_tile(b_proj.load_tile(shape![1, S], [b * groups + group, 0]));
            let c_row: Tile<f32, { [1, S] }> =
                convert_tile(c_proj.load_tile(shape![1, S], [b * groups + group, 0]));

            let raw: Tile<f32, { [1] }> = convert_tile(dt_raw.load_tile(shape![1], [token_head]))
                + dt_bias.load_tile(shape![1], [head]);
            let sp: Tile<f32, { [1] }> = log(exp(raw) + broadcast_scalar(1.0f32, shape![1]));
            let dt: Tile<f32, { [1] }> =
                select(gt_tile(raw, broadcast_scalar(20.0f32, shape![1])), raw, sp);
            let dt: Tile<f32, { [1] }> = select(
                lt_tile(dt, broadcast_scalar(dt_min, shape![1])),
                broadcast_scalar(dt_min, shape![1]),
                dt,
            );
            let dt: Tile<f32, { [1] }> = select(
                gt_tile(dt, broadcast_scalar(dt_max, shape![1])),
                broadcast_scalar(dt_max, shape![1]),
                dt,
            );
            let da: Tile<f32, { [1] }> = exp(negf(exp(a_log.load_tile(shape![1], [head]))) * dt);

            let da_w: Tile<f32, { [1, S] }> = da.reshape(shape![1, 1]).broadcast(shape![1, S]);
            let dt_w: Tile<f32, { [1, S] }> = dt.reshape(shape![1, 1]).broadcast(shape![1, S]);
            let x_w: Tile<f32, { [1, S] }> = xs.reshape(shape![1, 1]).broadcast(shape![1, S]);
            let h_new: Tile<f32, { [1, S] }> = da_w * h_old + x_w * dt_w * b_row;
            let lo = broadcast_scalar(-200.0f32, shape![1, S]);
            let hi = broadcast_scalar(200.0f32, shape![1, S]);
            let h_new: Tile<f32, { [1, S] }> = select(lt_tile(h_new, lo), lo, h_new);
            let h_new: Tile<f32, { [1, S] }> = select(gt_tile(h_new, hi), hi, h_new);
            h_out.store(h_new);

            let summed: Tile<f32, { [1] }> = reduce_sum(h_new * c_row, 1);
            let y: Tile<f32, { [1] }> = summed + d_param.load_tile(shape![1], [head]) * xs;
            let yb: Tile<bf16, { [1] }> = convert_tile(y);
            y_out.store(yb);
        }
    }

    pub(super) fn cutile_mamba2_launch(
        gpu: &dyn GpuBackend,
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
        if !head_dim.is_power_of_two() || head_dim > 256 {
            bail!("cutile mamba2: head_dim={head_dim} must be a power of two <= 256");
        }
        if !state_size.is_power_of_two() || state_size > 256 {
            bail!("cutile mamba2: state_size={state_size} must be a power of two <= 256");
        }
        if n_groups == 0 || num_heads % n_groups != 0 {
            bail!("cutile mamba2: num_heads={num_heads} must divide by n_groups={n_groups}");
        }
        let batch = batch_size as i32;
        let heads = num_heads as i32;
        let hd = head_dim as i32;
        let ss = state_size as i32;
        let groups = n_groups as i32;
        let nrows = batch * heads * hd;
        let bytes = (nrows * ss) as usize * std::mem::size_of::<f32>();
        let staged = gpu.alloc(bytes)?;
        gpu.copy_d2d(h_state, staged, bytes)?;
        let h_in =
            unsafe { Tensor::<f32>::borrow_raw_parts(staged.0, 0, vec![nrows, ss], vec![ss, 1]) };
        let mut h_out =
            unsafe { Tensor::<f32>::borrow_raw_parts(h_state.0, 0, vec![nrows, ss], vec![ss, 1]) };
        let x_t = unsafe { Tensor::<bf16>::borrow_raw_parts(x.0, 0, vec![nrows], vec![1]) };
        let b_t = unsafe {
            Tensor::<bf16>::borrow_raw_parts(b_proj.0, 0, vec![batch * groups, ss], vec![ss, 1])
        };
        let c_t = unsafe {
            Tensor::<bf16>::borrow_raw_parts(c_proj.0, 0, vec![batch * groups, ss], vec![ss, 1])
        };
        let dt_t =
            unsafe { Tensor::<bf16>::borrow_raw_parts(dt_raw.0, 0, vec![batch * heads], vec![1]) };
        let a_t = unsafe { Tensor::<f32>::borrow_raw_parts(a_log.0, 0, vec![heads], vec![1]) };
        let d_t = unsafe { Tensor::<f32>::borrow_raw_parts(d_param.0, 0, vec![heads], vec![1]) };
        let bias_t = unsafe { Tensor::<f32>::borrow_raw_parts(dt_bias.0, 0, vec![heads], vec![1]) };
        let mut y = unsafe { Tensor::<bf16>::borrow_raw_parts(output.0, 0, vec![nrows], vec![1]) };
        let op = mamba2::mamba2_row(
            &h_in,
            &x_t,
            &b_t,
            &c_t,
            &dt_t,
            &a_t,
            &d_t,
            &bias_t,
            (&mut h_out).partition([1, ss as usize]),
            (&mut y).partition([1]),
            hd,
            heads,
            groups,
            dt_min,
            dt_max,
        )
        .generics(vec![ss.to_string()]);
        let _staged = staged;
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
            .map_err(|e| anyhow::anyhow!("cutile mamba2 launch: {e}"))
    }
}

#[cfg(feature = "cutile")]
fn cutile_mamba2_launch(
    gpu: &dyn GpuBackend,
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
        gpu, h_state, x, b_proj, c_proj, dt_raw, a_log, d_param, dt_bias, output, batch_size,
        num_heads, head_dim, state_size, n_groups, dt_min, dt_max,
    )
}
