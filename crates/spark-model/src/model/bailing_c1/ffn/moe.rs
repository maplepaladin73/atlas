// SPDX-License-Identifier: AGPL-3.0-only
//! Routed-MoE forward pass, split out of `ffn.rs` for the file-size cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::{LingFfn, row_batches, shared_down, shared_gate_up};

impl LingFfn {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn forward_moe(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let w = self.moe.as_ref().expect("moe weights");
        let k = &self.kernels;
        let s = &self.scratch;
        // Routed experts over full chunks: the tile kernels stream every
        // live expert's tables per pass, so splitting the chunk re-reads
        // the expert tables per split (at rows=512 the old 32-row splits
        // re-read them 16x). The worklist and all row-indexed staging are
        // sized for `scratch.rows`, which bounds chunks here.
        for (start, batch) in row_batches(rows, self.scratch.rows as u32) {
            let batch = batch as u32;
            let base = start as usize;
            let xb = x.offset(base * self.hidden * 2);
            // 1. route + input quant + group-limited top-8 + sort + worklist.
            // Stream-ordered reset: the capturing decode/verify graphs bake
            // this memset as a graph node (a graph-captured fill goes
            // through `memset_async` per the GpuBackend contract), whereas
            // `memset` synchronizes the stream — illegal inside a capture.
            gpu.memset_async(s.moe_done, 0, 4, stream)?;
            // Router logits for wide batches come from a cuBLASLt TF32
            // tensor-core GEMM (default on, ATLAS_LING_ROUTER_TF32=0 opts
            // out): the fused kernel's warp-per-expert GEMV re-streams the
            // activations per expert row and serializes rows within a warp,
            // which is the wrong dataflow for a [rows, 2560] x [2560, 512]
            // GEMM. Narrow (decode-width) batches keep the scalar GEMV --
            // it is the right shape there and stays outside cuBLASLt for
            // the decode graph captures.
            let router_tf32 = batch >= 32
                && !matches!(std::env::var("ATLAS_LING_ROUTER_TF32").as_deref(), Ok("0"));
            if router_tf32 {
                let total = batch * self.hidden as u32;
                let xfp = s.moe_xfp32.offset(base * self.hidden * 4);
                KernelLaunch::new(gpu, k.moe_xwiden_k)
                    .grid([div_ceil(total, 1024), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(xb)
                    .arg_ptr(xfp)
                    .arg_u32(total)
                    .launch(stream)?;
                spark_runtime::cublaslt::tf32_gemm_act_weight_t(
                    xfp.0,
                    w.router.0,
                    s.moe_logits.offset(base * self.experts * 4).0,
                    batch,
                    self.experts as u32,
                    self.hidden as u32,
                    stream,
                )?;
            }
            let router_ctas = if router_tf32 {
                0
            } else {
                div_ceil(self.experts as u32, 8)
            };
            let groups = batch * (self.hidden as u32 / 16);
            let quant_ctas = div_ceil(groups, 256);
            KernelLaunch::new(gpu, k.moe_route_k)
                .grid([router_ctas + quant_ctas, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(xb)
                .arg_ptr(w.router)
                .arg_ptr(w.bias)
                .arg_f32(w.a_scale2)
                .arg_f32(1.0 / w.a_scale2)
                .arg_ptr(s.moe_logits.offset(base * self.experts * 4))
                .arg_ptr(s.a_packed.offset(base * (self.hidden / 2)))
                .arg_ptr(s.a_scale.offset(base * (self.hidden / 16)))
                .arg_ptr(s.topk_ids.offset(base * self.top_k * 4))
                .arg_ptr(s.topk_weights.offset(base * self.top_k * 4))
                .arg_ptr(s.sorted_tokens.offset(base * self.top_k * 4))
                .arg_ptr(s.token_to_perm.offset(base * self.top_k * 4))
                .arg_ptr(s.expert_offsets)
                .arg_ptr(s.worklist)
                .arg_u32(self.max_items(rows) as u32)
                .arg_u32(16)
                .arg_u32(batch)
                .arg_u32(self.experts as u32)
                .arg_u32(self.hidden as u32)
                .arg_u32(self.top_k as u32)
                .arg_u32(self.n_group as u32)
                .arg_u32(self.topk_group as u32)
                .arg_f32(self.scaling)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(s.moe_done)
                .arg_u32(u32::from(router_tf32))
                .launch(stream)?;
            // 1b. grid-parallel selection (one row per warp across
            // ceil(batch/8) CTAs) + the single-CTA sort/worklist tail --
            // bit-identical to the fused kernel's trailing CTA, but the
            // selection scan no longer serializes the chunk on 8 warps.
            KernelLaunch::new(gpu, k.moe_route_select_k)
                .grid([div_ceil(batch, 8), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(s.moe_logits.offset(base * self.experts * 4))
                .arg_ptr(w.bias)
                .arg_ptr(s.topk_ids.offset(base * self.top_k * 4))
                .arg_ptr(s.topk_weights.offset(base * self.top_k * 4))
                .arg_ptr(s.sorted_tokens.offset(base * self.top_k * 4))
                .arg_ptr(s.token_to_perm.offset(base * self.top_k * 4))
                .arg_ptr(s.expert_offsets)
                .arg_ptr(s.worklist)
                .arg_u32(self.max_items(rows) as u32)
                .arg_u32(16)
                .arg_u32(batch)
                .arg_u32(self.experts as u32)
                .arg_u32(self.top_k as u32)
                .arg_u32(self.n_group as u32)
                .arg_u32(self.topk_group as u32)
                .arg_f32(self.scaling)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(s.moe_done)
                .launch(stream)?;
            // 2. routed experts: fused gate/up/swiglu/re-quant, then down.
            let items = self.max_items(batch as usize) as u32;
            KernelLaunch::new(gpu, k.moe_gate_up_k)
                .grid([div_ceil(self.inter as u32, 128), items, 1])
                .block([256, 1, 1])
                .arg_ptr(s.a_packed.offset(base * (self.hidden / 2)))
                .arg_ptr(s.a_scale.offset(base * (self.hidden / 16)))
                .arg_ptr(w.gate.packed_ptrs)
                .arg_ptr(w.gate.scale_ptrs)
                .arg_ptr(w.gate.scale2_vals)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(s.expert_offsets)
                .arg_ptr(s.sorted_tokens.offset(base * self.top_k * 4))
                .arg_u32(self.experts as u32)
                .arg_u32(self.inter as u32)
                .arg_u32(self.hidden as u32)
                .arg_ptr(s.worklist)
                .arg_f32(w.swiglu_routed)
                .arg_ptr(w.up.packed_ptrs)
                .arg_ptr(w.up.scale_ptrs)
                .arg_ptr(w.up.scale2_vals)
                .arg_ptr(s.gu_packed.offset(base * self.top_k * (self.inter / 2)))
                .arg_ptr(s.gu_scale.offset(base * self.top_k * (self.inter / 16)))
                .arg_ptr(w.a2_gscale)
                .launch(stream)?;
            KernelLaunch::new(gpu, k.moe_down_k)
                .grid([div_ceil(self.hidden as u32, 256), items, 1])
                .block([256, 1, 1])
                .arg_ptr(s.gu_packed.offset(base * self.top_k * (self.inter / 2)))
                .arg_ptr(s.gu_scale.offset(base * self.top_k * (self.inter / 16)))
                .arg_ptr(w.down.packed_ptrs)
                .arg_ptr(w.down.scale_ptrs)
                .arg_ptr(w.down.scale2_vals)
                .arg_ptr(s.expert_out.offset(base * self.top_k * self.hidden * 2))
                .arg_ptr(s.expert_offsets)
                .arg_ptr(DevicePtr::NULL)
                .arg_u32(self.experts as u32)
                .arg_u32(self.hidden as u32)
                .arg_u32(self.inter as u32)
                .arg_ptr(s.worklist)
                .launch(stream)?;
        }
        // 3. shared expert (BF16, its own clamp list): bucket pass below 16
        // rows; wide chunks run the GEMMs on tensor cores, weights read
        // once. Its down rows land in `out` directly.
        if rows >= 16 {
            self.cublas_swiglu(
                gpu,
                x,
                &w.shared_gate,
                &w.shared_up,
                &w.shared_down,
                s.shared_h,
                out,
                rows,
                w.swiglu_shared,
                stream,
            )?;
        } else {
            for (start, batch) in row_batches(rows, 32) {
                let batch = batch as u32;
                let base = start as usize;
                let xb = x.offset(base * self.hidden * 2);
                let yb = out.offset(base * self.hidden * 2);
                let hb = s.shared_h.offset(base * self.inter * 2);
                shared_gate_up(
                    gpu,
                    k,
                    xb,
                    &w.shared_gate,
                    &w.shared_up,
                    hb,
                    batch,
                    self.inter,
                    self.hidden,
                    w.swiglu_shared,
                    stream,
                )?;
                shared_down(
                    gpu,
                    k,
                    hb,
                    &w.shared_down,
                    yb,
                    batch,
                    self.hidden,
                    self.inter,
                    stream,
                )?;
            }
        }
        // 4. unpermute: routed reduce + shared sum -> y. The shared rows
        // already sit at yb (the shared down wrote them), and the
        // kernel reads each shared element before writing the same
        // element of Y, so Y = shared_out = yb aliases safely.
        KernelLaunch::new(gpu, k.moe_unpermute_k)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.expert_out)
            .arg_ptr(out)
            .arg_ptr(s.token_to_perm)
            .arg_ptr(s.topk_weights)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(out)
            .arg_u32(rows as u32)
            .arg_u32(self.hidden as u32)
            .arg_u32(self.top_k as u32)
            .launch(stream)
    }
}
