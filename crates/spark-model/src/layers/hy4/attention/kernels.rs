// SPDX-License-Identifier: AGPL-3.0-only

use super::{cache::AttentionCache, loader::Hy4Attention};
use anyhow::Result;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};

/// DSA layers: non-circular paged cache score-key cap.
pub(super) const INDEX_KEYS: usize = 2048;

pub(super) struct Kernels {
    pub linear: KernelHandle,
    pub fp8: KernelHandle,
    pub join: KernelHandle,
    pub norm: KernelHandle,
    pub gemv: KernelHandle,
    pub rotary: KernelHandle,
    pub gate: KernelHandle,
    pub attention: KernelHandle,
    pub attention_fp8: KernelHandle,
    pub kv_norm_fp8: KernelHandle,
    pub rotary_fp8: KernelHandle,
    pub merge: KernelHandle,
    pub pool: KernelHandle,
    pub fwht: KernelHandle,
    pub score: KernelHandle,
    pub fwht_batch: KernelHandle,
    pub score_batch: KernelHandle,
    pub select_batch: KernelHandle,
    pub tail_norm_batch: KernelHandle,
    pub pool_fwht_batch: KernelHandle,
}
impl Kernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            linear: gpu.kernel("hy4_ffn", "hy4_linear")?,
            fp8: gpu.kernel("hy4_mla", "hy4_fp8_gemv")?,
            join: gpu.kernel("hy4_mla", "hy4_join_qkv")?,
            norm: gpu.kernel("hy4_ffn", "hy4_norm_add")?,
            gemv: gpu.kernel("hy4_mla", "hy4_head_gemv")?,
            rotary: gpu.kernel("hy4_mla", "hy4_rotary")?,
            gate: gpu.kernel("hy4_mla", "hy4_gate")?,
            attention: gpu.kernel("hy4_mla", "hy4_mla_partial")?,
            attention_fp8: gpu.kernel("hy4_mla", "hy4_mla_partial_fp8kv")?,
            kv_norm_fp8: gpu.kernel("hy4_mla", "hy4_kv_norm_fp8kv")?,
            rotary_fp8: gpu.kernel("hy4_mla", "hy4_rotary_fp8kv")?,
            merge: gpu.kernel("hy4_mla", "hy4_mla_merge")?,
            pool: gpu.kernel("hy4_indexpool", "hy4_pool")?,
            fwht: gpu.kernel("hy4_indexpool", "hy4_fwht")?,
            score: gpu.kernel("hy4_indexpool", "hy4_index_scores")?,
            fwht_batch: gpu.kernel("hy4_indexpool", "hy4_fwht_batch")?,
            score_batch: gpu.kernel("hy4_indexpool", "hy4_index_scores_batch")?,
            select_batch: gpu.kernel("hy4_indexpool", "hy4_index_select_batch")?,
            tail_norm_batch: gpu.kernel("hy4_indexpool", "hy4_tail_norm_batch")?,
            pool_fwht_batch: gpu.kernel("hy4_indexpool", "hy4_pool_fwht_batch")?,
        })
    }
}
impl Hy4Attention<'_> {
    /// Batched latent norm-write kernel for this cache dtype.
    pub(super) fn batch_kv_norm_kernel(&self) -> &'static str {
        if self.kv_fp8 {
            "hy4_batch_kv_norm_fp8kv"
        } else {
            "hy4_batch_kv_norm"
        }
    }
    /// Paged latent rotary-write kernel for this cache dtype.
    pub(super) fn batch_kv_rotary_kernel(&self) -> &'static str {
        if self.kv_fp8 {
            "hy4_batch_rotary_fp8kv"
        } else {
            "hy4_batch_rotary"
        }
    }
    /// Batched MLA partial kernel (scalar or tensor-core) for this cache dtype.
    pub(super) fn batch_mla_kernel(&self, mma: bool) -> &'static str {
        match (mma, self.kv_fp8) {
            (true, false) => "hy4_mla_batch_partial_mma",
            (true, true) => "hy4_mla_batch_partial_mma_fp8kv",
            (false, false) => "hy4_mla_batch_partial",
            (false, true) => "hy4_mla_batch_partial_fp8kv",
        }
    }
}
impl Hy4Attention<'_> {
    pub(super) fn linear(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        if self.scales.contains_key(&w) {
            return self.fp8(x, w, y, [n, k, 0, 0], 1, stream);
        }
        KernelLaunch::new(self.memory.gpu, self.kernels.linear)
            .grid([n.div_ceil(8) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(y)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(0)
            .arg_u32(0)
            .launch(stream)
    }
    pub(super) fn norm(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        let _phase = crate::layers::hy4::timing::span("attn_norm", stream)?;
        KernelLaunch::new(self.memory.gpu, self.kernels.norm)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(n as u32)
            .arg_f32(self.eps)
            .launch(stream)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rotary(
        &self,
        x: DevicePtr,
        y: DevicePtr,
        heads: usize,
        layout: [usize; 4],
        position: [i64; 3],
        split: bool,
        stream: u64,
    ) -> Result<()> {
        let _phase = crate::layers::hy4::timing::span("attn_rope", stream)?;
        let [input_stride, output_stride, input_offset, output_offset] = layout;
        KernelLaunch::new(self.memory.gpu, self.kernels.rotary)
            .grid([heads as u32, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(y)
            .arg_u32(input_stride as u32)
            .arg_u32(output_stride as u32)
            .arg_u32(input_offset as u32)
            .arg_u32(output_offset as u32)
            .arg_u64(position[0] as u64)
            .arg_u64(position[1] as u64)
            .arg_u64(position[2] as u64)
            .arg_f32(self.rope.rope_theta as f32)
            .arg_u32(self.rope.mrope_section[1] as u32)
            .arg_u32(self.rope.mrope_section[2] as u32)
            .arg_u32(u32::from(self.rope.mrope_interleaved))
            .arg_u32(u32::from(split))
            .launch(stream)
    }
    pub(super) fn gemv(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        dimensions: [usize; 4],
        stream: u64,
    ) -> Result<()> {
        let [n, k, input_stride, output_stride] = dimensions;
        if self.scales.contains_key(&w) {
            return self.fp8(x, w, y, dimensions, self.heads, stream);
        }
        KernelLaunch::new(self.memory.gpu, self.kernels.gemv)
            .grid([n.div_ceil(8) as u32, self.heads as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(y)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(input_stride as u32)
            .arg_u32(output_stride as u32)
            .launch(stream)
    }
    #[allow(clippy::too_many_arguments)]
    fn fp8(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        dimensions: [usize; 4],
        heads: usize,
        stream: u64,
    ) -> Result<()> {
        let [n, k, input_stride, output_stride] = dimensions;
        KernelLaunch::new(self.memory.gpu, self.kernels.fp8)
            .grid([n.div_ceil(8) as u32, heads as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(self.scales[&w])
            .arg_ptr(y)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(input_stride as u32)
            .arg_u32(output_stride as u32)
            .launch(stream)
    }
}

impl AttentionCache<'_, '_> {
    /// Append the fp8 rowwise-scale arg to a `_fp8kv` launch when this cache
    /// stores fp8 latents; the BF16 twin keeps its exact original arg list.
    fn with_kv_scales<'a>(&self, launch: KernelLaunch<'a>) -> KernelLaunch<'a> {
        if self.layer.kv_fp8 {
            launch.arg_ptr(self.latent_scales)
        } else {
            launch
        }
    }

    /// Batched causal-tile latent norm write (slot dims 0..512).
    pub(super) fn launch_tile_kv_norm(
        &self,
        kv: DevicePtr,
        batched: usize,
        start_position: usize,
        stream: u64,
    ) -> Result<()> {
        let a = self.layer;
        let gpu = a.memory.gpu;
        self.with_kv_scales(
            KernelLaunch::new(
                gpu,
                gpu.op_cache()
                    .kernel(gpu, "hy4_mla", a.batch_kv_norm_kernel())?,
            )
            .grid([batched as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(kv)
            .arg_ptr(a.kvn)
            .arg_ptr(self.latent)
            .arg_ptr(self.table)
            .arg_u32(start_position as u32)
            .arg_u32(self.layout.page_size as u32)
            .arg_u32(self.layout.table.len() as u32)
            .arg_u32(u32::from(a.window.is_some()))
            .arg_f32(a.eps),
        )
        .launch(stream)
    }

    /// Paged latent rotary write (slot dims 512..576) for a causal tile.
    pub(super) fn launch_tile_kv_rotary(
        &self,
        kv: DevicePtr,
        batched: usize,
        start_position: usize,
        base: [i64; 3],
        stream: u64,
    ) -> Result<()> {
        let a = self.layer;
        let gpu = a.memory.gpu;
        self.with_kv_scales(
            KernelLaunch::new(
                gpu,
                gpu.op_cache()
                    .kernel(gpu, "hy4_mla", a.batch_kv_rotary_kernel())?,
            )
            .grid([1, batched as u32, 1])
            .block([32, 1, 1])
            .arg_ptr(kv)
            .arg_ptr(self.latent)
            .arg_u32(576)
            .arg_u32(576)
            .arg_u32(512)
            .arg_u32(512)
            .arg_u32(576)
            .arg_u32(576)
            .arg_ptr(self.table)
            .arg_u32(start_position as u32)
            .arg_u32(self.layout.page_size as u32)
            .arg_u32(self.layout.table.len() as u32)
            .arg_u32(u32::from(a.window.is_some()))
            .arg_u32(1)
            .arg_u64(base[0] as u64)
            .arg_u64(base[1] as u64)
            .arg_u64(base[2] as u64)
            .arg_f32(a.rope.rope_theta as f32)
            .arg_u32(a.rope.mrope_section[1] as u32)
            .arg_u32(a.rope.mrope_section[2] as u32)
            .arg_u32(u32::from(a.rope.mrope_interleaved))
            .arg_u32(0),
        )
        .launch(stream)
    }

    /// Batched MLA partial over a causal tile (scalar or tensor-core kernel).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_tile_mla(
        &self,
        absorbed: DevicePtr,
        indices: DevicePtr,
        softmax_parts: DevicePtr,
        rows: usize,
        start_position: usize,
        count: usize,
        splits: usize,
        mma: bool,
        stream: u64,
    ) -> Result<()> {
        let a = self.layer;
        let gpu = a.memory.gpu;
        let (grid, smem) = if mma {
            (
                [splits as u32, rows as u32, a.heads.div_ceil(16) as u32],
                100544,
            )
        } else {
            ([a.heads as u32, splits as u32, rows as u32], 0)
        };
        self.with_kv_scales(
            KernelLaunch::new(
                gpu,
                gpu.op_cache()
                    .kernel(gpu, "hy4_mla", a.batch_mla_kernel(mma))?,
            )
            .grid(grid)
            .block([256, 1, 1])
            .shared_mem(smem)
            .arg_ptr(absorbed)
            .arg_ptr(self.latent)
            .arg_ptr(self.table)
            .arg_ptr(if a.window.is_some() {
                DevicePtr::NULL
            } else {
                indices
            })
            .arg_ptr(softmax_parts)
            .arg_u32(start_position as u32)
            .arg_u32(self.layout.page_size as u32)
            .arg_u32(self.layout.table.len() as u32)
            .arg_u32(a.window.unwrap_or(0) as u32)
            .arg_u32(count as u32)
            .arg_u32(splits as u32)
            .arg_u32(a.heads as u32)
            .arg_u32(INDEX_KEYS as u32),
        )
        .launch(stream)
    }
}
