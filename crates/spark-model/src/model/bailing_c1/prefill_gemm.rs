// SPDX-License-Identifier: AGPL-3.0-only

//! The Ling prefill BF16 projection lane shared by the family driver.
//!
//! Small-M cuBLASLt heuristics can pick a split-K kernel whose workspace /
//! reduction behavior corrupted Ling projections in serve traffic (PR #131).
//! The lane here therefore accepts only the runtime's explicit split-K-free
//! workspace-free cuBLASLt kernels and otherwise falls back to the scalar
//! GEMM. It never uses cuBLASLt's heuristic first pick.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::LingKernels;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// `ATLAS_LING_PROJ_GEMM_MIN_ROWS`: the smallest chunk the prefill
/// projections take the BF16 tensor-core lane for. Default 16; `0` restores
/// the scalar-only floor. Read once.
fn proj_gemm_min_rows() -> u32 {
    static MIN: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        proj_gemm_min_rows_from(
            std::env::var("ATLAS_LING_PROJ_GEMM_MIN_ROWS")
                .ok()
                .as_deref(),
        )
    })
}

/// The threshold resolution, split out for the unit test. `None` = env
/// absent (default 16); `Some("0")` = the scalar-only floor (u32::MAX);
/// `Some(other)` = that row count, clamped to >= 1.
fn proj_gemm_min_rows_from(v: Option<&str>) -> u32 {
    match v {
        None => 16,
        Some("0") => u32::MAX,
        Some(s) => s.parse::<u32>().ok().unwrap_or(16).max(1),
    }
}

/// One prefill-shaped projection: the explicit split-K-free workspace-free
/// BF16 tensor-core lane from the M threshold up, the scalar GEMM below it or
/// when no safe kernel exists for the shape.
#[allow(clippy::too_many_arguments)]
pub(crate) fn proj_gemm(
    gpu: &dyn GpuBackend,
    k: &LingKernels,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k_dim: u32,
    stream: u64,
) -> Result<()> {
    if rows >= proj_gemm_min_rows() {
        let lt_ok = spark_runtime::cublaslt::bf16_gemm_act_weight_t_in_order_no_workspace(
            input.0,
            weight.weight.0,
            output.0,
            [rows, n, k_dim],
            stream,
        )
        .unwrap_or_else(|e| {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    "ling prefill TC lane unavailable ({e}); projections fall back to the scalar path"
                );
            }
            false
        });
        if lt_ok {
            return Ok(());
        }
    }
    ops::dense_gemm(gpu, k.gemm_k, input, weight, output, rows, n, k_dim, stream)
}

#[cfg(test)]
mod tests {
    use super::proj_gemm_min_rows_from;

    #[test]
    fn proj_gemm_threshold_resolution() {
        assert_eq!(proj_gemm_min_rows_from(None), 16);
        assert_eq!(proj_gemm_min_rows_from(Some("0")), u32::MAX);
        assert_eq!(proj_gemm_min_rows_from(Some("16")), 16);
        assert_eq!(proj_gemm_min_rows_from(Some("3")), 3);
        assert_eq!(proj_gemm_min_rows_from(Some("8192")), 8192);
        assert_eq!(proj_gemm_min_rows_from(Some("not-a-number")), 16);
    }

    #[test]
    fn ffn_lane_is_safe_never_the_heuristic_first_pick() {
        let helper = include_str!("prefill_gemm.rs")
            .splitn(2, "#[cfg(test)]")
            .next()
            .unwrap();
        assert!(helper.contains("bf16_gemm_act_weight_t_in_order_no_workspace"));
        assert!(!helper.contains("bf16_gemm_act_weight_t("));
        let ffn = include_str!("ffn.rs");
        assert!(
            !ffn.contains("ops::bf16_gemm("),
            "Ling FFN wide GEMMs must use prefill_gemm::proj_gemm, not the heuristic first-pick lane"
        );
    }
}
