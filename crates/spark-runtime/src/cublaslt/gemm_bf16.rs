// SPDX-License-Identifier: AGPL-3.0-only
//! The shared cuBLASLt BF16 GEMM body (split out of `cublaslt.rs` for the
//! 500-LoC cap).

use super::*;
use anyhow::ensure;

// CUDA 13 cublasLt.h: CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK and the
// cublasLtReductionScheme_t values NONE and MASK (every scheme, the default).
const PREF_REDUCTION_SCHEME_MASK: u32 = 3;
const SCHEME_NONE: u32 = 0;
const SCHEME_ALL: u32 = 7;

/// `ATLAS_LT_NO_SPLIT_K=1`: pin the heuristic to reduction scheme NONE
/// (split-K 1, in-order accumulation). The default heuristic otherwise
/// picks split-K kernels for small-M shapes (e.g. o 64x2560x4096 at
/// splitK=8, numerically ~2.5x worse than the in-order reference); a shape
/// with no in-order algorithm runs cuBLASLt's own choice instead.
fn no_split_k() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_LT_NO_SPLIT_K").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// `heuristic(mask)` for the split-K 1 pin, else (or when the pinned query
/// finds no algorithm) for cuBLASLt's own choice. `None` leaves the
/// preference's reduction mask untouched, exactly as without the pin.
fn pinned_or_default<T>(
    no_split_k: bool,
    mut heuristic: impl FnMut(Option<u32>) -> Result<T>,
) -> Result<T> {
    if no_split_k && let Ok(algo) = heuristic(Some(SCHEME_NONE)) {
        return Ok(algo);
    }
    heuristic(no_split_k.then_some(SCHEME_ALL))
}

/// Shared body. `op_a` selects the weight's stored layout: `CUBLAS_OP_T` for a
/// row-major `[N,K]` weight, `CUBLAS_OP_N` for a row-major `[K,N]` one. The A
/// LAYOUT MUST MATCH: `(k, n, ld=k)` under opT, `(n, k, ld=n)` under opN.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_bf16(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    op_a: i32,
    stream: u64,
) -> Result<()> {
    let pick = if kchain_pin::applies(m, n, op_a == CUBLAS_OP_T) {
        kchain_pin::Pick::Pin
    } else {
        kchain_pin::Pick::First
    };
    gemm_bf16_pick(act, weight, out, [m, n, k], op_a, pick, CUDA_R_16BF, stream).map(|_| ())
}

/// [`gemm_bf16`] with the algorithm choice explicit (`kchain_pin::Pick`) and
/// the output element `out_type` (`CUDA_R_16BF`, or `CUDA_R_32F` for the
/// FP32 accumulator unrounded); `Ok(false)`: `Pick::Force` found no k-chain
/// kernel and launched nothing.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_bf16_pick(
    act: u64,
    weight: u64,
    out: u64,
    [m, n, k]: [u32; 3],
    op_a: i32,
    pick: kchain_pin::Pick,
    out_type: i32,
    stream: u64,
) -> Result<bool> {
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = op_a;
        let tb = CUBLAS_OP_N;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_TRANSA,
                &ta as *const i32 as *const c_void,
                4,
            ),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_TRANSB,
                &tb as *const i32 as *const c_void,
                4,
            ),
            "TRANSB",
        )?;
        // A = weight stored row-major [N,K] == col-major [K,N], ld=K, opT → [N,K]
        // B = act    stored row-major [M,K] == col-major [K,M], ld=K, opN → [K,M]
        // D = out    row-major [M,N]        == col-major [N,M], ld=N
        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let (a_rows, a_cols, a_ld) = if op_a == CUBLAS_OP_T {
            (k as u64, n as u64, k as i64) // row-major [N,K] -> col-major (K,N)
        } else {
            (n as u64, k as u64, n as i64) // row-major [K,N] -> col-major (N,K)
        };
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_16BF, a_rows, a_cols, a_ld),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_16BF, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, out_type, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        // Heuristic results: 96 B each, algo at offset 0 (+ margin, `kchain_pin`).
        let mut result = [0u8; 128 * kchain_pin::CANDIDATES];
        let returned = pinned_or_default(no_split_k(), |mask| {
            if let Some(mask) = mask {
                chk(
                    cublasLtMatmulPreferenceSetAttribute(
                        pref,
                        PREF_REDUCTION_SCHEME_MASK,
                        &mask as *const u32 as *const c_void,
                        std::mem::size_of::<u32>(),
                    ),
                    "PrefReductionMask",
                )?;
            }
            let mut returned: i32 = 0;
            chk(
                cublasLtMatmulAlgoGetHeuristic(
                    ctx.handle,
                    desc,
                    la,
                    lb,
                    ld_,
                    ld_,
                    pref,
                    pick.requested(),
                    result.as_mut_ptr() as *mut c_void,
                    &mut returned,
                ),
                "AlgoGetHeuristic",
            )?;
            ensure!(returned >= 1, "cuBLASLt: no algorithm for {m}x{n}x{k}");
            Ok(returned)
        })?;
        let Some(algo_off) = pick.offset(&result, returned) else {
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(la);
            cublasLtMatrixLayoutDestroy(lb);
            cublasLtMatrixLayoutDestroy(ld_);
            cublasLtMatmulDescDestroy(desc);
            return Ok(false);
        };
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            la,
            act as *const c_void,
            lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_,
            out as *mut c_void,
            ld_,
            result[algo_off..].as_ptr() as *const c_void,
            ctx.workspace()? as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(status, "Matmul")?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `pinned_or_default` over a fake heuristic: the masks it was asked for.
    fn masks(no_split_k: bool, answers: &[bool]) -> (Vec<Option<u32>>, Result<usize>) {
        let mut asked = Vec::new();
        let result = pinned_or_default(no_split_k, |mask| {
            asked.push(mask);
            ensure!(answers[asked.len() - 1], "no algorithm");
            Ok(asked.len())
        });
        (asked, result)
    }

    #[test]
    fn default_heuristic_leaves_the_reduction_mask_untouched() {
        let (asked, result) = masks(false, &[true]);
        assert_eq!((asked, result.unwrap()), (vec![None], 1));
        assert!(masks(false, &[false]).1.is_err());
    }

    #[test]
    fn split_k_pin_falls_back_to_every_scheme_when_it_finds_no_algorithm() {
        let (asked, result) = masks(true, &[true]);
        assert_eq!((asked, result.unwrap()), (vec![Some(SCHEME_NONE)], 1));
        let (asked, result) = masks(true, &[false, true]);
        assert_eq!(asked, [Some(SCHEME_NONE), Some(SCHEME_ALL)]);
        assert_eq!(result.unwrap(), 2);
        assert!(masks(true, &[false, false]).1.is_err());
    }
}
