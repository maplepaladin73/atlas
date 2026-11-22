// SPDX-License-Identifier: AGPL-3.0-only
// A3-fp8 — FP8 (e4m3) mma.sync variant of hy4_batch_expert, opt-in via
// ATLAS_HY4_EXPERT_MMA=2.
//
// Motivation: the mxf4nvf4 port (ATLAS_HY4_EXPERT_MMA=1) runs on the native
// FP4 instruction but must quantize activations bf16->e2m1 (1 mantissa bit),
// which showed 12/20 greedy text mismatches on GB10. The e4m3 path keeps 3
// mantissa bits on BOTH operands: activations are converted bf16->e4m3
// in-register at fragment load (no separate quantize pass), weights are
// dequantized fp4->e4m3 once per stage into a shared compute bank with the
// per-16 e4m3 scale folded in (the fp32 global table scale stays in the
// epilogue, exactly like the proven kernel). Instruction:
//
//   mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//
// (the fp8 mma used by the proven glm/deepseek moe_w4a16 path; unlike
// mxf4nvf4 it SASS-compiles for sm_121a, so the parity harness can run it).
//
// Geometry per CTA (same as the mxf4 port): 128 output columns x 64 rows.
//   grid [ceil(n/128), 1, min(experts, rows*slots)], block 256 (8 warps),
//   warp (w&3) -> 16 rows, warp (w>>2) -> 64 columns; the expert's entry
//   list is processed in 64-entry row blocks.
//   K pipeline stage = 128 (four k32 mma slices), single-buffered to stay
//   under the 48KB static shared limit; B packed fp4 is staged, then a
//   block-wide dequant pass produces the e4m3 compute bank.
//
// Semantics preserved from hy4_batch_expert:
//   * launch signature identical to hy4_batch_expert (x is plain bf16);
//   * down==0: dual gate/up outputs with hy4_round, SiLU gate, limit clamp;
//     down==1: plain projection, round only;
//   * EP null-expert guard: null table->w zero-fills every valid entry and
//     valid column of the block;
//   * K and N must be multiples of 128 (production 3072/1536 qualify; the
//     launcher keeps the proven kernel otherwise).

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

// One e4m3 value from fp32 (both source operands identical so the packed
// byte order is unambiguous; take the low byte).
__device__ __forceinline__ unsigned char fp8_cvt_e4m3(float v) {
    unsigned short p;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(p) : "f"(v));
    return (unsigned char)(p & 0xFFu);
}
// 4 bf16 (8 bytes) -> one reg of 4 e4m3, byte i = element i.
__device__ __forceinline__ unsigned int fp8_bf16x4_to_e4m3x4(const void* p) {
    const unsigned int* u = (const unsigned int*)p;
    unsigned short h0 = (unsigned short)(u[0] & 0xFFFFu);
    unsigned short h1 = (unsigned short)(u[0] >> 16);
    unsigned short h2 = (unsigned short)(u[1] & 0xFFFFu);
    unsigned short h3 = (unsigned short)(u[1] >> 16);
    __nv_bfloat16 b;
    float f0, f1, f2, f3;
    b = *(__nv_bfloat16*)&h0; f0 = (float)b;
    b = *(__nv_bfloat16*)&h1; f1 = (float)b;
    b = *(__nv_bfloat16*)&h2; f2 = (float)b;
    b = *(__nv_bfloat16*)&h3; f3 = (float)b;
    unsigned int r0 = (unsigned int)fp8_cvt_e4m3(f0) | ((unsigned int)fp8_cvt_e4m3(f1) << 8);
    unsigned int r1 = (unsigned int)fp8_cvt_e4m3(f2) | ((unsigned int)fp8_cvt_e4m3(f3) << 8);
    return r0 | (r1 << 16);
}
// One m16n8k32 e4m3 mma (fragment mapping copied from the proven
// moe_w4a16 fp8 path: A rows fr0/fr0+8 at k tid4*4 and +16; B col
// nt*8+gid at k tid4*4 and +16).
__device__ __forceinline__ void fp8_mma(
    float* acc, unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
}

#define F8_STAGE 128
#define F8_AP (F8_STAGE * 2 + 16)   // sA row pitch in bytes (bf16), 16B aligned
#define F8_BP (F8_STAGE / 2 + 16)   // sBr row pitch (packed fp4), 16B aligned
#define F8_FP (F8_STAGE + 16)       // sBf row pitch (fp8), 16B aligned

extern "C" __global__ void __launch_bounds__(256) hy4_batch_expert_mma_fp8(
    const __nv_bfloat16* x,           // plain bf16 activations
    const Hy4Q4* table, const unsigned* groups, __nv_bfloat16* out,
    unsigned n, unsigned k, unsigned rows, unsigned slots, unsigned group,
    unsigned down, float limit, unsigned experts)
{
    const unsigned* active = groups + (size_t)experts * (rows + 1);
    if (blockIdx.z >= active[0]) return;
    unsigned e = active[1 + blockIdx.z];
    const unsigned cta_n = blockIdx.x * 128;
    if (cta_n >= n) return;
    const unsigned t = threadIdx.x;
    const unsigned warp_id = t / 32, lane_id = t % 32;
    const unsigned warp_m = (warp_id & 3) * 16;
    const unsigned warp_n = (warp_id >> 2) * 64;
    const unsigned gid = lane_id >> 2, tid4 = lane_id & 3;

    const unsigned char* a_bf16 = (const unsigned char*)x;
    const unsigned* list = groups + (size_t)e * (rows + 1);
    const unsigned count = list[0];

    const Hy4Q4 gq = table[(size_t)(e + group) * 3 + (down ? 2u : 0u)];
    const Hy4Q4 uq = table[(size_t)(e + group) * 3 + 1];

    // EP null-expert guard (remote expert): zero-fill, proven semantics.
    if (!gq.w || (!down && !uq.w)) {
        for (unsigned i = t; i < count * 128; i += 256) {
            unsigned r = i >> 7, c = cta_n + (i & 127u);
            if (r < count && c < n)
                out[(size_t)list[1 + r] * n + c] = __float2bfloat16_rn(0.f);
        }
        return;
    }

    const float w_scale = gq.scale;
    const float u_scale = uq.scale;

    __shared__ __align__(16) unsigned char sA[64][F8_AP];
    __shared__ __align__(16) unsigned char sBr[128][F8_BP];
    __shared__ __align__(16) unsigned char sSf[128][F8_STAGE / 16];
    __shared__ __align__(16) unsigned char sBf[128][F8_FP];
    __shared__ int sTok[64];

    // Phase 0 reads token rows (entry/slots addressing), phase 1 the flat
    // activated rows (entry itself) — same as the proven kernel.
    #define F8_AROW(raw_) (down ? (raw_) : ((raw_) / slots))

    // One K128 stage: load A rows + packed B tile + B scales, dequant B to
    // the fp8 compute bank, then run the four k32 mma slices.
    #define F8_STAGE_WORK(ACC, bw_, bs_) do { \
        for (unsigned st_ = 0; st_ < stages; ++st_) { \
            const unsigned kb_ = st_ * F8_STAGE; \
            for (unsigned j_ = t; j_ < 64u * 16u; j_ += 256) { \
                const unsigned row_ = j_ >> 4, cb_ = (j_ & 15u) * 16u; \
                if (row_ < m_cnt) \
                    *(uint4*)&sA[row_][cb_] = *(const uint4*)( \
                        a_bf16 + (size_t)F8_AROW((unsigned)sTok[row_]) * k * 2 + kb_ * 2 + cb_); \
            } \
            for (unsigned j_ = t; j_ < 128u * 8u; j_ += 256) { \
                const unsigned c_ = j_ >> 3, q_ = (j_ & 7u) * 16u; \
                const unsigned gc_ = cta_n + c_; \
                if (gc_ < n) \
                    *(uint4*)&sBr[c_][q_] = *(const uint4*)( \
                        (bw_) + (size_t)gc_ * (k / 2) + kb_ / 2 + q_); \
            } \
            if (t < 128) { \
                const unsigned gc2_ = cta_n + t; \
                if (gc2_ < n) \
                    *(uint2*)&sSf[t][0] = *(const uint2*)( \
                        (bs_) + (size_t)gc2_ * (k / 16) + kb_ / 16); \
            } \
            __syncthreads(); \
            for (unsigned j_ = t; j_ < 128u * 128u; j_ += 256) { \
                const unsigned c_ = j_ >> 7, kk_ = j_ & 127u; \
                const unsigned char byte_ = sBr[c_][kk_ >> 1]; \
                const unsigned nib_ = (kk_ & 1u) ? (byte_ >> 4) : (byte_ & 15u); \
                __nv_fp8_e4m3 sf_; sf_.__x = sSf[c_][kk_ >> 4]; \
                sBf[c_][kk_] = fp8_cvt_e4m3(HY4_FP4[nib_] * (float)sf_); \
            } \
            __syncthreads(); \
            _Pragma("unroll") \
            for (int sl_ = 0; sl_ < 4; ++sl_) { \
                const unsigned base_ = sl_ * 32; \
                const unsigned a0_ = fp8_bf16x4_to_e4m3x4(&sA[warp_m + gid][base_ + tid4 * 4]); \
                const unsigned a1_ = fp8_bf16x4_to_e4m3x4(&sA[warp_m + gid + 8][base_ + tid4 * 4]); \
                const unsigned a2_ = fp8_bf16x4_to_e4m3x4(&sA[warp_m + gid][base_ + 16 + tid4 * 4]); \
                const unsigned a3_ = fp8_bf16x4_to_e4m3x4(&sA[warp_m + gid + 8][base_ + 16 + tid4 * 4]); \
                _Pragma("unroll") \
                for (int nt_ = 0; nt_ < 8; ++nt_) { \
                    const unsigned nc_ = warp_n + nt_ * 8 + gid; \
                    const unsigned b0_ = *(const unsigned int*)&sBf[nc_][base_ + tid4 * 4]; \
                    const unsigned b1_ = *(const unsigned int*)&sBf[nc_][base_ + 16 + tid4 * 4]; \
                    fp8_mma(ACC[nt_], a0_, a1_, a2_, a3_, b0_, b1_); \
                } \
            } \
            __syncthreads(); \
        } \
    } while (0)

    const unsigned stages = k / F8_STAGE;
    for (unsigned base = 0; base < count; base += 64) {
        const unsigned m_cnt = min(64u, count - base);
        for (unsigned i = t; i < 64; ++i)
            sTok[i] = i < m_cnt ? (int)list[1 + base + i] : 0;
        __syncthreads();

        float accg[8][4], accu[8][4];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            accg[i][0]=accg[i][1]=accg[i][2]=accg[i][3]=
            accu[i][0]=accu[i][1]=accu[i][2]=accu[i][3]=0.f;

        F8_STAGE_WORK(accg, gq.w, gq.s);
        if (!down) F8_STAGE_WORK(accu, uq.w, uq.s);
        #undef F8_STAGE_WORK

        // Epilogue: identical store/round/clamp semantics to the proven
        // kernel; the fp32 global scale lands here, per-16 scales are
        // already folded into the fp8 B values.
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const unsigned c0 = cta_n + warp_n + nt * 8 + tid4 * 2, c1 = c0 + 1;
            const unsigned r0 = warp_m + gid, r1 = r0 + 8;
            if (r0 < m_cnt) {
                const unsigned or0 = (unsigned)sTok[r0];
                if (c0 < n) {
                    float v = accg[nt][0] * w_scale;
                    if (!down) {
                        float u = accu[nt][0] * u_scale;
                        v = hy4_round(v); u = hy4_round(u);
                        if (limit > 0.f) { v = fminf(v, limit); u = fminf(limit, fmaxf(-limit, u)); }
                        v = v * hy4_sigmoid(v) * u;
                    } else v = hy4_round(v);
                    out[(size_t)or0 * n + c0] = __float2bfloat16_rn(v);
                }
                if (c1 < n) {
                    float v = accg[nt][1] * w_scale;
                    if (!down) {
                        float u = accu[nt][1] * u_scale;
                        v = hy4_round(v); u = hy4_round(u);
                        if (limit > 0.f) { v = fminf(v, limit); u = fminf(limit, fmaxf(-limit, u)); }
                        v = v * hy4_sigmoid(v) * u;
                    } else v = hy4_round(v);
                    out[(size_t)or0 * n + c1] = __float2bfloat16_rn(v);
                }
            }
            if (r1 < m_cnt) {
                const unsigned or1 = (unsigned)sTok[r1];
                if (c0 < n) {
                    float v = accg[nt][2] * w_scale;
                    if (!down) {
                        float u = accu[nt][2] * u_scale;
                        v = hy4_round(v); u = hy4_round(u);
                        if (limit > 0.f) { v = fminf(v, limit); u = fminf(limit, fmaxf(-limit, u)); }
                        v = v * hy4_sigmoid(v) * u;
                    } else v = hy4_round(v);
                    out[(size_t)or1 * n + c0] = __float2bfloat16_rn(v);
                }
                if (c1 < n) {
                    float v = accg[nt][3] * w_scale;
                    if (!down) {
                        float u = accu[nt][3] * u_scale;
                        v = hy4_round(v); u = hy4_round(u);
                        if (limit > 0.f) { v = fminf(v, limit); u = fminf(limit, fmaxf(-limit, u)); }
                        v = v * hy4_sigmoid(v) * u;
                    } else v = hy4_round(v);
                    out[(size_t)or1 * n + c1] = __float2bfloat16_rn(v);
                }
            }
        }
        __syncthreads();   // sTok/smem reuse across row blocks
    }
    #undef F8_AROW
}
