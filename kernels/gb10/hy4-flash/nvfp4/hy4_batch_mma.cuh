// SPDX-License-Identifier: AGPL-3.0-only
// A3 — tensor-core (mma.sync m16n8k16) variant of hy4_batch_expert.
// Mirrors proven geometry/semantics:
//   grid [ceil(n/64), 1, active_experts]; block 256 (8 warps)
//   warp w -> columns o0 = blockIdx.x*64 + w*8 .. +7 (8 cols per warp)
//   per expert: sub-batches of 8 rows (count <= 8, padded to 16 in mma)
//
// Manual register loading (ldmatrix is broken on GB10, see
// inferspark_prefill.cu). mma is row-major A (m16 x k16), col-major B
// (k16 x n8), fp32 accumulator.
//
// Numerics: dequant weights are rounded to bf16 before mma (same precision
// class as the bf16 input). Final result is rounded to bf16 as the proven
// path does (hy4_round). Gate/up phase applies limit clamp + sigmoid gate.
//
// Staging (v2): shared, block-cooperative A pack + B dequant.
//   - Every warp reads the SAME 16x16 bf16 A tile per sub-batch per k-slice
//     (previously each warp redundantly re-packed it).
//   - B tiles (gate and up) are dequantized once per sub-batch per k-slice by
//     the whole block (previously each warp dequantized its own 8 cols and
//     wrote its own private sB buffer). Each warp then reads its 8 cols from
//     the shared tile. This turns ~8x redundant pack + ~8x redundant dequant
//     into one block-cooperative pass.

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

extern "C" __device__ __forceinline__ float hy4_sigmoid_mma(float v) {
    if (v >= 0.f) return 1.f / (1.f + expf(-v));
    float e = expf(v); return e / (1.f + e);
}
extern "C" __device__ __forceinline__ float hy4_round_mma(float x) {
    return (float)__float2bfloat16_rn(x);
}

// Cooperative A pack: 256 threads, thread i -> (row = i>>4, col = i&15).
// Writes a 16x16 row-major tile (16 rows x 16 k), zero-padded rows >= count.
__device__ __forceinline__ void pack_a_tile(
    __nv_bfloat16* sharedA,             // [16][16] row-major
    const __nv_bfloat16* x,
    const unsigned* entries, unsigned count,
    unsigned slots, unsigned down, unsigned k,
    unsigned k_slice, unsigned n)
{
    unsigned thread = threadIdx.x;
    unsigned row = thread >> 4;         // 0..15
    unsigned col = thread & 15;         // 0..15
    unsigned k_end = min(16u, k - k_slice);
    __nv_bfloat16 v = __float2bfloat16_rn(0.f);
    if (row < count && col < k_end) {
        unsigned entry = entries[row];
        unsigned row_index = down ? entry : (entry / slots);
        unsigned gc = k_slice + col;
        if (gc < k) v = x[(size_t)row_index * k + gc];
    }
    sharedA[row * 16 + col] = v;
}

// Cooperative B dequant: 64 cols x 16 k = 1024 bf16.
// thread i -> col = i>>2 (0..63), quarter = i&3 (4 k-values).
// sharedB is [64][16] col-major (c*16 + k).
template<bool DOWN_PHASE>
__device__ __forceinline__ void dequant_b_tile(
    __nv_bfloat16* sharedB,          // [64][16] (c*16 + k)
    const Hy4Q4* table, unsigned o0, unsigned k, unsigned k_slice,
    const float* lut, unsigned n)
{
    unsigned thread = threadIdx.x;
    unsigned c = thread >> 2;         // 0..63
    unsigned q = thread & 3;          // 0..3 (k quarter)
    unsigned o = o0 + c;
    unsigned group_id = k_slice >> 4;
    unsigned lo = q * 4;              // 0/4/8/12
    // valid k count this quarter covers in-slice
    int need = 4;
    if (k_slice + 16 > k) need = max(0, k - k_slice - lo);
    float scale = 0.f;
    unsigned byte_a = 0, byte_b = 0;
    if (need > 0 && o < n) {
        __nv_fp8_e4m3 sf; sf.__x = table->s[(size_t)o * (k >> 4) + group_id];
        scale = (float)sf * table->scale;
        // bytes of the 16-k slice this quarter covers: slice_bytes(2q, 2q+1)
        int bytes_total = (k - k_slice + 1) >> 1;   // ceil((k-slice)/2)
        const uint8_t* colw = table->w + (size_t)o * (k >> 1) + (k_slice >> 1);
        // byte 2q (k = 4q, 4q+1)
        if (2 * (int)q < (int)bytes_total) byte_a = colw[2 * q];
        // byte 2q+1 (k = 4q+2, 4q+3)
        if (2 * (int)q + 1 < (int)bytes_total) byte_b = colw[2 * q + 1];
    }
    __nv_bfloat16* sB = sharedB + c * 16;
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        unsigned at = lo + j;                       // 0..15 within slice
        if (at >= (k - k_slice)) { sB[at] = __float2bfloat16_rn(0.f); continue; }
        unsigned byte = (j < 2) ? byte_a : byte_b;
        unsigned nibble = j & 1;
        float v = lut[(byte >> (4 * nibble)) & 0x0f] * scale;
        sB[at] = __float2bfloat16_rn(v);
    }
}

// Main kernel. grid [ceil(n/64), 1, active_experts], block 256.
// warp w -> cols o0 = blockIdx.x*64 + w*8 .. +7 (8 cols)
extern "C" __global__ __launch_bounds__(256, 4) void hy4_batch_expert_mma(
    const __nv_bfloat16* x,
    const Hy4Q4* table, const unsigned* groups, __nv_bfloat16* out,
    unsigned n, unsigned k, unsigned rows, unsigned slots, unsigned group,
    unsigned down, float limit, unsigned experts)
{
    const unsigned* active = groups + (size_t)experts * (rows + 1);
    if (blockIdx.z >= active[0]) return;
    unsigned e = active[1 + blockIdx.z];
    unsigned warp = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    unsigned o0 = blockIdx.x * 64 + warp * 8;        // first col of this warp
    unsigned group_id = lane >> 2;                  // 0..7 within tile (col = group_id)
    unsigned tid = lane & 3;                        // 0..3
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = HY4_FP4[threadIdx.x];
    __syncthreads();
    // shared A/B tiles (block-cooperative)
    __shared__ __nv_bfloat16 sA[16 * 16];
    __shared__ __nv_bfloat16 sB[2][64 * 16];
    const unsigned* list = groups + (size_t)e * (rows + 1);
    // block-level activity: whole block out of range -> nothing to do
    if (blockIdx.x * 64 >= n) return;
    // per-warp activation: only the last col-tile can have sub-range warps.
    // Those warps still join the block-wide pack/dequant + barriers, but skip
    // the mma and epilogue.
    const bool warp_active = o0 < n;

    const Hy4Q4* tq4_gate  = &table[(e + group) * 3 + (down ? 2 : 0)];
    const Hy4Q4* tq4_up    = &table[(e + group) * 3 + (down ? 2 : 1)];
    if (!tq4_gate->w) {
        // Null host-local weight (remote expert under EP): write zeros for
        // every active entry and every valid column this warp covers.
        // (per-warp, before any block-wide barrier — safe.)
        if (threadIdx.x % 32 == 0 && warp_active) {
            for (unsigned first0 = 0; first0 < list[0]; first0 += 8) {
                unsigned count0 = min(8u, list[0] - first0);
                const unsigned* entries0 = list + 1 + first0;
                for (unsigned r0 = 0; r0 < count0; ++r0) {
                    unsigned o_limit = min(o0 + 8u, n);
                    for (unsigned o = o0; o < o_limit; ++o)
                        out[(size_t)entries0[r0] * n + o] = __float2bfloat16_rn(0.f);
                }
            }
        }
        return;
    }

    for (unsigned first = 0; first < list[0]; first += 8) {
        unsigned count = min(8u, list[0] - first);
        const unsigned* entries = list + 1 + first;
        float acc_a[4] = {0.f, 0.f, 0.f, 0.f};
        float acc_b[4] = {0.f, 0.f, 0.f, 0.f};

        for (unsigned k_slice = 0; k_slice < k; k_slice += 16) {
            __syncthreads();                       // sA/sB reuse barrier
            pack_a_tile(sA, x, entries, count, slots, down, k, k_slice, n);
            dequant_b_tile<true>(sB[0], tq4_gate,  blockIdx.x * 64, k, k_slice, lut, n);
            if (!down) dequant_b_tile<true>(sB[1], tq4_up, blockIdx.x * 64, k, k_slice, lut, n);
            __syncthreads();
            if (warp_active) {
                const unsigned short* sA_u16 = (const unsigned short*)sA;
                const unsigned short* sBg_u16 = (const unsigned short*)sB[0];
                const unsigned short* sBu_u16 = (const unsigned short*)sB[1];
                unsigned int ar0 = group_id;
                unsigned int ar1 = group_id + 8;
                unsigned int ac0 = tid * 2;
                unsigned int ac1 = tid * 2 + 8;
                unsigned int a0 = *(const unsigned int*)&sA_u16[ar0 * 16 + ac0];
                unsigned int a1 = *(const unsigned int*)&sA_u16[ar1 * 16 + ac0];
                unsigned int a2 = *(const unsigned int*)&sA_u16[ar0 * 16 + ac1];
                unsigned int a3 = *(const unsigned int*)&sA_u16[ar1 * 16 + ac1];
                unsigned int b0 = 0, b1 = 0;
                {
                    unsigned int sBg_base = (warp * 8 + group_id) * 16;
                    b0 = ((unsigned int)sBg_u16[sBg_base + tid * 2 + 1] << 16)
                       | (unsigned int)sBg_u16[sBg_base + tid * 2];
                    b1 = ((unsigned int)sBg_u16[sBg_base + tid * 2 + 8 + 1] << 16)
                       | (unsigned int)sBg_u16[sBg_base + tid * 2 + 8];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(acc_a[0]), "+f"(acc_a[1]), "+f"(acc_a[2]), "+f"(acc_a[3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
                if (!down) {
                    unsigned int sBu_base = (warp * 8 + group_id) * 16;
                    b0 = ((unsigned int)sBu_u16[sBu_base + tid * 2 + 1] << 16)
                       | (unsigned int)sBu_u16[sBu_base + tid * 2];
                    b1 = ((unsigned int)sBu_u16[sBu_base + tid * 2 + 8 + 1] << 16)
                       | (unsigned int)sBu_u16[sBu_base + tid * 2 + 8];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(acc_b[0]), "+f"(acc_b[1]), "+f"(acc_b[2]), "+f"(acc_b[3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
        }

        if (!warp_active) continue;
        // Epilogue: rows group_id (0..7) -> acc[0],acc[1]; rows group_id+8 -> acc[2],acc[3]
        // col pair = tid*2, tid*2+1
        unsigned cols_local0 = tid * 2;
        unsigned cols_local1 = tid * 2 + 1;
        // row 0 -> group_id
        for (unsigned rr = 0; rr < 2; ++rr) {
            unsigned row_local = rr == 0 ? group_id : group_id + 8;
            if (row_local >= count) continue;
            unsigned entry = entries[row_local];
            // output row uses the flat entry (proven path)
            size_t out_row = (size_t)entry * n;
            for (unsigned cc = 0; cc < 2; ++cc) {
                unsigned cols_local = cc == 0 ? cols_local0 : cols_local1;
                unsigned o = o0 + cols_local;
                if (o >= n) continue;
                float v = acc_a[rr * 2 + cc];
                if (!down) {
                    float u = acc_b[rr * 2 + cc];
                    v = hy4_round_mma(v); u = hy4_round_mma(u);
                    if (limit > 0.f) {
                        v = fminf(v, limit);
                        u = fminf(limit, fmaxf(-limit, u));
                    }
                    v = v * hy4_sigmoid_mma(v) * u;
                } else {
                    v = hy4_round_mma(v);
                }
                out[out_row + o] = __float2bfloat16_rn(v);
            }
        }
    }
}
