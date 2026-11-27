// SPDX-License-Identifier: AGPL-3.0-only

// Ling-3.2-air gated-MLA attention kernels (GB10 / sm_121, NVFP4 target).
//
// Seven of the 42 text layers are full attention — indices {5, 11, 17, 23,
// 29, 35, 41}, the 5:1 pattern after the 2-layer dense prefix — and the
// NEXTN draft body model.layers.42 reuses the same attention form; every
// entry point below is layer-agnostic (the dispatcher passes each layer
// instance's own weight pointers), so those 8 instances share one code path.
// The draft-only eh_proj / enorm / hnorm machinery lives in ling_mtp_eh.cu.
//
// Layer ground truth (context/modeling_bailing_moe_v3_vl.py,
// BailingMoeV3MultiLatentAttention lines ~1101-1233, and the checkpoint's
// tensor index — the 7 layers' attention.* set is exactly {q_proj,
// kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, g_proj, dense}):
//
//   q_proj             hidden 2560 -> 32 * 192, per head [nope 128 | rope 64]
//                      (q_lora_rank is None in this config: no q_a/q_b pair,
//                      and no q_norm/k_norm tensors exist despite
//                      use_qk_norm=true — the MLA class has no qk norm)
//   kv_a_proj_with_mqa  hidden 2560 -> 576  = [kv_lora 512 | k rope raw 64]
//   kv_a_layernorm     BailingMoeV3RMSNorm(512), eps 1e-6, latent slice only
//   kv_b_proj          512 -> 32 * 256, per head [k_nope 128 | v 128]
//   g_proj             hidden 2560 -> 32 logits; gate = sigmoid(fp32(logit)),
//                      head_wise granularity, multiplied into the attention
//                      output per head BEFORE dense (o_proj)
//   dense (o_proj)     32 * 128 -> 2560
//
// Scale: self.scaling = qk_head_dim ** -0.5 with qk_head_dim = 192
// (nope 128 + rope 64) — the config's head_dim = 128 is NOT the scale dim,
// and rope_scaling is absent from the text_config so no YaRN mscale applies.
//
// Attention runs in the absorbed form used by the GLM-5.3-flash /
// deepseek-v4-flash machinery: q_nope . k_nope + q_rope . k_rope folds to
//   [q_nope @ W_UK (512) | q_rope (64)] . [latent 512 | rope 64]
// over a compressed KV cache (576 dims), and the attention output is the
// softmax-weighted latent (512), which per head extracts v = W_UV . latent
// (128) and then takes the head-wise gate. Layout notes for the dispatcher:
//
//   w_uk (ling_mla_q_absorb_rope): [heads, 512, 128] contiguous BF16, the
//     load-time transpose of kv_b_proj's per-head k_nope slice:
//     w_uk[h][n][j] == kv_b_proj[h * 256 + j][n].
//   w_uv (ling_mla_v_extract_gated): kv_b_proj's base pointer, no transform:
//     W_UV[h][n][k] == kv_b_proj[h * 256 + 128 + n][k].
//   q_proj / kv_a_proj_with_mqa / g_proj / dense: straight BF16 [N, 2560]
//     (or [2560, 4096] for dense) row-major gemv weights — the common dense
//     gemv kernels cover them; g_proj's sigmoid is fused into the extract.
//
// K cache rows are [normed latent 512 | roped rope 64] (576 BF16), V cache
// rows are the normed latent only (512 BF16): the reference value comes from
// kv_b_proj's v slice, which never sees the rope part, so unlike the
// deepseek-v4 shadow there is no V rope tail to carry, read or pad.
//
// Decode/verify flow per layer instance and step (gated-MLA layers and the
// draft body alike; rows = tokens per sequence: 1 decode, r+1 MTP verify):
//   q_full   = q_proj(hidden)                    [rows, 32*192]   (common gemv)
//   kv_a_raw = kv_a_proj_with_mqa(hidden)       [rows, 576]      (common gemv)
//   g_logits = g_proj(hidden)                   [rows, 32]       (common gemv)
//   ling_mla_kv_a_norm_rope_cache(kv_a_raw, ..) -> paged cache (+optional
//     contiguous K/V rows for prefill)
//   ling_mla_q_absorb_rope(q_full, w_uk, ..)     -> q_final [rows, 32, 576]
//   ling_mla_paged_attn(q_final, ..)             -> attn latent [seqs*rows,32,512]
//   ling_mla_v_extract_gated(latent, w_uv, g_logits) -> v [rows, 32*128]
//   dense(v)                                     (common gemv)
// Prefill (single-pass, chunk-local causal, conservative ISL <= 4096) swaps
// the paged kernel for ling_mla_prefill_attn over the contiguous rows.

#include <cuda_bf16.h>
#include <math.h>

#include "../../common/atlas_pdl.cuh"
#include "ling_mla_rope.cuh"

// The Ling-3.2-air MLA shape (see the header). Register-mapped vector sizes
// are derived from these; the runtime dim args are guarded against them.
#define LING_MLA_HEADS 32u     // num_attention_heads
#define LING_MLA_NOPE 128u     // qk_nope_head_dim
#define LING_MLA_ROPE 64u      // qk_rope_head_dim == rotary_dim
#define LING_MLA_QK 192u       // qk_head_dim: softmax scale = LING_MLA_QK ** -0.5
#define LING_MLA_LORA 512u     // kv_lora_rank
#define LING_MLA_V 128u       // v_head_dim
#define LING_MLA_KV_DIM 576u   // kv_lora_rank + qk_rope_head_dim (cache row)
#define LING_MLA_ATTN_WARPS 8u
#define LING_MLA_KVEC (LING_MLA_KV_DIM / 32u)   // 18 score dims a paged-attn lane
#define LING_MLA_VVEC (LING_MLA_LORA / 32u)     // 16 value dims a paged-attn lane
#define LING_MLA_PFKVEC (LING_MLA_KV_DIM / 16u) // 36 score dims a prefill lane
#define LING_MLA_PFVVEC (LING_MLA_LORA / 16u)   // 32 value dims a prefill lane
#define LING_MLA_PF_ROWS 16u                   // query rows a prefill CTA

// ============================================================================
// kv_a path: RMSNorm + partial rope + compressed cache write
// ============================================================================

// One CTA per row (a token): RMS-normalizes the 512-dim latent slice with
// kv_a_layernorm (fp32 stats; the normed value re-rounds to BF16 before the
// weight multiply, exactly like BailingMoeV3RMSNorm), ropes the raw 64-dim
// k-rope slice (pair-interleaved, theta 6e6), and writes the cache rows:
//   K = [normed latent 512 | roped rope 64]   V = [normed latent 512]
// to every non-null destination: the paged pool via the slot map (vLLM
// convention: slot = block * block_size + offset, -1 skips the paged write)
// and the contiguous k_contig/v_contig rows prefill attention reads —
// prefill typically populates both, decode only the pool.
//
// Grid: (rows, 1, 1)   Block: (128, 1, 1)
extern "C" __global__ void ling_mla_kv_a_norm_rope_cache(
    const __nv_bfloat16* __restrict__ kv_a_raw,  // [rows, 576] = [latent | raw rope]
    const __nv_bfloat16* __restrict__ norm_w,    // [512] kv_a_layernorm.weight
    const float eps,                              // rms_norm_eps (1e-6)
    __nv_bfloat16* __restrict__ k_contig,         // [rows, 576] or null: prefill K row
    __nv_bfloat16* __restrict__ v_contig,         // [rows, 512] or null: prefill V row
    __nv_bfloat16* __restrict__ k_cache,          // [pool, block_size, 576] or null
    __nv_bfloat16* __restrict__ v_cache,          // [pool, block_size, 512] or null
    const long long* __restrict__ slots,          // [rows] pool slots; -1 skips paged
    const unsigned int* __restrict__ positions,  // [rows] rope positions
    const unsigned int block_size,
    const unsigned int kv_dim,                    // 576
    const unsigned int kv_lora,                   // 512
    const double rope_theta                      // 6e6
) {
    atlas_pdl_enter();
    if (kv_dim != LING_MLA_KV_DIM || kv_lora != LING_MLA_LORA) return;
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* raw = kv_a_raw + (unsigned long long)row * kv_dim;

    // fp32 sum of squares over the latent slice.
    float ss = 0.0f;
    for (unsigned int d = tid; d < kv_lora; d += blockDim.x) {
        const float x = __bfloat162float(raw[d]);
        ss += x * x;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, offset);
    __shared__ float s_warp[4];  // one partial a warp: 4 warps of the 128 threads
    if ((tid & 31u) == 0u) s_warp[tid >> 5] = ss;
    __syncthreads();
    const float inv = rsqrtf(
        (s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3]) / (float)kv_lora + eps);

    // Destinations: the paged slot (if any) and/or the contiguous prefill
    // rows are all written — prefill wants both at once.
    const long long slot = slots[row];
    const bool k_paged = slot >= 0 && k_cache != nullptr;
    const bool v_paged = slot >= 0 && v_cache != nullptr;
    unsigned long long cache_off = 0ull;
    if (k_paged || v_paged) {
        cache_off = (unsigned long long)(slot / (long long)block_size) * block_size
                  + (unsigned long long)(slot % (long long)block_size);
    }
    __nv_bfloat16* k_cache_dst = k_paged ? k_cache + cache_off * kv_dim : nullptr;
    __nv_bfloat16* v_cache_dst = v_paged ? v_cache + cache_off * kv_lora : nullptr;
    __nv_bfloat16* k_ct = k_contig ? k_contig + (unsigned long long)row * kv_dim : nullptr;
    __nv_bfloat16* v_ct = v_contig ? v_contig + (unsigned long long)row * kv_lora : nullptr;

    // Normed latent: one BF16 round before the weight product, one after.
    for (unsigned int d = tid; d < kv_lora; d += blockDim.x) {
        const __nv_bfloat16 nrm = __float2bfloat16(__bfloat162float(raw[d]) * inv);
        const __nv_bfloat16 val =
            __float2bfloat16(__bfloat162float(nrm) * __bfloat162float(norm_w[d]));
        if (k_ct != nullptr) k_ct[d] = val;
        if (k_cache_dst != nullptr) k_cache_dst[d] = val;
        if (v_ct != nullptr) v_ct[d] = val;
        if (v_cache_dst != nullptr) v_cache_dst[d] = val;
    }

    // The rope tail: threads 0..31 each rotate one pair of the raw slice.
    const unsigned int rope_dim = kv_dim - kv_lora;
    if (tid < rope_dim / 2u) {
        const float freq = ling_mla_rope_pair_freq(tid, rope_dim, rope_theta);
        const float ang = (float)positions[row] * freq;
        const float cos_v = cosf(ang);
        const float sin_v = sinf(ang);
        const __nv_bfloat16* r = raw + kv_lora;
        float re, ro;
        ling_mla_rope_rotate_pair(__bfloat162float(r[2u * tid]),
                                  __bfloat162float(r[2u * tid + 1u]), cos_v, sin_v, re, ro);
        const __nv_bfloat16 re_bf = __float2bfloat16(re);
        const __nv_bfloat16 ro_bf = __float2bfloat16(ro);
        if (k_ct != nullptr) {
            k_ct[kv_lora + 2u * tid] = re_bf;
            k_ct[kv_lora + 2u * tid + 1u] = ro_bf;
        }
        if (k_cache_dst != nullptr) {
            k_cache_dst[kv_lora + 2u * tid] = re_bf;
            k_cache_dst[kv_lora + 2u * tid + 1u] = ro_bf;
        }
    }
}

// ============================================================================
// q path: absorption + rope + final q assembly
// ============================================================================

// One CTA per (head, absorbed-dim tile -- and row tile, the third grid
// dim): each of the 256 threads owns one latent output dim n and loops
// `row_tile` of the query rows, so w_uk's rows stream once per row batch
// instead of once per row. Tile-0 CTAs also rope the head's
// q_rope slice and stitch the final row:
//   q_final[row, head] = [q_nope @ W_UK (512) | roped q_rope (64)]
//
// w_uk: [heads, 512, 128] contiguous BF16, the load-time transpose of
// kv_b_proj's per-head k_nope slice (w_uk[h][n][j] == kv_b_proj[h*256 + j][n]).
//
// Grid: (heads, kv_lora / 256, ceil(rows / row_tile))   Block: (256, 1, 1)
extern "C" __global__ void ling_mla_q_absorb_rope(
    const __nv_bfloat16* __restrict__ q_full,    // [rows, heads * 192]
    const __nv_bfloat16* __restrict__ w_uk,     // [heads, 512, 128] (transposed view)
    __nv_bfloat16* __restrict__ q_final,         // [rows, heads, 576]
    const unsigned int* __restrict__ positions, // [rows]
    const unsigned int rows,
    const unsigned int heads,    // 32
    const unsigned int nope,     // 128
    const unsigned int rope_dim, // 64
    const unsigned int kv_lora,  // 512
    const double rope_theta,     // 6e6
    const unsigned int row_tile   // rows per blockIdx.z
) {
    atlas_pdl_enter();
    if (heads != LING_MLA_HEADS || nope != LING_MLA_NOPE || rope_dim != LING_MLA_ROPE
        || kv_lora != LING_MLA_LORA) {
        return;
    }
    const unsigned int head = blockIdx.x;
    if (head >= heads) return;
    const unsigned int out = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned int qk = nope + rope_dim;  // 192
    const unsigned int q_row = (unsigned long long)heads * qk;   // q_full row stride
    const unsigned int qf_row = (unsigned long long)heads * (kv_lora + rope_dim);
    const __nv_bfloat16* w = w_uk + (unsigned long long)head * kv_lora * nope
                           + (unsigned long long)out * nope;

    // Wide chunks split the row loop across blockIdx.z (row_tile rows a
    // CTA): hundreds of CTAs instead of heads x 2, w_uk streaming once per
    // CTA either way. row counts below the tile keep gridDim.z == 1.
    const unsigned int r_begin = blockIdx.z * row_tile;
    if (r_begin >= rows) return;
    const unsigned int r_end = (r_begin + row_tile < rows) ? r_begin + row_tile : rows;

    if (out < kv_lora) {
        for (unsigned int r = r_begin; r < r_end; ++r) {
            const __nv_bfloat16* qn =
                q_full + (unsigned long long)r * q_row + (unsigned long long)head * qk;
            float acc = 0.0f;
            for (unsigned int j = 0; j < nope; ++j) {
                acc += __bfloat162float(qn[j]) * __bfloat162float(w[j]);
            }
            q_final[(unsigned long long)r * qf_row + (unsigned long long)head * (kv_lora + rope_dim)
                    + out] = __float2bfloat16(acc);
        }
    }

    // Tile 0: the head's roped q_rope slice for the SAME row slice (32
    // pairs a row).
    if (blockIdx.y == 0u && threadIdx.x < rope_dim / 2u) {
        const unsigned int p = threadIdx.x;
        const float freq = ling_mla_rope_pair_freq(p, rope_dim, rope_theta);
        for (unsigned int r = r_begin; r < r_end; ++r) {
            const float ang = (float)positions[r] * freq;
            const float cos_v = cosf(ang);
            const float sin_v = sinf(ang);
            const __nv_bfloat16* qr = q_full + (unsigned long long)r * q_row
                                     + (unsigned long long)head * qk + nope;
            float re, ro;
            ling_mla_rope_rotate_pair(__bfloat162float(qr[2u * p]),
                                      __bfloat162float(qr[2u * p + 1u]), cos_v, sin_v, re, ro);
            __nv_bfloat16* dst =
                q_final + (unsigned long long)r * qf_row
                + (unsigned long long)head * (kv_lora + rope_dim) + kv_lora;
            dst[2u * p] = __float2bfloat16(re);
            dst[2u * p + 1u] = __float2bfloat16(ro);
        }
    }
}

// ============================================================================
// attention: paged decode / MTP verify
// ============================================================================

// Absorbed attention over the paged compressed BF16 cache. One CTA per
// (head, sequence); 8 warps. rows = query rows per sequence (1 on the decode
// path; r + 1 on the MTP verify path). Warp w owns query row (w % rows) and
// the (w / rows)-th KV chunk of that row's causal range [0, base_len + row):
// the row's own just-written cache entry at position base_len + row included,
// so base_len counts the cache entries BEFORE this decode/verify block.
// The row's per-warp partials merge through shared memory: 8 * 512 floats =
// 16 KB per CTA (the in-tree compressed-decode kernel's 18 KB, minus the V
// rope tail this model does not have). Non-power-of-two row counts stay
// correct: row r is served by warps {r, r + rows, ...} < 8, however many
// those are; up to 8 rows.
//
// Q: [num_seqs, rows, heads, 576]    O: [num_seqs, rows, heads, 512]
// K cache: [pool, block_size, 576]   V cache: [pool, block_size, 512]
// heads = gridDim.x; num_seqs = gridDim.y.
//
// Grid: (heads, num_seqs)   Block: (256, 1, 1)
extern "C" __global__ void ling_mla_paged_attn(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,           // [pool, block_size, 576]
    const __nv_bfloat16* __restrict__ V_cache,           // [pool, block_size, 512]
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_tables,                // [num_seqs, max_blocks]
    const unsigned int* __restrict__ base_len,           // [num_seqs]
    const unsigned int max_blocks_per_seq,
    const unsigned int rows,                            // 1..8 query rows a seq
    const unsigned int head_dim,                         // 576
    const unsigned int v_dim,                           // 512
    const float inv_sqrt_d,                             // 192 ** -0.5
    const unsigned int block_size
) {
    atlas_pdl_enter();
    if (head_dim != LING_MLA_KV_DIM || v_dim != LING_MLA_LORA) return;
    if (rows == 0u || rows > LING_MLA_ATTN_WARPS) return;
    const unsigned int head = blockIdx.x;
    const unsigned int seq = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;

    const unsigned int row = warp % rows;
    const unsigned int chunk = warp / rows;
    const unsigned int n_chunks = (LING_MLA_ATTN_WARPS - 1u - row) / rows + 1u;
    const unsigned int kv_len = base_len[seq] + row + 1u;

    // This warp's query row: 32 lanes x 18 dims cover [ absorbed 512 | rope 64 ].
    const __nv_bfloat16* q_row =
        Q + ((unsigned long long)(seq * rows + row) * gridDim.x + head) * LING_MLA_KV_DIM;
    float q_reg[LING_MLA_KVEC];
    #pragma unroll
    for (unsigned int i = 0; i < LING_MLA_KVEC; ++i)
        q_reg[i] = __bfloat162float(q_row[lane * LING_MLA_KVEC + i]);

    // This warp's KV chunk: [chunk * cs, min((chunk + 1) * cs, kv_len)).
    const unsigned int chunk_size = (kv_len + n_chunks - 1u) / n_chunks;
    unsigned int pos = chunk * chunk_size;
    unsigned int end = pos + chunk_size;
    if (end > kv_len) end = kv_len;

    float m = -1e30f;
    float l = 0.0f;
    float o_reg[LING_MLA_VVEC];
    #pragma unroll
    for (unsigned int i = 0; i < LING_MLA_VVEC; ++i) o_reg[i] = 0.0f;

    const int* table = block_tables + (unsigned long long)seq * max_blocks_per_seq;
    while (pos < end) {
        const unsigned int logical_block = pos / block_size;
        const unsigned int block_offset = pos % block_size;
        unsigned int in_block = block_size - block_offset;
        if (in_block > end - pos) in_block = end - pos;
        const unsigned int physical_block = (unsigned int)table[logical_block];
        const __nv_bfloat16* k_base =
            K_cache + ((unsigned long long)physical_block * block_size + block_offset)
                          * LING_MLA_KV_DIM;
        const __nv_bfloat16* v_base =
            V_cache + ((unsigned long long)physical_block * block_size + block_offset)
                          * LING_MLA_LORA;

        for (unsigned int b = 0; b < in_block; ++b, ++pos) {
            const __nv_bfloat16* k_row = k_base + (unsigned long long)b * LING_MLA_KV_DIM;
            const __nv_bfloat16* v_row = v_base + (unsigned long long)b * LING_MLA_LORA;

            float dot = 0.0f;
            #pragma unroll
            for (unsigned int i = 0; i < LING_MLA_KVEC; ++i)
                dot += q_reg[i] * __bfloat162float(k_row[lane * LING_MLA_KVEC + i]);
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, offset);

            const float score = dot * inv_sqrt_d;
            const float m_new = fmaxf(m, score);
            const float exp_old = __expf(m - m_new);
            const float p = __expf(score - m_new);
            l = l * exp_old + p;
            m = m_new;
            #pragma unroll
            for (unsigned int i = 0; i < LING_MLA_VVEC; ++i)
                o_reg[i] = o_reg[i] * exp_old
                         + p * __bfloat162float(v_row[lane * LING_MLA_VVEC + i]);
        }
    }

    // Inter-warp merge: every warp's partial lands in shared memory; each
    // row's chunk-0 warp (warp == row, i.e. warp < rows) collects its row.
    __shared__ float s_m[LING_MLA_ATTN_WARPS];
    __shared__ float s_l[LING_MLA_ATTN_WARPS];
    __shared__ float s_o[LING_MLA_ATTN_WARPS][LING_MLA_LORA];
    if (lane == 0u) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    #pragma unroll
    for (unsigned int i = 0; i < LING_MLA_VVEC; ++i)
        s_o[warp][lane * LING_MLA_VVEC + i] = o_reg[i];
    __syncthreads();

    if (warp < rows) {
        float m_r = s_m[warp];
        float l_r = s_l[warp];
        float o_r[LING_MLA_VVEC];
        #pragma unroll
        for (unsigned int i = 0; i < LING_MLA_VVEC; ++i)
            o_r[i] = s_o[warp][lane * LING_MLA_VVEC + i];
        for (unsigned int w2 = warp + rows; w2 < LING_MLA_ATTN_WARPS; w2 += rows) {
            if (s_l[w2] > 0.0f) {
                const float m_new = fmaxf(m_r, s_m[w2]);
                const float exp_me = __expf(m_r - m_new);
                const float exp_w = __expf(s_m[w2] - m_new);
                l_r = l_r * exp_me + s_l[w2] * exp_w;
                #pragma unroll
                for (unsigned int i = 0; i < LING_MLA_VVEC; ++i)
                    o_r[i] = o_r[i] * exp_me
                           + s_o[w2][lane * LING_MLA_VVEC + i] * exp_w;
                m_r = m_new;
            }
        }
        const float inv_l = (l_r > 0.0f) ? 1.0f / l_r : 0.0f;
        __nv_bfloat16* o_row =
            O + ((unsigned long long)(seq * rows + warp) * gridDim.x + head) * LING_MLA_LORA;
        #pragma unroll
        for (unsigned int i = 0; i < LING_MLA_VVEC; ++i)
            o_row[lane * LING_MLA_VVEC + i] = __float2bfloat16(o_r[i] * inv_l);
    }
}

// ============================================================================
// head-wise gate: V extraction with the sigmoid gate fused in
// ============================================================================

// One CTA per (row, head, 4-dim output tile): the batched-per-head gemv of
// the in-tree grouped_gemm_mla shape (64 threads a K=512 reduction per
// output, two warps merged through shared memory), with the modeling file's
// head-wise gate applied at the store:
//   v[row, head, n] = sigmoid(fp32(g_logits[row, head]))
//                     * sum_k attn_latent[row, head, k] * W_UV[head][n][k]
// The gate is the raw g_proj output element: sigmoid(fp32(bf16 logit)) --
// the gate is computed in fp32 and folded into the fp32 dot before the one
// BF16 round, instead of a separate pass over the attention output.
//
// w_uv is kv_b_proj's base pointer (no load-time transform):
//   W_UV[head][n][k] == kv_b_proj[head * (nope + v_dim) + nope + n][k]
//
// Grid: (rows * heads * (v_dim / 4))   Block: (256, 1, 1)
extern "C" __global__ void ling_mla_v_extract_gated(
    const __nv_bfloat16* __restrict__ attn_latent,  // [rows, heads, 512]
    const __nv_bfloat16* __restrict__ w_uv,         // kv_b_proj [heads*256, 512]
    const __nv_bfloat16* __restrict__ g_logits,    // [rows, heads] raw g_proj out
    __nv_bfloat16* __restrict__ v_out,             // [rows, heads, 128]
    const unsigned int rows,
    const unsigned int heads,    // 32
    const unsigned int nope,     // 128
    const unsigned int v_dim,    // 128
    const unsigned int kv_lora   // 512
) {
    atlas_pdl_enter();
    if (heads != LING_MLA_HEADS || nope != LING_MLA_NOPE || v_dim != LING_MLA_V
        || kv_lora != LING_MLA_LORA) {
        return;
    }
    const unsigned int tiles = v_dim / 4u;
    const unsigned int rh = blockIdx.x / tiles;
    if (rh >= rows * heads) return;
    const unsigned int tile = blockIdx.x % tiles;
    const unsigned int row = rh / heads;
    const unsigned int head = rh % heads;
    const unsigned int tid = threadIdx.x;
    const unsigned int threads_per_out = blockDim.x / 4u;  // 64
    const unsigned int local_n = tid / threads_per_out;   // 0..3
    const unsigned int k_lane = tid % threads_per_out;    // 0..63
    const unsigned int n = tile * 4u + local_n;

    const __nv_bfloat16* a =
        attn_latent + ((unsigned long long)(row * heads) + head) * kv_lora;
    const __nv_bfloat16* w =
        w_uv + (unsigned long long)(head * (nope + v_dim) + nope + n) * kv_lora;

    float acc = 0.0f;
    for (unsigned int k = k_lane; k < kv_lora; k += threads_per_out)
        acc += __bfloat162float(a[k]) * __bfloat162float(w[k]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, offset);

    __shared__ float s_p[4][2];
    const unsigned int warp_in_out = k_lane / 32u;
    if ((k_lane & 31u) == 0u) s_p[local_n][warp_in_out] = acc;
    __syncthreads();

    if (k_lane == 0u) {
        const float gate =
            1.0f / (1.0f + __expf(-__bfloat162float(g_logits[(unsigned long long)row * heads + head])));
        const float v = (s_p[local_n][0] + s_p[local_n][1]) * gate;
        v_out[((unsigned long long)(row * heads) + head) * v_dim + n] = __float2bfloat16(v);
    }
}

// ============================================================================
// attention: single-pass causal prefill over the packed context
// ============================================================================

// Causal prefill over one contiguous K/V run: the driver's packed context,
// which holds positions [0, kv_base + seq_len) — the prefix the ctx gather
// refilled from the paged pools plus the chunk's own rows the cache write
// just appended. The chunk's query rows [0, seq_len) attend the full run:
// rows < kv_base unmasked (causal by construction), the rest causal
// (q_pos + kv_base >= kv_pos). kv_base == 0 is the whole-prompt single pass
// (chunk-local causal from position 0). Scalar fp32 dots like the in-tree
// mla_prefill_attn: at these row counts the pass is KV-bandwidth-bound.
//
// 16 query rows a CTA, 16 lanes a row: 36 score dims a lane (the 576 cache
// row), 32 value dims a lane (the 512 latent). Each lane loads its 36 / 32
// contiguous elements as 8-byte uint2 slices (4 bf16 each: 8B-aligned by
// construction), preserving the scalar FMA order -> bit-identical output.
// causal == 0 runs the full run unmasked (parity tests).
//
// Q: [batch, seq_len, heads, 576]  K: [batch, kv_base + seq_len, 576]
// V: [batch, kv_base + seq_len, 512]  O: [batch, seq_len, heads, 512]
// heads = gridDim.x.
//
// Grid: (heads, ceil(seq_len / 16), batch)   Block: (256, 1, 1)
extern "C" __global__ void ling_mla_prefill_attn(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,   // [batch, kv_base + seq_len, 576]
    const __nv_bfloat16* __restrict__ V,   // [batch, kv_base + seq_len, 512]
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,            // the chunk's query rows
    const unsigned int causal,
    const float inv_sqrt_d,                // 192 ** -0.5
    const unsigned int kv_base            // prefix rows carried in K/V
) {
    atlas_pdl_enter();
    const unsigned int head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    const unsigned int q_row = q_block * LING_MLA_PF_ROWS + tid / 16u;
    if (q_row >= seq_len) return;
    const unsigned int lane = tid % 16u;
    const unsigned int warp_lane = tid % 32u;

    const __nv_bfloat16* q_base =
        Q + ((unsigned long long)(batch * seq_len + q_row) * gridDim.x + head) * LING_MLA_KV_DIM;
    float q_reg[LING_MLA_PFKVEC];
    #pragma unroll
    for (unsigned int i = 0; i < LING_MLA_PFKVEC / 4u; ++i) {
        const uint2 pk = *reinterpret_cast<const uint2*>(q_base + lane * LING_MLA_PFKVEC + i * 4u);
        const __nv_bfloat16* elems = reinterpret_cast<const __nv_bfloat16*>(&pk);
        #pragma unroll
        for (unsigned int j = 0; j < 4u; ++j)
            q_reg[i * 4u + j] = __bfloat162float(elems[j]);
    }

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float o_acc[LING_MLA_PFVVEC];
    #pragma unroll
    for (unsigned int i = 0; i < LING_MLA_PFVVEC; ++i) o_acc[i] = 0.0f;

    const unsigned int kv_run = kv_base + seq_len;
    const unsigned int kv_end = causal ? (kv_base + q_row + 1u) : kv_run;
    for (unsigned int kv_pos = 0; kv_pos < kv_end; ++kv_pos) {
        const __nv_bfloat16* k_row =
            K + (unsigned long long)(batch * kv_run + kv_pos) * LING_MLA_KV_DIM;
        float dot = 0.0f;
        #pragma unroll
        for (unsigned int i = 0; i < LING_MLA_PFKVEC / 4u; ++i) {
            const uint2 pk = *reinterpret_cast<const uint2*>(k_row + lane * LING_MLA_PFKVEC + i * 4u);
            const __nv_bfloat16* elems = reinterpret_cast<const __nv_bfloat16*>(&pk);
            #pragma unroll
            for (unsigned int j = 0; j < 4u; ++j)
                dot += q_reg[i * 4u + j] * __bfloat162float(elems[j]);
        }
        // Reduce within the row's 16 lanes (two half-warps; full mask).
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        float score = dot * inv_sqrt_d;
        // Broadcast from lane 0 of each 16-lane group.
        score = __shfl_sync(0xffffffffu, score, (warp_lane / 16u) * 16u);

        const float m_new = fmaxf(m_prev, score);
        const float alpha = __expf(m_prev - m_new);
        const float p = __expf(score - m_new);
        l_prev = alpha * l_prev + p;

        const __nv_bfloat16* v_row =
            V + (unsigned long long)(batch * kv_run + kv_pos) * LING_MLA_LORA;
        #pragma unroll
        for (unsigned int i = 0; i < LING_MLA_PFVVEC / 4u; ++i) {
            const uint2 pv = *reinterpret_cast<const uint2*>(v_row + lane * LING_MLA_PFVVEC + i * 4u);
            const __nv_bfloat16* elems = reinterpret_cast<const __nv_bfloat16*>(&pv);
            #pragma unroll
            for (unsigned int j = 0; j < 4u; ++j)
                o_acc[i * 4u + j] = alpha * o_acc[i * 4u + j] + p * __bfloat162float(elems[j]);
        }
        m_prev = m_new;
    }

    const float inv_l = (l_prev > 0.0f) ? 1.0f / l_prev : 0.0f;
    __nv_bfloat16* o_row =
        O + ((unsigned long long)(batch * seq_len + q_row) * gridDim.x + head) * LING_MLA_LORA;
    #pragma unroll
    for (unsigned int i = 0; i < LING_MLA_PFVVEC / 4u; ++i) {
        uint2 ov;
        __nv_bfloat16* elems = reinterpret_cast<__nv_bfloat16*>(&ov);
        #pragma unroll
        for (unsigned int j = 0; j < 4u; ++j)
            elems[j] = __float2bfloat16(o_acc[i * 4u + j] * inv_l);
        *reinterpret_cast<uint2*>(o_row + lane * LING_MLA_PFVVEC + i * 4u) = ov;
    }
}

// --------------------------------------------------------------------------------
// Tensor-core (mma.m16n8k16, bf16) prefill attention — same layout and
// signature as `ling_mla_prefill_attn`. Kept as a separate entry so the
// proven scalar kernel stays the default; env ATLAS_LING_MLA_PREFILL_MMA=1
// flips the dispatch (see `prefill_attn_contig` in mla.rs).
//
// Grid: (heads, ceil(seq_len / 16), 1)   Block: (128, 1, 1)
//
// CTA = 16 query rows (one m-tile) against 32-row KV tiles. The four warps
// split the 32-wide score tile into 8-column n-tiles for QK^T and the
// 512-dim V axis into 128-column quarters for the PV pass. Score/softmax
// are shadowed in smem; P is stored as a bf16 hi/lo split pair for the
// PV mma: a plain bf16 P carries only 8 mantissa bits against the scalar
// kernel's fp32 probabilities, which measurably flips borderline greedy
// tokens downstream (the tools e2e email-copy check fails with a plain
// bf16 P), so the PV pass runs a second accumulating mma over the lo
// residual. hi + lo costs one extra mma per step and restores ~16
// mantissa bits of P precision.
//
// K and V are streamed straight from global memory (L2/L1-cached) instead
// of staged in smem: the smem-staged design held 91,392 B per CTA, which
// capped GB10 at 1 CTA/SM (4 warps in flight). Staging only the query tile
// (18,432 B) plus the score/P/softmax shadow leaves 22,784 B per
// CTA, so shared memory alone allows 4 CTAs on a 102,400 B SM. K rows past kv_run
// are skipped outright (their scores are masked downstream anyway) and the
// Q rows past seq_len are zero-filled, so no OOB read is ever performed.
//
// Fragment layout mirrors the house SM121 workaround (no ldmatrix):
//   group_id = lane >> 2 (rows 0..7 / cols 0..7), tid_in_group = lane & 3.
// --------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128, 4) ling_mla_prefill_attn_mma(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int causal,
    const float inv_sqrt_d,
    const unsigned int kv_base
) {
    atlas_pdl_enter();
    const unsigned int head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int group_id = lane >> 2;
    const unsigned int tid_in_group = lane & 3;

    const unsigned int q_row0 = q_block * 16u;
    const unsigned int kv_run = kv_base + seq_len;

    __shared__ __nv_bfloat16 q_smem[16u * 576u];
    __shared__ float score_smem[16u * 32u];
    __shared__ __nv_bfloat16 p_smem[16u * 32u];
    __shared__ __nv_bfloat16 p_lo_smem[16u * 32u];
    // per-row m_new, l_prev, alpha (slot 3 unused)
    __shared__ float ml_smem[16u * 4u];

    // --- Q tile: [16 rows][576], zero-filled for rows >= seq_len ---
    {
        #pragma unroll 4
        for (unsigned int i = tid; i < 16u * 72u; i += 128u) {
            const unsigned int r = i / 72u;
            const unsigned int c = i % 72u;
            const unsigned int q_row = q_row0 + r;
            uint4 dst = make_uint4(0u, 0u, 0u, 0u);
            if (q_row < seq_len) {
                const __nv_bfloat16* src =
                    Q + ((unsigned long long)(batch * seq_len + q_row) * gridDim.x + head) * 576u + c * 8u;
                dst = *reinterpret_cast<const uint4*>(src);
            }
            *reinterpret_cast<uint4*>(&q_smem[r * 576u + c * 8u]) = dst;
        }
    }
    if (tid < 16u) {
        ml_smem[tid * 4u + 0] = -1e30f;
        ml_smem[tid * 4u + 1] = 0.0f;
    }
    __syncthreads();

    float o_acc[64];
    #pragma unroll
    for (unsigned int i = 0; i < 64u; ++i) o_acc[i] = 0.0f;

    // CTA-wide KV loop bound: the highest valid row's causal limit.
    const unsigned int q_row_hi = min(q_row0 + 15u, seq_len - 1u);
    const unsigned int kv_limit =
        causal ? (kv_base + q_row_hi + 1u) : kv_run;

    for (unsigned int tile_start = 0u; tile_start < kv_limit; tile_start += 32u) {
        // --- QK^T (4 warps; warp w covers tile cols [w*8, w*8+8)) ---
        float acc_s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        {
            const unsigned short* q_u16 = reinterpret_cast<const unsigned short*>(q_smem);
            const unsigned short* k_u16 = reinterpret_cast<const unsigned short*>(K);
            const unsigned int n_col = warp * 8u + group_id;
            // K row this warp's n-tile reads (0..31 within the tile). Rows
            // past kv_run are skipped; their scores are masked downstream.
            const unsigned int k_row = tile_start + n_col;
            const bool k_row_ok = k_row < kv_run;
            const unsigned int k_row_c = min(k_row, kv_run - 1u);
            const unsigned long long k_row_off =
                ((unsigned long long)(batch * kv_run + k_row_c)) * 576u;
            #pragma unroll 4
            for (unsigned int kc = 0; kc < 36u; ++kc) {
                const unsigned int k_base = kc * 16u;
                const unsigned int ac0 = k_base + tid_in_group * 2u;
                const unsigned int ac1 = ac0 + 8u;
                const unsigned int a0 =
                    *reinterpret_cast<const unsigned int*>(&q_u16[group_id * 576u + ac0]);
                const unsigned int a1 =
                    *reinterpret_cast<const unsigned int*>(&q_u16[(group_id + 8u) * 576u + ac0]);
                const unsigned int a2 =
                    *reinterpret_cast<const unsigned int*>(&q_u16[group_id * 576u + ac1]);
                const unsigned int a3 =
                    *reinterpret_cast<const unsigned int*>(&q_u16[(group_id + 8u) * 576u + ac1]);
                unsigned int b0 = 0u, b1 = 0u;
                if (k_row_ok) {
                    b0 = *reinterpret_cast<const unsigned int*>(
                        &k_u16[k_row_off + k_base + tid_in_group * 2u]);
                    b1 = *reinterpret_cast<const unsigned int*>(
                        &k_u16[k_row_off + k_base + tid_in_group * 2u + 8u]);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(acc_s[0]), "+f"(acc_s[1]), "+f"(acc_s[2]), "+f"(acc_s[3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            #pragma unroll
            for (unsigned int i = 0; i < 4u; ++i) acc_s[i] *= inv_sqrt_d;
        }
        {
            // float2 pairs: cols tid_in_group*2 and +1 within the warp's n-tile
            const unsigned int base = warp * 8u + tid_in_group * 2u;
            *reinterpret_cast<float2*>(&score_smem[group_id * 32u + base]) =
                make_float2(acc_s[0], acc_s[1]);
            *reinterpret_cast<float2*>(&score_smem[(group_id + 8u) * 32u + base]) =
                make_float2(acc_s[2], acc_s[3]);
        }
        __syncthreads();

        // --- online softmax; writes P (bf16) to smem; warp 0, lanes 0-15 ---
        if (warp == 0 && lane < 16u) {
            const unsigned int row = lane;
            const unsigned int q_row = q_row0 + row;
            const bool row_valid = q_row < seq_len;
            const unsigned int limit =
                causal ? (kv_base + q_row + 1u) : kv_run;
            float rmax = -1e30f;
            #pragma unroll 4
            for (unsigned int c = 0; c < 32u; ++c) {
                const unsigned int pos = tile_start + c;
                const bool valid = row_valid && (pos < limit);
                rmax = fmaxf(rmax, valid ? score_smem[row * 32u + c] : -1e30f);
            }
            const float m_prev = ml_smem[row * 4u + 0];
            const float m_new = fmaxf(m_prev, rmax);
            const float alpha = __expf(m_prev - m_new);
            const float l_prev = ml_smem[row * 4u + 1];
            float l_tile = 0.0f;
            #pragma unroll 4
            for (unsigned int c = 0; c < 32u; ++c) {
                const unsigned int pos = tile_start + c;
                const bool valid = row_valid && (pos < limit);
                const float p =
                    valid ? __expf(score_smem[row * 32u + c] - m_new) : 0.0f;
                const __nv_bfloat16 p_hi = __float2bfloat16(p);
                p_smem[row * 32u + c] = p_hi;
                p_lo_smem[row * 32u + c] =
                    __float2bfloat16(p - __bfloat162float(p_hi));
                l_tile += p;
            }
            ml_smem[row * 4u + 0] = m_new;
            ml_smem[row * 4u + 1] = alpha * l_prev + l_tile;
            ml_smem[row * 4u + 2] = alpha;
        }
        __syncthreads();

        // --- rescale running accumulator, then PV mma (all warps) ---
        {
            const float alpha0 = ml_smem[group_id * 4u + 2];
            const float alpha1 = ml_smem[(group_id + 8u) * 4u + 2];
            #pragma unroll
            for (unsigned int nt = 0; nt < 16u; ++nt) {
                o_acc[nt * 4u + 0] = alpha0 * o_acc[nt * 4u + 0];
                o_acc[nt * 4u + 1] = alpha0 * o_acc[nt * 4u + 1];
                o_acc[nt * 4u + 2] = alpha1 * o_acc[nt * 4u + 2];
                o_acc[nt * 4u + 3] = alpha1 * o_acc[nt * 4u + 3];
            }
        }
        {
            const unsigned short* p_u16 = reinterpret_cast<const unsigned short*>(p_smem);
            const unsigned short* p_lo_u16 = reinterpret_cast<const unsigned short*>(p_lo_smem);
            const unsigned short* v_u16 = reinterpret_cast<const unsigned short*>(V);
            #pragma unroll
            for (unsigned int nt = 0; nt < 16u; ++nt) {
                const unsigned int vdim = warp * 128u + nt * 8u + group_id;
                #pragma unroll
                for (unsigned int ks = 0; ks < 2u; ++ks) {
                    const unsigned int k_off = ks * 16u;
                    const unsigned int k0 = k_off + tid_in_group * 2u;
                    const unsigned int k1 = k0 + 8u;
                    const unsigned int g_k0 = tile_start + k0;
                    const unsigned int g_k1 = tile_start + k1;
                    const bool ok0 = g_k0 < kv_run;
                    const bool ok1 = (g_k0 + 1u) < kv_run;
                    const bool ok2 = g_k1 < kv_run;
                    const bool ok3 = (g_k1 + 1u) < kv_run;
                    // Clamp addresses for rows past kv_run; the values are
                    // discarded by the ok-predicates, only the address needs
                    // to stay in bounds.
                    const unsigned int g_k0c = min(g_k0, kv_run - 1u);
                    const unsigned int g_k1c = min(g_k1, kv_run - 1u);
                    const unsigned short* v_base =
                        v_u16 + ((unsigned long long)(batch * kv_run + g_k0c)) * 512u + vdim;
                    const unsigned short* v_base1 =
                        v_u16 + ((unsigned long long)(batch * kv_run + min(g_k0 + 1u, kv_run - 1u))) * 512u + vdim;
                    const unsigned short* v_base2 =
                        v_u16 + ((unsigned long long)(batch * kv_run + g_k1c)) * 512u + vdim;
                    const unsigned short* v_base3 =
                        v_u16 + ((unsigned long long)(batch * kv_run + min(g_k1 + 1u, kv_run - 1u))) * 512u + vdim;
                    unsigned int b0 = 0u, b1 = 0u;
                    if (ok0) b0 |= (unsigned int)v_base[0];
                    if (ok1) b0 |= (unsigned int)v_base1[0] << 16;
                    if (ok2) b1 |= (unsigned int)v_base2[0];
                    if (ok3) b1 |= (unsigned int)v_base3[0] << 16;
                    const unsigned int ac0 = k_off + tid_in_group * 2u;
                    const unsigned int ac1 = ac0 + 8u;
                    const unsigned int a0 =
                        *reinterpret_cast<const unsigned int*>(&p_u16[group_id * 32u + ac0]);
                    const unsigned int a1 =
                        *reinterpret_cast<const unsigned int*>(&p_u16[(group_id + 8u) * 32u + ac0]);
                    const unsigned int a2 =
                        *reinterpret_cast<const unsigned int*>(&p_u16[group_id * 32u + ac1]);
                    const unsigned int a3 =
                        *reinterpret_cast<const unsigned int*>(&p_u16[(group_id + 8u) * 32u + ac1]);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(o_acc[nt * 4u + 0]), "+f"(o_acc[nt * 4u + 1]),
                          "+f"(o_acc[nt * 4u + 2]), "+f"(o_acc[nt * 4u + 3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    const unsigned int l0 =
                        *reinterpret_cast<const unsigned int*>(&p_lo_u16[group_id * 32u + ac0]);
                    const unsigned int l1 =
                        *reinterpret_cast<const unsigned int*>(&p_lo_u16[(group_id + 8u) * 32u + ac0]);
                    const unsigned int l2 =
                        *reinterpret_cast<const unsigned int*>(&p_lo_u16[group_id * 32u + ac1]);
                    const unsigned int l3 =
                        *reinterpret_cast<const unsigned int*>(&p_lo_u16[(group_id + 8u) * 32u + ac1]);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(o_acc[nt * 4u + 0]), "+f"(o_acc[nt * 4u + 1]),
                          "+f"(o_acc[nt * 4u + 2]), "+f"(o_acc[nt * 4u + 3])
                        : "r"(l0), "r"(l1), "r"(l2), "r"(l3), "r"(b0), "r"(b1));
                }
            }
        }
        __syncthreads();
    }

    // --- final write: rows [0,16) of this CTA, all warps split V dims ---
    {
        const float l0 = ml_smem[group_id * 4u + 1];
        const float l1 = ml_smem[(group_id + 8u) * 4u + 1];
        const float inv_l0 = (l0 > 0.0f) ? 1.0f / l0 : 0.0f;
        const float inv_l1 = (l1 > 0.0f) ? 1.0f / l1 : 0.0f;
        const unsigned int q_rowA = q_row0 + group_id;
        const unsigned int q_rowB = q_row0 + group_id + 8u;
        #pragma unroll
        for (unsigned int nt = 0; nt < 16u; ++nt) {
            const unsigned int col0 = warp * 128u + nt * 8u + tid_in_group * 2u;
            if (q_rowA < seq_len) {
                const unsigned int lo =
                    (unsigned int)__bfloat16_as_ushort(__float2bfloat16(o_acc[nt * 4u + 0] * inv_l0));
                const unsigned int hi =
                    (unsigned int)__bfloat16_as_ushort(__float2bfloat16(o_acc[nt * 4u + 1] * inv_l0));
                *reinterpret_cast<unsigned int*>(
                    O + ((unsigned long long)(batch * seq_len + q_rowA) * gridDim.x + head) * 512u + col0) =
                    lo | (hi << 16);
            }
            if (q_rowB < seq_len) {
                const unsigned int lo =
                    (unsigned int)__bfloat16_as_ushort(__float2bfloat16(o_acc[nt * 4u + 2] * inv_l1));
                const unsigned int hi =
                    (unsigned int)__bfloat16_as_ushort(__float2bfloat16(o_acc[nt * 4u + 3] * inv_l1));
                *reinterpret_cast<unsigned int*>(
                    O + ((unsigned long long)(batch * seq_len + q_rowB) * gridDim.x + head) * 512u + col0) =
                    lo | (hi << 16);
            }
        }
    }
}

// Refill the packed prefill context's prefix rows [0, rows) from the paged
// pools, through the block table the paged attention path itself walks: row
// `pos`'s physical pool row is table[pos / block_size] * block_size +
// pos % block_size. One 2-byte element a thread; the pass is a K/V copy of
// a few MB at ISL 4096.
//
// Grid: (ceil(rows * (576 + 512) / 256), 1, 1) Block: (256, 1, 1)
extern "C" __global__ void ling_mla_ctx_gather(
    const __nv_bfloat16* __restrict__ k_cache,   // [pool, block_size, 576]
    const __nv_bfloat16* __restrict__ v_cache,   // [pool, block_size, 512]
    __nv_bfloat16* __restrict__ k_ctx,           // [rows, 576]
    __nv_bfloat16* __restrict__ v_ctx,           // [rows, 512]
    const int* __restrict__ block_tables,
    const unsigned int max_blocks_per_seq,
    const unsigned int rows,
    const unsigned int block_size
) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int k_elems = rows * LING_MLA_KV_DIM;
    const unsigned int v_elems = rows * LING_MLA_LORA;
    if (i >= k_elems + v_elems) return;
    const unsigned int pos = (i < k_elems) ? i / LING_MLA_KV_DIM
                                          : (i - k_elems) / LING_MLA_LORA;
    const unsigned int off = (i < k_elems) ? i % LING_MLA_KV_DIM
                                           : (i - k_elems) % LING_MLA_LORA;
    const int table_entry = block_tables[pos / block_size];
    const unsigned long long pool_row =
        (unsigned long long)table_entry * block_size + pos % block_size;
    if (i < k_elems) {
        k_ctx[i] = k_cache[pool_row * LING_MLA_KV_DIM + off];
    } else {
        v_ctx[i - k_elems] = v_cache[pool_row * LING_MLA_LORA + off];
    }
}
