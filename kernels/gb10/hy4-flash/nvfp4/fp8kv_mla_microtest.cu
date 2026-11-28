// SPDX-License-Identifier: AGPL-3.0-only

// Standalone fp8kv latent-attention numeric smoke harness. This is a manual
// microtest, not part of the production KERNEL.toml module set. Build and run
// on a GB10 host from the repository root with:
//   nvcc -std=c++17 -arch=sm_121f -O2 \
//     kernels/gb10/hy4-flash/nvfp4/fp8kv_mla_microtest.cu \
//     -o /tmp/fp8kv_mla_microtest && /tmp/fp8kv_mla_microtest
// It checks scalar fp8 against bf16, batched fp8 against single-row fp8, and
// the fp8 MMA path against both. E4M3 quantization is intentionally lossy, so
// the bf16 comparisons report error magnitudes rather than requiring bitwise
// equality.
#include "hy4_mla.cu"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}
static float frand(unsigned& s) {
    s = s * 1664525u + 1013904223u;
    unsigned x = (s >> 9) | 0x3f800000u;
    float u = *(float*)&x - 1.f;
    return u * 2.f - 1.f;
}

int main() {
    constexpr int ROWS = 128, DIMS = 576, PAGE = 16, PAGES = 8;
    unsigned rng = 12345;
    std::vector<__nv_bfloat16> h_q(DIMS), h_kv((size_t)ROWS * DIMS);
    std::vector<unsigned char> h_bf16((size_t)ROWS * DIMS * 2), h_fp8((size_t)ROWS * 640);
    std::vector<float> h_scales(ROWS * 4);
    for (int d = 0; d < DIMS; ++d) h_q[d] = __float2bfloat16_rn(frand(rng) * 0.7f);
    for (int r = 0; r < ROWS; ++r) {
        float amax[4] = {0,0,0,0};
        std::vector<float> v(DIMS);
        for (int d = 0; d < DIMS; ++d) {
            float x = frand(rng) * (d % 97 == 0 ? 6.f : 0.8f);
            v[d] = x;
            if (d < 512) amax[d/128] = fmaxf(amax[d/128], fabsf(x));
        }
        for (int t = 0; t < 4; ++t) {
            float sc = amax[t] / 448.f;
            h_scales[r*4+t] = sc < 1e-12f ? 1e-12f : sc;
        }
        for (int d = 0; d < DIMS; ++d) {
            __nv_bfloat16 b = __float2bfloat16_rn(v[d]);
            h_kv[(size_t)r*DIMS+d] = b;
            if (d < 512) {
                h_bf16[(size_t)r*1152+d*2] = ((unsigned char*)&b)[0];
                h_bf16[(size_t)r*1152+d*2+1] = ((unsigned char*)&b)[1];
                float sc = h_scales[r*4+d/128];
                h_fp8[(size_t)r*640+d] = (unsigned char)__nv_cvt_float_to_fp8((float)b/sc, __NV_SATFINITE, __NV_E4M3);
            } else {
                h_bf16[(size_t)r*1152+d*2] = ((unsigned char*)&b)[0];
                h_bf16[(size_t)r*1152+d*2+1] = ((unsigned char*)&b)[1];
                h_fp8[(size_t)r*640+512+(d-512)*2] = ((unsigned char*)&b)[0];
                h_fp8[(size_t)r*640+512+(d-512)*2+1] = ((unsigned char*)&b)[1];
            }
        }
    }
    std::vector<unsigned> h_table(PAGES);
    for (int i = 0; i < PAGES; ++i) h_table[i] = i;

    __nv_bfloat16 *dq, *dbref;
    unsigned char *dcache8;
    unsigned *dtab;
    float *dscales, *dout_ref, *dout_fp8, *dout_batch, *dout_mma;
    ck(cudaMalloc(&dq, DIMS*2), "malloc q");
    ck(cudaMalloc(&dbref, h_bf16.size()), "malloc bf16");
    ck(cudaMalloc(&dcache8, h_fp8.size()), "malloc fp8");
    ck(cudaMalloc(&dtab, PAGES*4), "malloc table");
    ck(cudaMalloc(&dscales, h_scales.size()*4), "malloc scales");
    ck(cudaMalloc(&dout_ref, 514*4), "malloc out ref");
    ck(cudaMalloc(&dout_fp8, 514*4), "malloc out fp8");
    ck(cudaMalloc(&dout_batch, 514*4), "malloc out batch");
    ck(cudaMalloc(&dout_mma, 514*4), "malloc out mma");
    ck(cudaMemcpy(dq, h_q.data(), DIMS*2, cudaMemcpyHostToDevice), "copy q");
    ck(cudaMemcpy(dbref, h_bf16.data(), h_bf16.size(), cudaMemcpyHostToDevice), "copy bf16");
    ck(cudaMemcpy(dcache8, h_fp8.data(), h_fp8.size(), cudaMemcpyHostToDevice), "copy fp8");
    ck(cudaMemcpy(dtab, h_table.data(), PAGES*4, cudaMemcpyHostToDevice), "copy table");
    ck(cudaMemcpy(dscales, h_scales.data(), h_scales.size()*4, cudaMemcpyHostToDevice), "copy scales");
    ck(cudaMemset(dout_ref, 0, 514*4), "zero ref");
    ck(cudaMemset(dout_fp8, 0, 514*4), "zero fp8");
    ck(cudaMemset(dout_batch, 0, 514*4), "zero batch");
    ck(cudaMemset(dout_mma, 0, 514*4), "zero mma");

    hy4_mla_partial<<<dim3(1,1,1),256>>>(dq, dbref, dtab, nullptr, dout_ref, 127, PAGE, PAGES, 0, 128, 1);
    ck(cudaGetLastError(), "launch ref");
    hy4_mla_partial_fp8kv<<<dim3(1,1,1),256>>>(dq, dcache8, dtab, nullptr, dout_fp8, 127, PAGE, PAGES, 0, 128, 1, dscales);
    ck(cudaGetLastError(), "launch fp8");
    hy4_mla_batch_partial_fp8kv<<<dim3(1,1,1),256>>>(dq, dcache8, dtab, nullptr, dout_batch, 127, PAGE, PAGES, 0, 128, 1, 1, 2048, dscales);
    ck(cudaGetLastError(), "launch batch fp8");
    ck(cudaFuncSetAttribute((const void*)hy4_mla_batch_partial_mma_fp8kv, cudaFuncAttributeMaxDynamicSharedMemorySize, MLM_SMEM_BYTES), "smem attr");
    hy4_mla_batch_partial_mma_fp8kv<<<dim3(1,1,1),256,MLM_SMEM_BYTES>>>(dq, dcache8, dtab, nullptr, dout_mma, 127, PAGE, PAGES, 0, 128, 1, 1, 2048, dscales);
    ck(cudaGetLastError(), "launch mma fp8");
    ck(cudaDeviceSynchronize(), "sync");

    float out_ref[514], out_fp8[514], out_batch[514], out_mma[514];
    ck(cudaMemcpy(out_ref, dout_ref, sizeof(out_ref), cudaMemcpyDeviceToHost), "read ref");
    ck(cudaMemcpy(out_fp8, dout_fp8, sizeof(out_fp8), cudaMemcpyDeviceToHost), "read fp8");
    ck(cudaMemcpy(out_batch, dout_batch, sizeof(out_batch), cudaMemcpyDeviceToHost), "read batch");
    ck(cudaMemcpy(out_mma, dout_mma, sizeof(out_mma), cudaMemcpyDeviceToHost), "read mma");
    double ref2=0,qe2=0,me2=0,se2=0,maxq=0,maxm=0;
    for (int d = 0; d < 512; ++d) {
        double qe = out_fp8[d]-out_ref[d], me = out_mma[d]-out_ref[d], se = out_mma[d]-out_fp8[d];
        ref2 += out_ref[d]*out_ref[d]; qe2 += qe*qe; me2 += me*me; se2 += se*se;
        maxq = fmax(maxq, fabs(qe)); maxm = fmax(maxm, fabs(me));
    }
    printf("fp8_vs_bf16 rel_l2=%.6f max_abs=%.6f\n", sqrt(qe2/ref2), maxq);
    printf("mma_vs_bf16 rel_l2=%.6f max_abs=%.6f\n", sqrt(me2/ref2), maxm);
    double be2=0;
    for (int d = 0; d < 512; ++d) { double e=out_batch[d]-out_fp8[d]; be2 += e*e; }
    printf("batch_vs_single_fp8 rel_l2=%.6g\n", sqrt(be2/ref2));
    printf("mma_vs_scalar_fp8 rel_l2=%.6f scalar_maxmax=%.6g mma_maxmax=%.6g denom=%.6g/%.6g\n",
        sqrt(se2/ref2), out_fp8[512], out_mma[512], out_fp8[513], out_mma[513]);
    cudaFree(dq);cudaFree(dbref);cudaFree(dcache8);cudaFree(dtab);cudaFree(dscales);cudaFree(dout_ref);cudaFree(dout_fp8);cudaFree(dout_batch);cudaFree(dout_mma);
    return 0;
}
