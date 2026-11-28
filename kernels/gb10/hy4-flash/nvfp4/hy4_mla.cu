// SPDX-License-Identifier: AGPL-3.0-only

#include "hy4_gemv.cuh"
#include "hy4_fp8.cuh"
#include "../../common/quant_rowwise_fp8.cu"
#include <math.h>

extern "C" __global__ void hy4_head_gemv(const __nv_bfloat16* x,
    const __nv_bfloat16* w, __nv_bfloat16* y, unsigned n, unsigned k,
    unsigned input_stride, unsigned output_stride) {
    unsigned row=blockIdx.x*8+threadIdx.x/32,head=blockIdx.y;
    if (row>=n) return;
    float value=hy4_dense_dot(x+(size_t)head*input_stride,
        w+(size_t)head*n*k,row,k,0);
    if (threadIdx.x%32==0) y[(size_t)head*output_stride+row]=__float2bfloat16_rn(value);
}

extern "C" __global__ void hy4_rotary(const __nv_bfloat16* x, __nv_bfloat16* y,
    unsigned in_stride, unsigned out_stride, unsigned in_offset, unsigned out_offset,
    unsigned long long pt, unsigned long long ph, unsigned long long pw,
    float theta, unsigned sh, unsigned sw, unsigned interleaved, unsigned split) {
    unsigned i=threadIdx.x, head=blockIdx.x;
    if (i>=32) return;
    unsigned axis=interleaved?((i%3==1 && i<sh*3)?1:((i%3==2 && i<sw*3)?2:0)):
        (i<32-sh-sw?0:(i<32-sw?1:2));
    float p=(float)(axis==0?pt:(axis==1?ph:pw));
    float angle=p*powf(theta,-(float)i/32.f),s,c;
    sincosf(angle,&s,&c);
    const __nv_bfloat16* src=x+(size_t)head*in_stride+in_offset;
    float a=(float)src[i*2],b=(float)src[i*2+1];
    __nv_bfloat16* dst=y+(size_t)head*out_stride+out_offset;
    // A separate barrier permits the indexer's in-place pair-to-half shuffle.
    __syncthreads();
    dst[split?i:i*2]=__float2bfloat16_rn(a*c-b*s);
    dst[split?32+i:i*2+1]=__float2bfloat16_rn(a*s+b*c);
}

extern "C" __global__ void hy4_gate(__nv_bfloat16* out,
    const __nv_bfloat16* logits, unsigned width) {
    unsigned d=blockIdx.x*256+threadIdx.x;
    if (d<width) {
        float g=(float)logits[d];
        float s=g>=0.f?1.f/(1.f+expf(-g)):expf(g)/(1.f+expf(g));
        out[d]=__float2bfloat16_rn((float)out[d]*s);
    }
}

// FP8 latent-slot layout: dims 0..512 are E4M3 bytes with four FP32 scales
// per slot (one per 128 dims); dims 512..576 stay BF16 at byte offset 512.
// The BF16 path keeps the original vectorized loads.
template<bool FP8>
__device__ __forceinline__ float hy4_mla_kv(const unsigned char* cache,
    const float* scales, size_t row, unsigned kv_stride, unsigned d) {
    const unsigned char* base=cache+row*kv_stride;
    if constexpr (FP8) {
        if (d<512) {
            __nv_fp8_e4m3 v; v.__x=base[d];
            return (float)v*scales[4*row+d/128];
        }
        return (float)((const __nv_bfloat16*)(base+512))[d-512];
    } else {
        return (float)((const __nv_bfloat16*)base)[d];
    }
}

// Each head/128-key partition produces an unnormalized FP32 softmax partial.
// `cache` is the latent base in bytes; `kv_stride` is the per-slot byte stride
// (1152 bf16 / 640 fp8).
template<bool FP8>
__device__ __forceinline__ void hy4_mla_partial_impl(const __nv_bfloat16* q,
    const unsigned char* cache, const float* scales, unsigned kv_stride,
    const unsigned* table, const int* indices,
    float* partials, unsigned position, unsigned page_size,
    unsigned columns, unsigned window, unsigned count, unsigned splits) {
    unsigned h=blockIdx.x,part=blockIdx.y,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    __shared__ float scores[128],red[256],query[576],maximum,denominator;
    __shared__ int rows[128];
    for (unsigned d=tid;d<576;d+=256) query[d]=(float)q[(size_t)h*576+d];
    __syncthreads();
    unsigned begin=window && position+1>window?position+1-window:0;
    for (unsigned j=warp;j<128;j+=8) {
        unsigned at=part*128+j;
        int logical=at<count?(indices?indices[at]:(int)(begin+at)):-1;
        bool valid=logical>=0 && (unsigned)logical<=position && (unsigned)logical>=begin;
        size_t row=0;
        if (valid) {
            unsigned page=(unsigned)logical/page_size;
            row=(size_t)table[window?page%columns:page]*page_size+(unsigned)logical%page_size;
        }
        float dot=0.f;
        if (valid) {
            if constexpr (FP8) {
                const unsigned char* base=cache+row*kv_stride;
                uint4 raw=*(const uint4*)(base+lane*16);
                const unsigned words[4]={raw.x,raw.y,raw.z,raw.w};
                float scale=scales[4*row+lane/8];
                #pragma unroll
                for (unsigned w=0;w<4;w++) {
                    #pragma unroll
                    for (unsigned j=0;j<4;j++) {
                        __nv_fp8_e4m3 v; v.__x=(unsigned char)(words[w]>>(j*8));
                        unsigned d=lane*16+w*4+j;
                        dot+=query[d]*(float)v*scale;
                    }
                }
                const __nv_bfloat16* rope=(const __nv_bfloat16*)(base+512);
                dot+=query[512+lane*2]*(float)rope[lane*2];
                dot+=query[512+lane*2+1]*(float)rope[lane*2+1];
            } else {
                const __nv_bfloat16* kv=(const __nv_bfloat16*)(cache+row*kv_stride);
                for (unsigned d=lane*8;d<576;d+=256) {
                    uint4 v=*(const uint4*)(kv+d);
                    const unsigned words[4]={v.x,v.y,v.z,v.w};
                    #pragma unroll
                    for (unsigned k=0;k<4;k++) {
                        float2 b=__bfloat1622float2(*((const __nv_bfloat162*)&words[k]));
                        dot+=query[d+k*2]*b.x;dot+=query[d+k*2+1]*b.y;
                    }
                }
            }
        }
        dot=hy4_warp_sum(dot);
        if (lane==0) { scores[j]=valid?dot*(1.f/16.f):-INFINITY;rows[j]=valid?(int)row:-1; }
    }
    __syncthreads();
    red[tid]=tid<128?scores[tid]:-INFINITY;
    for (unsigned s=128;s;s>>=1) {
        __syncthreads();
        if (tid<s) red[tid]=fmaxf(red[tid],red[tid+s]);
    }
    if (tid==0) maximum=red[0];
    __syncthreads();
    float weight=tid<128 && rows[tid]>=0?expf(scores[tid]-maximum):0.f;
    if (tid<128) scores[tid]=weight;
    red[tid]=weight;
    for (unsigned s=128;s;s>>=1) {
        __syncthreads();
        if (tid<s) red[tid]+=red[tid+s];
    }
    if (tid==0) denominator=red[0];
    __syncthreads();
    float a=0.f,b=0.f;
    for (unsigned j=0;j<128;j++) if (rows[j]>=0) {
        if constexpr (FP8) {
            a+=scores[j]*hy4_mla_kv<true>(cache,scales,(size_t)rows[j],640,tid);
            b+=scores[j]*hy4_mla_kv<true>(cache,scales,(size_t)rows[j],640,tid+256);
        } else {
            const __nv_bfloat16* kv=(const __nv_bfloat16*)(cache+(size_t)rows[j]*kv_stride);
            a+=scores[j]*(float)kv[tid];b+=scores[j]*(float)kv[tid+256];
        }
    }
    float* out=partials+((size_t)h*splits+part)*514;
    out[tid]=a;out[tid+256]=b;
    if (tid==0) {out[512]=maximum;out[513]=denominator;}
}

extern "C" __global__ void hy4_mla_partial(const __nv_bfloat16* q,
    const __nv_bfloat16* cache,const unsigned* table,const int* indices,
    float* partials,unsigned position,unsigned page_size,
    unsigned columns,unsigned window,unsigned count,unsigned splits) {
    hy4_mla_partial_impl<false>(q,(const unsigned char*)cache,nullptr,1152,
        table,indices,partials,position,page_size,columns,window,count,splits);
}

// FP8 KV variant: same args plus the trailing rowwise-scale buffer; the latent
// cache arg uses the 640-byte fp8kv slot stride.
extern "C" __global__ void hy4_mla_partial_fp8kv(const __nv_bfloat16* q,
    const unsigned char* cache,const unsigned* table,const int* indices,
    float* partials,unsigned position,unsigned page_size,
    unsigned columns,unsigned window,unsigned count,unsigned splits,
    const float* scales) {
    hy4_mla_partial_impl<true>(q,cache,scales,640,
        table,indices,partials,position,page_size,columns,window,count,splits);
}

extern "C" __global__ void hy4_mla_batch_partial(const __nv_bfloat16* q,
    const __nv_bfloat16* cache,const unsigned* table,const int* indices,
    float* partials,unsigned position,unsigned page_size,
    unsigned columns,unsigned window,unsigned count,unsigned splits,
    unsigned heads,unsigned index_stride) {
    unsigned row=blockIdx.z;position+=row;
    count=window?min(position+1,window):min(position/4,index_stride/4-1)*4+position%4+1;
    hy4_mla_partial_impl<false>(q+(size_t)row*heads*576,(const unsigned char*)cache,
        nullptr,1152,table,
        indices?indices+(size_t)row*index_stride:nullptr,
        partials+(size_t)row*heads*splits*514,position,page_size,columns,window,count,splits);
}

extern "C" __global__ void hy4_mla_batch_partial_fp8kv(const __nv_bfloat16* q,
    const unsigned char* cache,const unsigned* table,const int* indices,
    float* partials,unsigned position,unsigned page_size,
    unsigned columns,unsigned window,unsigned count,unsigned splits,
    unsigned heads,unsigned index_stride,const float* scales) {
    unsigned row=blockIdx.z;position+=row;
    count=window?min(position+1,window):min(position/4,index_stride/4-1)*4+position%4+1;
    hy4_mla_partial_impl<true>(q+(size_t)row*heads*576,cache,scales,640,table,
        indices?indices+(size_t)row*index_stride:nullptr,
        partials+(size_t)row*heads*splits*514,position,page_size,columns,window,count,splits);
}

// The zero-value sink participates once, including when every key is masked.
__device__ __forceinline__ void hy4_mla_merge_impl(const float* partials,const float* sink,
    __nv_bfloat16* out,unsigned splits) {
    unsigned h=blockIdx.x,d=threadIdx.x;
    const float* parts=partials+(size_t)h*splits*514;
    float maximum=sink[h];
    for (unsigned p=0;p<splits;p++) maximum=fmaxf(maximum,parts[p*514+512]);
    float denominator=expf(sink[h]-maximum),a=0.f,b=0.f;
    for (unsigned p=0;p<splits;p++) {
        float scale=expf(parts[p*514+512]-maximum);
        denominator+=scale*parts[p*514+513];
        a+=scale*parts[p*514+d];b+=scale*parts[p*514+d+256];
    }
    out[(size_t)h*512+d]=__float2bfloat16_rn(a/denominator);
    out[(size_t)h*512+d+256]=__float2bfloat16_rn(b/denominator);
}

extern "C" __global__ void hy4_mla_merge(const float* partials,const float* sink,
    __nv_bfloat16* out,unsigned splits) {
    hy4_mla_merge_impl(partials,sink,out,splits);
}
extern "C" __global__ void hy4_mla_batch_merge(const float* partials,const float* sink,
    __nv_bfloat16* out,unsigned splits,unsigned heads) {
    unsigned row=blockIdx.y;
    hy4_mla_merge_impl(partials+(size_t)row*heads*splits*514,sink,
        out+(size_t)row*heads*512,splits);
}

// ── Batched causal prologue (speculative verify / decode batching) ──────
// Same math as hy4_norm_add (the a.norm call with H=512) and hy4_rotary,
// one launch per layer instead of three launches per row. Outputs are
// byte-identical: each row block replicates the single-row reduction and
// rope arithmetic in the same order.
extern "C" __global__ void hy4_batch_kv_norm(const __nv_bfloat16* kv,
    const __nv_bfloat16* w, __nv_bfloat16* latent, const unsigned* table,
    unsigned start, unsigned page_size, unsigned columns, unsigned circular,
    float eps) {
    unsigned row=blockIdx.x, tid=threadIdx.x;
    unsigned pos=start+row, page=pos/page_size;
    unsigned column=circular?page%columns:page;
    size_t slot=(size_t)table[column]*page_size+pos%page_size;
    const __nv_bfloat16* x=kv+(size_t)row*576;
    __nv_bfloat16* out=latent+slot*576;
    __shared__ float red[256], r;
    float ss=0.f;
    for (unsigned d=tid; d<512; d+=256) { float v=(float)x[d]; ss+=v*v; }
    red[tid]=ss;
    __syncthreads();
    for (unsigned s=128;s;s>>=1) {
        if (tid<s) red[tid]+=red[tid+s];
        __syncthreads();
    }
    if (tid==0) r=rsqrtf(red[0]/512+eps);
    __syncthreads();
    for (unsigned d=tid; d<512; d+=256)
        out[d]=__float2bfloat16_rn((float)x[d]*r*(float)w[d]);
}

// rows share one (in_stride,out_stride,in_offset,out_offset) layout via the
// row_in/row_out element strides; paged=1 targets latent slot rows through
// the page table instead of contiguous output rows.
// ── FP8 (E4M3) latent KV cache writes ────────────────────────────────
// An fp8 latent slot is 640 bytes: dims 0..512 as E4M3, then dims 512..576
// as BF16. Four FP32 scales per slot cover 128-dim tiles of the first half.
__device__ __forceinline__ unsigned char hy4_enc_e4m3(float v) {
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
    return rw_enc_fp8(v);
#else
    return (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#endif
}
__device__ __forceinline__ float hy4_fp8_row_scale(float amax) {
    float s=amax/448.f;
    return s<1e-12f?1e-12f:s;
}

// RMSNorm over dims 0..512 (same math as hy4_batch_kv_norm), then tilewise
// E4M3 quantization of the normed values into the fp8 latent slot.
__device__ __forceinline__ void hy4_kv_norm_fp8_impl(const __nv_bfloat16* x,
    const __nv_bfloat16* w, unsigned char* out, float* scale_out, float eps) {
    unsigned tid=threadIdx.x;
    __shared__ float red[256], r, amax[4], sc[4];
    float ss=0.f;
    for (unsigned d=tid; d<512; d+=256) { float v=(float)x[d]; ss+=v*v; }
    red[tid]=ss;
    if (tid<4) amax[tid]=0.f;
    __syncthreads();
    for (unsigned half=128;half;half>>=1) {
        if (tid<half) red[tid]+=red[tid+half];
        __syncthreads();
    }
    if (tid==0) r=rsqrtf(red[0]/512+eps);
    __syncthreads();
    for (unsigned d=tid; d<512; d+=256) {
        float v=(float)x[d]*r*(float)w[d];
        atomicMax((unsigned*)&amax[d/128],__float_as_uint(fabsf(v)));
    }
    __syncthreads();
    if (tid<4) {
        sc[tid]=hy4_fp8_row_scale(amax[tid]);
        scale_out[tid]=sc[tid];
    }
    __syncthreads();
    for (unsigned d=tid; d<512; d+=256) {
        float v=(float)x[d]*r*(float)w[d];
        out[d]=hy4_enc_e4m3(v/sc[d/128]);
    }
}

// Rotary outputs remain BF16 in the fp8kv slot at byte offset out_offset.
// Block must be exactly 32 threads, matching hy4_rotary.
__device__ __forceinline__ void hy4_rotary_fp8kv_impl(const __nv_bfloat16* src,
    __nv_bfloat16* dst, unsigned long long pt, unsigned long long ph,
    unsigned long long pw, float theta, unsigned sh, unsigned sw,
    unsigned interleaved, unsigned split) {
    unsigned i=threadIdx.x;
    if (i>=32) return;
    unsigned axis=interleaved?((i%3==1 && i<sh*3)?1:((i%3==2 && i<sw*3)?2:0)):
        (i<32-sh-sw?0:(i<32-sw?1:2));
    float p=(float)(axis==0?pt:(axis==1?ph:pw));
    float angle=p*powf(theta,-(float)i/32.f),s,c;
    sincosf(angle,&s,&c);
    float a=(float)src[i*2],b=(float)src[i*2+1];
    __syncthreads();
    dst[split?i:i*2]=__float2bfloat16_rn(a*c-b*s);
    dst[split?32+i:i*2+1]=__float2bfloat16_rn(a*s+b*c);
}

// Batched fp8kv variant of hy4_batch_kv_norm: same args plus the trailing
// scale buffer; latent uses 640-byte slots.
extern "C" __global__ void hy4_batch_kv_norm_fp8kv(const __nv_bfloat16* kv,
    const __nv_bfloat16* w, unsigned char* latent8, const unsigned* table,
    unsigned start, unsigned page_size, unsigned columns, unsigned circular,
    float eps, float* scales) {
    unsigned row=blockIdx.x;
    unsigned pos=start+row, page=pos/page_size;
    unsigned column=circular?page%columns:page;
    size_t slot=(size_t)table[column]*page_size+pos%page_size;
    hy4_kv_norm_fp8_impl(kv+(size_t)row*576, w, latent8+slot*640, scales+4*slot, eps);
}

// Paged fp8kv variant of hy4_batch_rotary. out_offset is a byte offset within
// the 640-byte latent slot (512 for the rope half).
extern "C" __global__ void hy4_batch_rotary_fp8kv(const __nv_bfloat16* x,
    unsigned char* latent8, unsigned in_stride, unsigned out_stride,
    unsigned in_offset, unsigned out_offset, unsigned row_in, unsigned row_out,
    const unsigned* table, unsigned start, unsigned page_size,
    unsigned columns, unsigned circular, unsigned paged,
    unsigned long long pt, unsigned long long ph, unsigned long long pw,
    float theta, unsigned sh, unsigned sw, unsigned interleaved, unsigned split) {
    unsigned head=blockIdx.x, row=blockIdx.y;
    unsigned pos=start+row, page=pos/page_size;
    unsigned column=circular?page%columns:page;
    size_t slot=(size_t)table[column]*page_size+pos%page_size;
    const __nv_bfloat16* src=x+(size_t)row*row_in+(size_t)head*in_stride+in_offset;
    hy4_rotary_fp8kv_impl(src, (__nv_bfloat16*)(latent8+slot*640+out_offset),
        pt+row, ph+row, pw+row, theta, sh, sw, interleaved, split);
}

// Decode single-row fp8kv latent writes. Norm quantizes dims 0..512; rotary
// keeps dims 512..576 in BF16 at byte offset 512.
extern "C" __global__ void hy4_kv_norm_fp8kv(const __nv_bfloat16* kv,
    const __nv_bfloat16* w, unsigned char* latent8, float* scales,
    unsigned slot, float eps) {
    hy4_kv_norm_fp8_impl(kv, w, latent8+(size_t)slot*640, scales+4*slot, eps);
}

extern "C" __global__ void hy4_rotary_fp8kv(const __nv_bfloat16* x,
    unsigned char* latent8, unsigned slot,
    unsigned in_stride, unsigned in_offset, unsigned out_offset,
    unsigned long long pt, unsigned long long ph, unsigned long long pw,
    float theta, unsigned sh, unsigned sw, unsigned interleaved, unsigned split) {
    unsigned head=blockIdx.x;
    const __nv_bfloat16* src=x+(size_t)head*in_stride+in_offset;
    hy4_rotary_fp8kv_impl(src,
        (__nv_bfloat16*)(latent8+(size_t)slot*640+out_offset),
        pt, ph, pw, theta, sh, sw, interleaved, split);
}

extern "C" __global__ void hy4_batch_rotary(const __nv_bfloat16* x, __nv_bfloat16* y,
    unsigned in_stride, unsigned out_stride, unsigned in_offset, unsigned out_offset,
    unsigned row_in, unsigned row_out, const unsigned* table,
    unsigned start, unsigned page_size, unsigned columns, unsigned circular,
    unsigned paged, unsigned long long pt, unsigned long long ph, unsigned long long pw,
    float theta, unsigned sh, unsigned sw, unsigned interleaved, unsigned split) {
    unsigned i=threadIdx.x, head=blockIdx.x, row=blockIdx.y;
    if (i>=32) return;
    unsigned axis=interleaved?((i%3==1 && i<sh*3)?1:((i%3==2 && i<sw*3)?2:0)):
        (i<32-sh-sw?0:(i<32-sw?1:2));
    float p=(float)(axis==0?pt+row:(axis==1?ph+row:pw+row));
    float angle=p*powf(theta,-(float)i/32.f),s,c;
    sincosf(angle,&s,&c);
    const __nv_bfloat16* src=x+(size_t)row*row_in+(size_t)head*in_stride+in_offset;
    float a=(float)src[i*2],b=(float)src[i*2+1];
    __nv_bfloat16* dst;
    if (paged) {
        unsigned pos=start+row, page=pos/page_size;
        unsigned column=circular?page%columns:page;
        dst=y+((size_t)table[column]*page_size+pos%page_size)*576+out_offset;
    } else {
        dst=y+(size_t)row*row_out+(size_t)head*out_stride+out_offset;
    }
    // A separate barrier permits the indexer's in-place pair-to-half shuffle.
    __syncthreads();
    dst[split?i:i*2]=__float2bfloat16_rn(a*c-b*s);
    dst[split?32+i:i*2+1]=__float2bfloat16_rn(a*s+b*c);
}

#include "hy4_mla_mma.cuh"
