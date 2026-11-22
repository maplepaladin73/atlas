// SPDX-License-Identifier: AGPL-3.0-only
// Parity + bench: hy4_batch_expert (proven scalar) vs
// hy4_batch_expert_mma_fp8 (e4m3 mma.sync m16n8k32, ATLAS_HY4_EXPERT_MMA=2).
// Standalone on purpose: it does NOT include hy4_ffn.cu (that pulls in the
// mxf4nvf4 kernel, which ptxas rejects in SASS mode for sm_121a/f), so this
// harness SASS-compiles the e4m3 path: nvcc -O3 -std=c++17 -arch=sm_121a.
// Includes the null/remote-expert (EP) case and both phases.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>
#include <cstring>
#include <vector>
#include <cstdio>
#include <cstdlib>

// --- minimal copies of the hy4_ffn.cu preamble (struct/LUT/helpers) ---
struct Hy4Q4 { const uint8_t* w; const uint8_t* s; float scale; uint32_t pad; };
static_assert(sizeof(Hy4Q4)==24, "Hy4 descriptor ABI");
__device__ __constant__ float HY4_FP4[16]={
    0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_gemv.cuh"
__device__ __forceinline__ float hy4_qdot(const __nv_bfloat16* x,Hy4Q4 q,
    unsigned row,unsigned k,const float* lut) {
    float sum=0.f;
    if (q.w) for (unsigned col=(threadIdx.x%32)*32;col<k;col+=1024) {
        unsigned count=min(32u,k-col);
        uint4 packed;
        if (count==32 && k%32==0) packed=*(const uint4*)(q.w+(size_t)row*(k/2)+col/2);
        else {
            const uint2* p=(const uint2*)(q.w+(size_t)row*(k/2)+col/2);
            const uint2 lo=p[0],hi=count==32?p[1]:make_uint2(0,0);
            packed=make_uint4(lo.x,lo.y,hi.x,hi.y);
        }
        const unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
        for (unsigned group=0;group<count/16;group++) {
            __nv_fp8_e4m3 sf;sf.__x=q.s[(size_t)row*(k/16)+col/16+group];
            float scale=(float)sf*q.scale;
            #pragma unroll
            for (unsigned v=0;v<2;v++) {
                uint4 a=*(const uint4*)(x+col+group*16+v*8);
                const unsigned ar[4]={a.x,a.y,a.z,a.w};
                unsigned w=words[group*2+v];
                #pragma unroll
                for (unsigned j=0;j<4;j++) {
                    float2 values=__bfloat1622float2(*((const __nv_bfloat162*)&ar[j]));
                    sum+=(lut[(w>>(j*8))&15]*scale)*values.x;
                    sum+=(lut[(w>>(j*8+4))&15]*scale)*values.y;
                }
            }
        }
    }
    return hy4_warp_sum(sum);
}
__device__ float hy4_sigmoid(float v) {
    if (v >= 0.f) return 1.f / (1.f + expf(-v));
    float e = expf(v); return e / (1.f + e);
}
__device__ float hy4_round(float x) {return (float)__float2bfloat16_rn(x);}
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_batch.cuh"
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_batch_mma_fp8.cuh"

#define CUDA(call) do {auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1);}}while(0)
template<class T> struct Buffer {
    T* p;size_t n;
    Buffer(size_t count):n(count){CUDA(cudaMalloc(&p,n*sizeof(T)));CUDA(cudaMemset(p,0,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<T>& v){CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);CUDA(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};

int main(int argc, char** argv) {
    // Production shapes: latent/hidden 3072, inter 1536.
    unsigned latent=3072, inter=1536, e=128, slots=16, rows=256;
    int bench=0, iters=200;
    for (int i=1;i<argc;i++){
        if (!strcmp(argv[i],"bench")) bench=1;
        else if (!strcmp(argv[i],"iter")&&i+1<argc) iters=atoi(argv[++i]);
        else {
            static int pos=0;
            unsigned v=(unsigned)atoi(argv[i]);
            switch (pos++) {
                case 0: rows=v; break;
                case 1: slots=v; break;
                case 2: e=v; break;
                case 3: latent=v; break;
                case 4: inter=v; break;
            }
        }
    }
    if (latent % 128 || inter % 128) {
        fprintf(stderr,"latent/inter must be multiples of 128 for the fp8 kernel\n");
        return 1;
    }

    Buffer<unsigned char>wbuf((size_t)e*inter*latent/2),sbuf((size_t)e*inter*latent/16);
    std::vector<unsigned char>aw(wbuf.n),bs(sbuf.n);
    for(size_t j=0;j<aw.size();j++)aw[j]=(j*73+19)%256;
    for(size_t j=0;j<bs.size();j++)bs[j]=0x28+j%4;
    wbuf.put(aw);sbuf.put(bs);
    std::vector<Hy4Q4>desc((size_t)e*3);
    for(unsigned j=0;j<e;j++){
        if (j % 2 == 1) {  // half the experts are "remote" (null)
            desc[j*3]={nullptr,nullptr,0.f,0};
            desc[j*3+1]={nullptr,nullptr,0.f,0};
            desc[j*3+2]={nullptr,nullptr,0.f,0};
        } else {
            desc[j*3]={wbuf.p,sbuf.p,0.113f*(j%8+1),0};
            desc[j*3+1]={wbuf.p,sbuf.p,0.311f,0};
            desc[j*3+2]={wbuf.p,sbuf.p,0.071f,0};
        }
    }
    std::vector<unsigned>ids((size_t)rows*slots);
    std::vector<__nv_bfloat16>x((size_t)rows*latent);
    for(unsigned t=0;t<rows;t++)for(unsigned s=0;s<slots;s++)ids[t*slots+s]=(t+s)%e;
    for(unsigned j=0;j<x.size();j++)x[j]=__float2bfloat16_rn((int(j%13)-6)*.125f);
    Buffer<Hy4Q4>table(desc.size());table.put(desc);
    Buffer<unsigned>di(ids.size()),groups((size_t)e*(rows+2)+1);di.put(ids);
    Buffer<__nv_bfloat16>dx(x.size()),out_a((size_t)rows*slots*inter),ref_a((size_t)rows*slots*inter),out_p((size_t)rows*slots*latent),ref_p((size_t)rows*slots*latent);
    dx.put(x);
    hy4_group_init<<<1,1>>>(groups.p,rows,e);
    hy4_group_rows<<<e,256>>>(di.p,groups.p,rows,slots);

    auto run_proven=[&](){
        hy4_batch_expert<<<dim3((inter+7)/8,1,e),256>>>(dx.p,table.p,groups.p,out_a.p,inter,latent,rows,slots,0,0,.25f,e);
        hy4_batch_expert<<<dim3((latent+7)/8,1,e),256>>>(out_a.p,table.p,groups.p,out_p.p,latent,inter,rows,slots,0,1,.25f,e);
    };
    auto run_fp8=[&](){
        // fp8 path takes bf16 activations directly (no quantize pass)
        hy4_batch_expert_mma_fp8<<<dim3((inter+127)/128,1,e),256>>>(dx.p,table.p,groups.p,ref_a.p,inter,latent,rows,slots,0,0,.25f,e);
        hy4_batch_expert_mma_fp8<<<dim3((latent+127)/128,1,e),256>>>(ref_a.p,table.p,groups.p,ref_p.p,latent,inter,rows,slots,0,1,.25f,e);
    };
    CUDA(cudaDeviceSynchronize());
    run_proven();
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaGetLastError());
    run_fp8();
    CUDA(cudaDeviceSynchronize());
    {auto err=cudaGetLastError(); if(err!=cudaSuccess){fprintf(stderr,"fp8 run: %s\n",cudaGetErrorString(err));return 1;}}
    auto go=out_a.get(), gm=ref_a.get(), gp=out_p.get(), gmp=ref_p.get();
    float ma=0.f,mp=0.f; size_t maidx=0, mpidx=0; double sa=0.,sp=0.;
    size_t na=0, np2=0;
    for(size_t j=0;j<go.size();j++){float d=fabsf(float(go[j])-float(gm[j])); if(d>ma){ma=d;maidx=j;} sa+=d; if(go[j]!=gm[j])na++;}
    for(size_t j=0;j<gp.size();j++){float d=fabsf(float(gp[j])-float(gmp[j])); if(d>mp){mp=d;mpidx=j;} sp+=d; if(gp[j]!=gmp[j])np2++;}
    printf("max_err_gate_up=%.6f (n=%zu, idx=%zu, mismatched=%zu, mean_err=%.6f)\n", ma, go.size(), maidx, na, sa/go.size());
    printf("max_err_down=%.6f (n=%zu, idx=%zu, mismatched=%zu, mean_err=%.6f)\n", mp, gp.size(), mpidx, np2, sp/gp.size());
    int null_bad = 0;
    for (unsigned t = 0; t < rows && t < 64; ++t) {
        for (unsigned s = 0; s < slots; ++s) {
            if (ids[(size_t)t*slots+s] % 2 == 0) continue;
            const size_t at = (size_t)t*slots+s;
            for (unsigned c = 0; c < inter; ++c) {
                if (*(const unsigned short*)&go[at*inter+c] || *(const unsigned short*)&gm[at*inter+c]) { null_bad++; break; }
            }
            for (unsigned c = 0; c < latent; ++c) {
                if (*(const unsigned short*)&gp[at*latent+c] || *(const unsigned short*)&gmp[at*latent+c]) { null_bad++; break; }
            }
        }
    }
    if (null_bad) { fprintf(stderr, "FAIL: %d null-expert rows non-zero\n", null_bad); return 1; }
    if (!(ma == ma) || !(mp == mp) || ma > 0.1f || mp > 0.1f) {
        fprintf(stderr, "FAIL: parity error out of band (gate_up %.6f, down %.6f)\n", ma, mp);
        return 1;
    }
    printf("null-expert zero-fill: OK (both paths, both phases)\n");
    if (bench) {
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        for(int i=0;i<10;i++) run_proven();
        CUDA(cudaDeviceSynchronize());
        cudaEventRecord(a);
        for(int i=0;i<iters;i++) run_proven();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms,a,b);
        printf("proven: %.3f ms/iter (gate_up+down, %d iters)\n", ms/iters, iters);
        for(int i=0;i<10;i++) run_fp8();
        CUDA(cudaDeviceSynchronize());
        cudaEventRecord(a);
        for(int i=0;i<iters;i++) run_fp8();
        cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms,a,b);
        printf("fp8   : %.3f ms/iter (gate_up+down, %d iters)\n", ms/iters, iters);
    }
    return 0;
}
