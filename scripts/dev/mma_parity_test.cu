// SPDX-License-Identifier: AGPL-3.0-only
// Parity test: hy4_batch_expert (proven) vs hy4_batch_expert_mma (new).
// Includes the null/remote-expert case (EP).
#include "../../kernels/gb10/hy4-flash/nvfp4/hy4_ffn.cu"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA(call) do {auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(1);}}while(0)
template<class T> struct Buffer {
    T* p;size_t n;
    Buffer(size_t count):n(count){CUDA(cudaMalloc(&p,n*sizeof(T)));CUDA(cudaMemset(p,0,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
    void put(const std::vector<T>& v){CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);CUDA(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};

int main(int argc, char** argv) {
    unsigned latent=1536, inter=1536, e=512, slots=16, rows=256;
    if (argc > 1) rows = atoi(argv[1]);
    if (argc > 2) slots = atoi(argv[2]);
    if (argc > 3) e = atoi(argv[3]);
    if (argc > 4) latent = atoi(argv[4]);
    if (argc > 5) inter = atoi(argv[5]);

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
            // Non-bf16-exact scales so the mma path's weight-rounding
            // difference is visible (bf16 mantissa has 8 bits).
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
    // proven
    hy4_batch_expert<<<dim3((inter+7)/8,1,e),256>>>(dx.p,table.p,groups.p,out_a.p,inter,latent,rows,slots,0,0,.25f,e);
    hy4_batch_expert<<<dim3((latent+7)/8,1,e),256>>>(out_a.p,table.p,groups.p,out_p.p,latent,inter,rows,slots,0,1,.25f,e);
    CUDA(cudaDeviceSynchronize());
    // mma
    hy4_batch_expert_mma<<<dim3((inter+63)/64,1,e),256>>>(dx.p,table.p,groups.p,ref_a.p,inter,latent,rows,slots,0,0,.25f,e);
    hy4_batch_expert_mma<<<dim3((latent+63)/64,1,e),256>>>(ref_a.p,table.p,groups.p,ref_p.p,latent,inter,rows,slots,0,1,.25f,e);
    CUDA(cudaDeviceSynchronize());
    auto go=out_a.get(), gm=ref_a.get(), gp=out_p.get(), gmp=ref_p.get();
    float ma=0.f,mp=0.f; size_t maidx=0, mpidx=0;
    for(size_t i=0;i<go.size();i++){float d=fabsf(float(go[i])-float(gm[i])); if(d>ma){ma=d;maidx=i;}}
    for(size_t i=0;i<gp.size();i++){float d=fabsf(float(gp[i])-float(gmp[i])); if(d>mp){mp=d;mpidx=i;}}
    printf("max_err_gate_up=%.6f (n=%zu, idx=%zu)\n", ma, go.size(), maidx);
    printf("max_err_down=%.6f (n=%zu, idx=%zu)\n", mp, gp.size(), mpidx);
    return 0;
}
