// dequant_kernels.metal — REFERENCE ONLY (not build input)
//
// Recovered from the 2026-09-16 prototype revision of the Verum inference tree.
// Dequantizing matvec / row-extract kernels for Q8_0, Q4_0, Q4_K, Q3_K, Q5_K,
// Q6_K, Q2_K, Q4_1, Q5_0, Q5_1, Q6_0 (plus _amd discrete-GPU variants), plus
// rms_norm, rope, kv_cache_store, GQA attention, silu_mul, residual_add,
// embed/argmax helpers.
//
// NOT numerically validated against ggml — ground-truth candidate for the
// runtime MSL generator to match. q3_2/q3_3 kernels still missing.
// Parity tests required before any correctness claim.

#include <metal_stdlib>
using namespace metal;

constant uint MAX_CTX = 2048;
#define NROWS     32u
#define NROWS_Q4K 32u
#define NROWS_Q5K 32u
#define NROWS_Q6K 32u
#define NROWS_Q3K 32u

struct block_q8_0 { half d; char qs[32]; };
struct block_q4_0 { half d; uchar qs[16]; };
struct block_q4_K { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct block_q5_K { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; };
struct block_q6_K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };
struct block_q3_K { uchar hmask[32]; uchar qs[64]; uchar scales[12]; half d; };
struct argmax_partial { float val; uint idx; };

static inline uint2 qK_scale_min(device const uchar *sc, uint j) {
    uint s, mn;
    if (j < 4) { s = sc[j] & 0x3Fu; mn = sc[j+4] & 0x3Fu; }
    else { s = (sc[j+4] & 0x0Fu) | ((sc[j-4] >> 6u) << 4u); mn = (sc[j+4] >> 4u) | ((sc[j] >> 6u) << 4u); }
    return uint2(s, mn);
}

static inline int q3K_sc(device const uchar *sc, uint sb) {
    const uint m1=0x03030303u, m2=0x0f0f0f0fu;
    uint a0=uint(sc[0])|(uint(sc[1])<<8)|(uint(sc[2])<<16)|(uint(sc[3])<<24);
    uint a1=uint(sc[4])|(uint(sc[5])<<8)|(uint(sc[6])<<16)|(uint(sc[7])<<24);
    uint a2=uint(sc[8])|(uint(sc[9])<<8)|(uint(sc[10])<<16)|(uint(sc[11])<<24);
    uint r[4];
    r[0]=(a0&m2)|(((a2>>0)&m1)<<4); r[1]=(a1&m2)|(((a2>>2)&m1)<<4);
    r[2]=((a0>>4)&m2)|(((a2>>4)&m1)<<4); r[3]=((a1>>4)&m2)|(((a2>>6)&m1)<<4);
    return (int)(int8_t)((r[sb>>2]>>(8u*(sb&3u)))&0xFFu);
}

kernel void q8_extract_row(device const block_q8_0 *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid>=dim) return;
    block_q8_0 b = B[row*(dim/32)+tid/32];
    o[tid] = float(b.d)*float(b.qs[tid%32]);
}
kernel void q8_matvec_multi(device const block_q8_0 *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q8_0&blk=W[row*bpr+b];float sc=float(blk.d),acc=0.f;
            for(uint i=0;i<32;i+=4)acc+=dot(float4(float(blk.qs[i]),float(blk.qs[i+1]),float(blk.qs[i+2]),float(blk.qs[i+3])),float4(xv[i],xv[i+1],xv[i+2],xv[i+3]));
            s[r]+=sc*acc;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=simd_sum(s[r]);if(lane==0)y[br+r]=t;}}
}
kernel void q8_matvec_argmax(device const block_q8_0 *W [[buffer(0)]], device const float *x [[buffer(1)]], device argmax_partial *P [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q8_0&blk=W[row*bpr+b];float sc=float(blk.d),acc=0.f;
            for(uint i=0;i<32;i+=4)acc+=dot(float4(float(blk.qs[i]),float(blk.qs[i+1]),float(blk.qs[i+2]),float(blk.qs[i+3])),float4(xv[i],xv[i+1],xv[i+2],xv[i+3]));
            s[r]+=sc*acc;}}
    float bv=-MAXFLOAT;uint bi=0;
    for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;float t=simd_sum(s[r]);if(t>bv){bv=t;bi=row;}}
    if(lane==0){P[tgid].val=bv;P[tgid].idx=bi;}
}

kernel void q4_0_extract_row(device const block_q4_0 *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid>=dim) return;
    block_q4_0 b=B[row*(dim/32)+tid/32];
    uint ib=tid%32; uchar q=b.qs[ib/2]; float qv=float((ib&1u)?(q>>4u):(q&0xFu));
    o[tid]=float(b.d)*(qv-8.f);
}
kernel void q4_0_matvec_multi(device const block_q4_0 *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q4_0&blk=W[row*bpr+b];float sc=float(blk.d),acc=0.f;
            for(uint i=0;i<16;i++){uchar q=blk.qs[i];acc+=(float(q&0xFu)-8.f)*xv[i*2]+(float(q>>4u)-8.f)*xv[i*2+1];}
            s[r]+=sc*acc;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=simd_sum(s[r]);if(lane==0)y[br+r]=t;}}
}
kernel void q4_0_matvec_argmax(device const block_q4_0 *W [[buffer(0)]], device const float *x [[buffer(1)]], device argmax_partial *P [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q4_0&blk=W[row*bpr+b];float sc=float(blk.d),acc=0.f;
            for(uint i=0;i<16;i++){uchar q=blk.qs[i];acc+=(float(q&0xFu)-8.f)*xv[i*2]+(float(q>>4u)-8.f)*xv[i*2+1];}
            s[r]+=sc*acc;}}
    float bv=-MAXFLOAT;uint bi=0;
    for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;float t=simd_sum(s[r]);if(t>bv){bv=t;bi=row;}}
    if(lane==0){P[tgid].val=bv;P[tgid].idx=bi;}
}

kernel void q4_K_extract_row(device const block_q4_K *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if(tid>=dim) return;
    uint bpr=dim/256u, sb=(tid%256u)/32u, pos=(tid%256u)%32u;
    device const block_q4_K &b=B[row*bpr+tid/256u];
    uint2 sm=qK_scale_min(b.scales,sb);
    uint qsb=(sb/2u)*32u+pos, qss=(sb&1u)*4u;
    float q4=float((b.qs[qsb]>>qss)&0xFu);
    o[tid]=float(b.d)*float(sm.x)*q4-float(b.dmin)*float(sm.y);
}

kernel void q4_K_matvec_multi(
    device const block_q4_K *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    constexpr uint ROWS_PER_TG = 4;
    const uint base_row = tgid.x * ROWS_PER_TG;
    const uint n_blocks = cols / 256u;
    
    threadgroup float shared_x[256];
    float row_sums[ROWS_PER_TG];
    for (uint i = 0; i < ROWS_PER_TG; i++) row_sums[i] = 0.0f;
    
    for (uint b = 0; b < n_blocks; b++) {
        for (uint i = tiitg; i < 256u; i += 64u) {
            shared_x[i] = x[b * 256u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint r = 0; r < ROWS_PER_TG; r++) {
            uint row = base_row + r;
            if (row >= rows) continue;
            
            device const block_q4_K &blk = W[row * n_blocks + b];
            const uint sub = tiitg / 8u;
            const uint lane_in_sub = tiitg % 8u;
            if (sub < 8u) {
                uint2 sm = qK_scale_min(blk.scales, sub);
                float d = float(blk.d) * float(sm.x);
                float dm = float(blk.dmin) * float(sm.y);
                
                const uint qbase = (sub / 2u) * 32u;
                const uint qshift = (sub & 1u) * 4u;
                
                float acc = 0.0f, xsum = 0.0f;
                for (uint k = 0; k < 4u; k++) {
                    uint idx = lane_in_sub * 4u + k;
                    float qval = float((blk.qs[qbase + idx] >> qshift) & 0xFu);
                    float xval = shared_x[sub * 32u + idx];
                    acc += qval * xval;
                    xsum += xval;
                }
                row_sums[r] += d * acc - dm * xsum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    for (uint r = 0; r < ROWS_PER_TG; r++) {
        uint row = base_row + r;
        if (row < rows) {
            float val = simd_sum(row_sums[r]);
            if (tiisg == 0) y[row] = val;
        }
    }
}

kernel void q5_K_extract_row(device const block_q5_K *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if(tid>=dim) return;
    uint bpr=dim/256u, sb=(tid%256u)/32u, pos=(tid%256u)%32u;
    device const block_q5_K &b=B[row*bpr+tid/256u];
    uint2 sm=qK_scale_min(b.scales,sb);
    uint qsb=(sb/2u)*32u+pos, qss=(sb&1u)*4u;
    float q4=float((b.qs[qsb]>>qss)&0xFu);
    float qh=float((b.qh[pos]>>sb)&1u)*16.f;
    o[tid]=float(b.d)*float(sm.x)*(q4+qh)-float(b.dmin)*float(sm.y);
}
kernel void q5_K_matvec_multi(device const block_q5_K *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS_Q5K, bpr=cols/256u, ns=bpr*8u; float s[NROWS_Q5K]; for(uint r=0;r<NROWS_Q5K;r++) s[r]=0.f;
    for(uint sb=lane;sb<ns;sb+=sw){uint bb=sb/8u,sub=sb%8u,xb=bb*256u+sub*32u;float xv[32];for(uint i=0;i<32u;i++)xv[i]=x[xb+i];
        for(uint r=0;r<NROWS_Q5K;r++){uint row=br+r;if(row>=rows)continue;device const block_q5_K&blk=W[row*bpr+bb];
            uint2 sm=qK_scale_min(blk.scales,sub);float ds=float(blk.d)*float(sm.x),dm=float(blk.dmin)*float(sm.y);
            uint qb=(sub/2u)*32u,qs=(sub&1u)*4u;float acc=0.f,sx=0.f;
            for(uint l=0u;l<32u;l++){float q=float((blk.qs[qb+l]>>qs)&0xFu)+float((blk.qh[l]>>sub)&1u)*16.f;acc+=q*xv[l];sx+=xv[l];}
            s[r]+=ds*acc-dm*sx;}}
    for(uint r=0;r<NROWS_Q5K;r++){if(br+r<rows){float t=simd_sum(s[r]);if(lane==0)y[br+r]=t;}}
}
kernel void q5_K_matvec_argmax(device const block_q5_K *W [[buffer(0)]], device const float *x [[buffer(1)]], device argmax_partial *P [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS_Q5K, bpr=cols/256u, ns=bpr*8u; float s[NROWS_Q5K]; for(uint r=0;r<NROWS_Q5K;r++) s[r]=0.f;
    for(uint sb=lane;sb<ns;sb+=sw){uint bb=sb/8u,sub=sb%8u,xb=bb*256u+sub*32u;float xv[32];for(uint i=0;i<32u;i++)xv[i]=x[xb+i];
        for(uint r=0;r<NROWS_Q5K;r++){uint row=br+r;if(row>=rows)continue;device const block_q5_K&blk=W[row*bpr+bb];
            uint2 sm=qK_scale_min(blk.scales,sub);float ds=float(blk.d)*float(sm.x),dm=float(blk.dmin)*float(sm.y);
            uint qb=(sub/2u)*32u,qs=(sub&1u)*4u;float acc=0.f,sx=0.f;
            for(uint l=0u;l<32u;l++){float q=float((blk.qs[qb+l]>>qs)&0xFu)+float((blk.qh[l]>>sub)&1u)*16.f;acc+=q*xv[l];sx+=xv[l];}
            s[r]+=ds*acc-dm*sx;}}
    float bv=-MAXFLOAT;uint bi=0;
    for(uint r=0;r<NROWS_Q5K;r++){uint row=br+r;if(row>=rows)continue;float t=simd_sum(s[r]);if(t>bv){bv=t;bi=row;}}
    if(lane==0){P[tgid].val=bv;P[tgid].idx=bi;}
}

kernel void q6_K_extract_row(device const block_q6_K *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if(tid>=dim) return;
    uint bpr=dim/256u; device const block_q6_K &b=B[row*bpr+tid/256u];
    uint el=tid%256u, h=el/128u, q=(el%128u)/32u, l=el%32u;
    uint qi=(q<2u)?(h*64u+q*32u+l):(h*64u+(q-2u)*32u+l), qs=(q<2u)?0u:4u;
    uint lo=(uint(b.ql[qi])>>qs)&0xFu, hi=(uint(b.qh[h*32u+l])>>(q*2u))&3u;
    o[tid]=float(b.d)*float(b.scales[h*8u+q*2u+l/16u])*float(int(lo|(hi<<4u))-32);
}

kernel void q6_K_matvec_multi(
    device const block_q6_K *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    constexpr uint ROWS_PER_TG = 4;
    const uint base_row = tgid.x * ROWS_PER_TG;
    const uint n_blocks = cols / 256u;
    
    threadgroup float shared_x[256];
    float row_sums[ROWS_PER_TG];
    for (uint i = 0; i < ROWS_PER_TG; i++) row_sums[i] = 0.0f;
    
    for (uint b = 0; b < n_blocks; b++) {
        for (uint i = tiitg; i < 256u; i += 64u) shared_x[i] = x[b * 256u + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < ROWS_PER_TG; r++) {
            uint row = base_row + r;
            if (row >= rows) continue;
            device const block_q6_K &blk = W[row * n_blocks + b];
            float d = float(blk.d), acc = 0.0f;
            for (uint i = tiitg; i < 256u; i += 64u) {
                uint h = i / 128u, q = (i % 128u) / 32u, l = i % 32u;
                uint qi = (q < 2u) ? (h * 64u + q * 32u + l) : (h * 64u + (q - 2u) * 32u + l);
                uint qs = (q < 2u) ? 0u : 4u;
                uint lo = (uint(blk.ql[qi]) >> qs) & 0xFu;
                uint hi = (uint(blk.qh[h * 32u + l]) >> (q * 2u)) & 3u;
                float scale = float(blk.scales[h * 8u + q * 2u + l / 16u]);
                acc += scale * float(int(lo | (hi << 4u)) - 32) * shared_x[i];
            }
            row_sums[r] += d * acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < ROWS_PER_TG; r++) {
        uint row = base_row + r;
        if (row < rows) { float val = simd_sum(row_sums[r]); if (tiisg == 0) y[row] = val; }
    }
}

kernel void q3_K_extract_row(device const block_q3_K *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if(tid>=dim) return;
    uint bpr=dim/256u; device const block_q3_K &b=B[row*bpr+tid/256u];
    uint i=tid%256u, ni=i>>7, j=(i>>5)&3u, l=i&31u;
    int lo=(int)((b.qs[ni*32u+l]>>(2u*j))&3u), hb=(int)((b.hmask[l]>>(j+4u*ni))&1u);
    uint sb=8u*ni+2u*j+(l>=16u?1u:0u);
    o[tid]=float(b.d)*(float(q3K_sc(b.scales,sb))-32.f)*float(lo-(hb?0:4));
}
kernel void q3_K_matvec_multi(device const block_q3_K *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS_Q3K, bpr=cols/256u; float s[NROWS_Q3K]; for(uint r=0;r<NROWS_Q3K;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){uint xb=b*256u;
        for(uint r=0;r<NROWS_Q3K;r++){uint row=br+r;if(row>=rows)continue;device const block_q3_K&blk=W[row*bpr+b];float d=float(blk.d);
            float ds[16];for(uint ss=0u;ss<16u;ss++)ds[ss]=d*(float(q3K_sc(blk.scales,ss))-32.f);float acc=0.f;
            for(uint ni=0u;ni<2u;ni++){for(uint j=0u;j<4u;j++){uint sh=2u*j,hs=j+4u*ni,qb=ni*32u,xo=xb+ni*128u+j*32u,sb0=8u*ni+2u*j;
                for(uint k=0u;k<16u;k++){int lo=(int)((blk.qs[qb+k]>>sh)&3u),hb=(int)((blk.hmask[k]>>hs)&1u);acc+=ds[sb0]*float(lo-(hb?0:4))*x[xo+k];}
                for(uint k=0u;k<16u;k++){int lo=(int)((blk.qs[qb+16u+k]>>sh)&3u),hb=(int)((blk.hmask[16u+k]>>hs)&1u);acc+=ds[sb0+1u]*float(lo-(hb?0:4))*x[xo+16u+k];}
            }}s[r]+=acc;}}
    for(uint r=0;r<NROWS_Q3K;r++){if(br+r<rows){float t=simd_sum(s[r]);if(lane==0)y[br+r]=t;}}
}
kernel void q3_K_matvec_argmax(device const block_q3_K *W [[buffer(0)]], device const float *x [[buffer(1)]], device argmax_partial *P [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS_Q3K, bpr=cols/256u; float s[NROWS_Q3K]; for(uint r=0;r<NROWS_Q3K;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){uint xb=b*256u;
        for(uint r=0;r<NROWS_Q3K;r++){uint row=br+r;if(row>=rows)continue;device const block_q3_K&blk=W[row*bpr+b];float d=float(blk.d);
            float ds[16];for(uint ss=0u;ss<16u;ss++)ds[ss]=d*(float(q3K_sc(blk.scales,ss))-32.f);float acc=0.f;
            for(uint ni=0u;ni<2u;ni++){for(uint j=0u;j<4u;j++){uint sh=2u*j,hs=j+4u*ni,qb=ni*32u,xo=xb+ni*128u+j*32u,sb0=8u*ni+2u*j;
                for(uint k=0u;k<16u;k++){int lo=(int)((blk.qs[qb+k]>>sh)&3u),hb=(int)((blk.hmask[k]>>hs)&1u);acc+=ds[sb0]*float(lo-(hb?0:4))*x[xo+k];}
                for(uint k=0u;k<16u;k++){int lo=(int)((blk.qs[qb+16u+k]>>sh)&3u),hb=(int)((blk.hmask[16u+k]>>hs)&1u);acc+=ds[sb0+1u]*float(lo-(hb?0:4))*x[xo+16u+k];}
            }}s[r]+=acc;}}
    float bv=-MAXFLOAT;uint bi=0;
    for(uint r=0;r<NROWS_Q3K;r++){uint row=br+r;if(row>=rows)continue;float t=simd_sum(s[r]);if(t>bv){bv=t;bi=row;}}
    if(lane==0){P[tgid].val=bv;P[tgid].idx=bi;}
}

kernel void rms_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *o [[buffer(2)]], constant uint &n [[buffer(3)]], constant float &eps [[buffer(4)]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    float ss=0.f; for(uint i=lane;i<n;i+=sw) ss+=x[i]*x[i]; ss=simd_sum(ss);
    float inv=rsqrt(ss/float(n)+eps); for(uint i=lane;i<n;i+=sw) o[i]=x[i]*inv*w[i];
}

kernel void bias_add(device float *x [[buffer(0)]], device const float *b [[buffer(1)]], constant uint &n [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    if(tid<n) x[tid]+=b[tid];
}

kernel void rope_apply(device float *v [[buffer(0)]], constant uint &pos [[buffer(1)]], constant uint &hd [[buffer(2)]], constant uint &nh [[buffer(3)]], constant float &theta [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
    if(tid>=nh*(hd/2)) return;
    uint h=tid/(hd/2), p=tid%(hd/2), b1=h*hd+p, b2=h*hd+p+hd/2;
    float freq=pow(theta,-(float)(p*2)/(float)hd), a=(float)pos*freq;
    float cv=cos(a),sv=sin(a),v0=v[b1],v1=v[b2];
    v[b1]=v0*cv-v1*sv; v[b2]=v1*cv+v0*sv;
}

kernel void kv_cache_store(device const float *k [[buffer(0)]], device const float *v [[buffer(1)]], device half *kc [[buffer(2)]], device half *vc [[buffer(3)]], constant uint &pos [[buffer(4)]], constant uint &kvd [[buffer(5)]], uint tid [[thread_position_in_grid]]) {
    if(tid<kvd){kc[pos*kvd+tid]=half(k[tid]);vc[pos*kvd+tid]=half(v[tid]);}
}

kernel void attention_gqa(device const float *Q [[buffer(0)]], device const half *K [[buffer(1)]], device const half *V [[buffer(2)]], device float *out [[buffer(3)]], constant uint &sl [[buffer(4)]], constant uint &nqh [[buffer(5)]], constant uint &nkh [[buffer(6)]], constant uint &hd [[buffer(7)]], constant float &sc [[buffer(8)]], uint hid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    if(hid>=nqh) return;
    uint kh=hid/(nqh/nkh), kvd=nkh*hd; device const float *q=Q+hid*hd;
    threadgroup float scores[MAX_CTX];
    for(uint p=0;p<sl;p++){device const half *k=K+p*kvd+kh*hd;float part=0.f;for(uint d=lane;d<hd;d+=sw)part+=q[d]*float(k[d]);float dd=simd_sum(part);if(lane==0)scores[p]=dd*sc;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=-MAXFLOAT; for(uint p=lane;p<sl;p+=sw)mx=max(mx,scores[p]); mx=simd_max(mx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sm=0.f; for(uint p=lane;p<sl;p+=sw){float e=exp(scores[p]-mx);scores[p]=e;sm+=e;} sm=simd_sum(sm);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=1.f/sm; for(uint p=lane;p<sl;p+=sw)scores[p]*=inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=lane;d<hd;d+=sw){float s=0.f;for(uint p=0;p<sl;p++)s+=scores[p]*float(V[p*kvd+kh*hd+d]);out[hid*hd+d]=s;}
}

kernel void silu_mul(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]], device float *o [[buffer(2)]], constant uint &n [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if(tid<n){float gv=g[tid];o[tid]=(gv/(1.f+exp(-gv)))*u[tid];}
}

kernel void residual_add(device float *x [[buffer(0)]], device const float *y [[buffer(1)]], constant uint &n [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    if(tid<n) x[tid]+=y[tid];
}

kernel void argmax_reduce(device const argmax_partial *P [[buffer(0)]], device uint *result [[buffer(1)]], constant uint &n [[buffer(2)]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    float bv=-MAXFLOAT;uint bi=0;
    for(uint i=lane;i<n;i+=sw){float v=P[i].val;if(v>bv){bv=v;bi=P[i].idx;}}
    for(uint s=sw/2;s>0;s>>=1){float ov=simd_shuffle_down(bv,s);uint oi=simd_shuffle_down(bi,s);if(ov>bv){bv=ov;bi=oi;}}
    if(lane==0) result[0]=bi;
}

kernel void embed_token_from_buffer(
    device const uint32_t *token_buffer [[buffer(0)]],
    constant uint &buffer_index [[buffer(1)]],
    device const void *embed_table [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant uint &embed_type [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= dim) return;
    uint32_t token_id = token_buffer[buffer_index];
    if (embed_type == 12) {
        device const block_q4_K *B = (device const block_q4_K *)embed_table;
        uint bpr = dim / 256u, sb = (tid % 256u) / 32u, pos = (tid % 256u) % 32u;
        device const block_q4_K &b = B[token_id * bpr + tid / 256u];
        uint2 sm = qK_scale_min(b.scales, sb);
        uint qsb = (sb / 2u) * 32u + pos, qss = (sb & 1u) * 4u;
        float q4 = float((b.qs[qsb] >> qss) & 0xFu);
        output[tid] = float(b.d) * float(sm.x) * q4 - float(b.dmin) * float(sm.y);
    } else if (embed_type == 14) {
        device const block_q6_K *B = (device const block_q6_K *)embed_table;
        uint bpr = dim / 256u;
        device const block_q6_K &b = B[token_id * bpr + tid / 256u];
        uint el = tid % 256u, h = el / 128u, q = (el % 128u) / 32u, l = el % 32u;
        uint qi = (q < 2u) ? (h * 64u + q * 32u + l) : (h * 64u + (q - 2u) * 32u + l), qs = (q < 2u) ? 0u : 4u;
        uint lo = (uint(b.ql[qi]) >> qs) & 0xFu, hi = (uint(b.qh[h * 32u + l]) >> (q * 2u)) & 3u;
        output[tid] = float(b.d) * float(b.scales[h * 8u + q * 2u + l / 16u]) * float(int(lo | (hi << 4u)) - 32);
    } else {
        device const block_q8_0 *B = (device const block_q8_0 *)embed_table;
        block_q8_0 b = B[token_id * (dim / 32) + tid / 32];
        output[tid] = float(b.d) * float(b.qs[tid % 32]);
    }
}

kernel void argmax_write_token(
    device const float *logits [[buffer(0)]],
    device uint32_t *output_token [[buffer(1)]],
    constant uint &vocab_size [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sw [[threads_per_simdgroup]])
{
    float best_val = -MAXFLOAT;
    uint best_idx = 0;
    for (uint i = lane; i < vocab_size; i += sw) {
        float v = logits[i];
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    for (uint s = sw / 2; s > 0; s >>= 1) {
        float other_val = simd_shuffle_down(best_val, s);
        uint other_idx = simd_shuffle_down(best_idx, s);
        if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
    }
    if (lane == 0) *output_token = best_idx;
}


// ============================================================================
// Portable threadgroup reductions. Work at any threadExecutionWidth
// (AMD 64, Apple 32, Intel 16). Use instead of simd_sum / simd_max /
// simd_shuffle_down in any kernel that must run on non-Apple GPUs.
// ============================================================================

static inline float tg_sum(threadgroup float *scratch,
                           float val, uint lane, uint tg_size) {
    scratch[lane] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lane < s) scratch[lane] += scratch[lane + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline float tg_max(threadgroup float *scratch,
                           float val, uint lane, uint tg_size) {
    scratch[lane] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lane < s) scratch[lane] = max(scratch[lane], scratch[lane + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline void tg_argmax(threadgroup float *sv, threadgroup uint *si,
                             float val, uint idx,
                             thread float &out_val, thread uint &out_idx,
                             uint lane, uint tg_size) {
    sv[lane] = val;
    si[lane] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lane < s) {
            if (sv[lane + s] > sv[lane]) {
                sv[lane] = sv[lane + s];
                si[lane] = si[lane + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out_val = sv[0];
    out_idx = si[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ============================================================================
// AMD-portable matvec / argmax variants
// ============================================================================

kernel void q8_matvec_multi_amd(
    device const block_q8_0 *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float scratch[64];
    const uint br=tgid*NROWS, bpr=cols/32;
    float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){
        float xv[32]; for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){
            uint row=br+r; if(row>=rows)continue;
            device const block_q8_0&blk=W[row*bpr+b];
            float sc=float(blk.d),acc=0.f;
            for(uint i=0;i<32;i+=4)
                acc+=dot(float4(float(blk.qs[i]),float(blk.qs[i+1]),
                                float(blk.qs[i+2]),float(blk.qs[i+3])),
                         float4(xv[i],xv[i+1],xv[i+2],xv[i+3]));
            s[r]+=sc*acc;
        }
    }
    for(uint r=0;r<NROWS;r++){
        if(br+r<rows){
            float t=tg_sum(scratch, s[r], tiitg, sw);
            if(tiitg==0) y[br+r]=t;
        }
    }
}

kernel void q4_0_matvec_multi_amd(
    device const block_q4_0 *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float scratch[64];
    const uint br=tgid*NROWS, bpr=cols/32;
    float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){
        float xv[32]; for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){
            uint row=br+r; if(row>=rows)continue;
            device const block_q4_0&blk=W[row*bpr+b];
            float sc=float(blk.d),acc=0.f;
            for(uint i=0;i<16;i++){
                uchar q=blk.qs[i];
                acc+=(float(q&0xFu)-8.f)*xv[i*2]+(float(q>>4u)-8.f)*xv[i*2+1];
            }
            s[r]+=sc*acc;
        }
    }
    for(uint r=0;r<NROWS;r++){
        if(br+r<rows){
            float t=tg_sum(scratch, s[r], tiitg, sw);
            if(tiitg==0) y[br+r]=t;
        }
    }
}

kernel void q4_K_matvec_multi_amd(
    device const block_q4_K *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[64];
    constexpr uint ROWS_PER_TG = 4;
    const uint base_row = tgid.x * ROWS_PER_TG;
    const uint n_blocks = cols / 256u;
    const uint TG_SIZE = 64u;

    threadgroup float shared_x[256];
    float row_sums[ROWS_PER_TG] = {0,0,0,0};

    for (uint b = 0; b < n_blocks; b++) {
        for (uint i = tiitg; i < 256u; i += TG_SIZE) shared_x[i] = x[b*256u+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < ROWS_PER_TG; r++) {
            uint row = base_row + r;
            if (row >= rows) continue;
            device const block_q4_K &blk = W[row * n_blocks + b];
            uint sub = tiitg / 8u;
            uint lane_in_sub = tiitg % 8u;
            if (sub < 8u) {
                uint2 sm = qK_scale_min(blk.scales, sub);
                float d = float(blk.d) * float(sm.x);
                float dm = float(blk.dmin) * float(sm.y);
                uint qbase = (sub / 2u) * 32u;
                uint qshift = (sub & 1u) * 4u;
                float acc = 0.f, xsum = 0.f;
                for (uint k = 0; k < 4u; k++) {
                    uint idx = lane_in_sub * 4u + k;
                    float qval = float((blk.qs[qbase + idx] >> qshift) & 0xFu);
                    float xval = shared_x[sub * 32u + idx];
                    acc += qval * xval;
                    xsum += xval;
                }
                row_sums[r] += d * acc - dm * xsum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < ROWS_PER_TG; r++) {
        uint row = base_row + r;
        if (row < rows) {
            float val = tg_sum(scratch, row_sums[r], tiitg, TG_SIZE);
            if (tiitg == 0) y[row] = val;
        }
    }
}

kernel void q5_K_matvec_multi_amd(
    device const block_q5_K *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float scr[64];
    const uint br=tgid*NROWS_Q5K, bpr=cols/256u, ns=bpr*8u;
    float s[NROWS_Q5K]; for(uint r=0;r<NROWS_Q5K;r++) s[r]=0.f;
    for(uint sb=lane;sb<ns;sb+=sw){
        uint bb=sb/8u,sub=sb%8u,xb=bb*256u+sub*32u;
        float xv[32]; for(uint i=0;i<32u;i++)xv[i]=x[xb+i];
        for(uint r=0;r<NROWS_Q5K;r++){
            uint row=br+r; if(row>=rows)continue;
            device const block_q5_K&blk=W[row*bpr+bb];
            uint2 sm=qK_scale_min(blk.scales,sub);
            float ds=float(blk.d)*float(sm.x),dm=float(blk.dmin)*float(sm.y);
            uint qb=(sub/2u)*32u,qs=(sub&1u)*4u;
            float acc=0.f,sx=0.f;
            for(uint l=0u;l<32u;l++){
                float q=float((blk.qs[qb+l]>>qs)&0xFu)
                       +float((blk.qh[l]>>sub)&1u)*16.f;
                acc+=q*xv[l];
                sx+=xv[l];
            }
            s[r]+=ds*acc-dm*sx;
        }
    }
    for(uint r=0;r<NROWS_Q5K;r++){
        if(br+r<rows){
            float t=tg_sum(scr, s[r], tiitg, sw);
            if(tiitg==0) y[br+r]=t;
        }
    }
}

kernel void q6_K_matvec_multi_amd(
    device const block_q6_K *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    threadgroup float scr[64];
    constexpr uint ROWS_PER_TG = 4;
    constexpr uint TG_SIZE = 64u;
    const uint base_row = tgid.x * ROWS_PER_TG;
    const uint n_blocks = cols / 256u;

    threadgroup float shared_x[256];
    float row_sums[ROWS_PER_TG] = {0,0,0,0};

    for (uint b = 0; b < n_blocks; b++) {
        for (uint i = tiitg; i < 256u; i += TG_SIZE) shared_x[i] = x[b*256u+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < ROWS_PER_TG; r++) {
            uint row = base_row + r;
            if (row >= rows) continue;
            device const block_q6_K &blk = W[row * n_blocks + b];
            float d = float(blk.d), acc = 0.0f;
            for (uint i = tiitg; i < 256u; i += TG_SIZE) {
                uint h = i / 128u, q = (i % 128u) / 32u, l = i % 32u;
                uint qi = (q < 2u) ? (h*64u + q*32u + l) : (h*64u + (q-2u)*32u + l);
                uint qs = (q < 2u) ? 0u : 4u;
                uint lo = (uint(blk.ql[qi]) >> qs) & 0xFu;
                uint hi = (uint(blk.qh[h*32u + l]) >> (q*2u)) & 3u;
                float scale = float(blk.scales[h*8u + q*2u + l/16u]);
                acc += scale * float(int(lo | (hi << 4u)) - 32) * shared_x[i];
            }
            row_sums[r] += d * acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < ROWS_PER_TG; r++) {
        uint row = base_row + r;
        if (row < rows) {
            float val = tg_sum(scr, row_sums[r], tiitg, TG_SIZE);
            if (tiitg == 0) y[row] = val;
        }
    }
}

kernel void q3_K_matvec_multi_amd(
    device const block_q3_K *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float scr[64];
    const uint br=tgid*NROWS_Q3K, bpr=cols/256u;
    float s[NROWS_Q3K]; for(uint r=0;r<NROWS_Q3K;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){
        uint xb=b*256u;
        for(uint r=0;r<NROWS_Q3K;r++){
            uint row=br+r; if(row>=rows)continue;
            device const block_q3_K&blk=W[row*bpr+b];
            float d=float(blk.d);
            float ds[16];
            for(uint ss=0u;ss<16u;ss++) ds[ss]=d*(float(q3K_sc(blk.scales,ss))-32.f);
            float acc=0.f;
            for(uint ni=0u;ni<2u;ni++){
                for(uint j=0u;j<4u;j++){
                    uint sh=2u*j,hs=j+4u*ni,qb=ni*32u,xo=xb+ni*128u+j*32u,sb0=8u*ni+2u*j;
                    for(uint k=0u;k<16u;k++){
                        int lo=(int)((blk.qs[qb+k]>>sh)&3u);
                        int hb=(int)((blk.hmask[k]>>hs)&1u);
                        acc+=ds[sb0]*float(lo-(hb?0:4))*x[xo+k];
                    }
                    for(uint k=0u;k<16u;k++){
                        int lo=(int)((blk.qs[qb+16u+k]>>sh)&3u);
                        int hb=(int)((blk.hmask[16u+k]>>hs)&1u);
                        acc+=ds[sb0+1u]*float(lo-(hb?0:4))*x[xo+16u+k];
                    }
                }
            }
            s[r]+=acc;
        }
    }
    for(uint r=0;r<NROWS_Q3K;r++){
        if(br+r<rows){
            float t=tg_sum(scr, s[r], tiitg, sw);
            if(tiitg==0) y[br+r]=t;
        }
    }
}

kernel void q8_matvec_argmax_amd(
    device const block_q8_0 *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device argmax_partial *P [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float scr[64];
    const uint br=tgid*NROWS, bpr=cols/32;
    float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){
        float xv[32]; for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){
            uint row=br+r; if(row>=rows)continue;
            device const block_q8_0&blk=W[row*bpr+b];
            float sc=float(blk.d),acc=0.f;
            for(uint i=0;i<32;i+=4)
                acc+=dot(float4(float(blk.qs[i]),float(blk.qs[i+1]),
                                float(blk.qs[i+2]),float(blk.qs[i+3])),
                         float4(xv[i],xv[i+1],xv[i+2],xv[i+3]));
            s[r]+=sc*acc;
        }
    }
    float bv=-MAXFLOAT; uint bi=0;
    for(uint r=0;r<NROWS;r++){
        uint row=br+r; if(row>=rows)continue;
        float t=tg_sum(scr, s[r], tiitg, sw);
        if(t>bv){bv=t;bi=row;}
    }
    if(tiitg==0){P[tgid].val=bv;P[tgid].idx=bi;}
}

kernel void q4_0_matvec_argmax_amd(
    device const block_q4_0 *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device argmax_partial *P [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float scr[64];
    const uint br=tgid*NROWS, bpr=cols/32;
    float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){
        float xv[32]; for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){
            uint row=br+r; if(row>=rows)continue;
            device const block_q4_0&blk=W[row*bpr+b];
            float sc=float(blk.d),acc=0.f;
            for(uint i=0;i<16;i++){
                uchar q=blk.qs[i];
                acc+=(float(q&0xFu)-8.f)*xv[i*2]+(float(q>>4u)-8.f)*xv[i*2+1];
            }
            s[r]+=sc*acc;
        }
    }
    float bv=-MAXFLOAT; uint bi=0;
    for(uint r=0;r<NROWS;r++){
        uint row=br+r; if(row>=rows)continue;
        float t=tg_sum(scr, s[r], tiitg, sw);
        if(t>bv){bv=t;bi=row;}
    }
    if(tiitg==0){P[tgid].val=bv;P[tgid].idx=bi;}
}

kernel void q5_K_matvec_argmax_amd(
    device const block_q5_K *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device argmax_partial *P [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float scr[64];
    const uint br=tgid*NROWS_Q5K, bpr=cols/256u, ns=bpr*8u;
    float s[NROWS_Q5K]; for(uint r=0;r<NROWS_Q5K;r++) s[r]=0.f;
    for(uint sb=lane;sb<ns;sb+=sw){
        uint bb=sb/8u,sub=sb%8u,xb=bb*256u+sub*32u;
        float xv[32]; for(uint i=0;i<32u;i++)xv[i]=x[xb+i];
        for(uint r=0;r<NROWS_Q5K;r++){
            uint row=br+r; if(row>=rows)continue;
            device const block_q5_K&blk=W[row*bpr+bb];
            uint2 sm=qK_scale_min(blk.scales,sub);
            float ds=float(blk.d)*float(sm.x),dm=float(blk.dmin)*float(sm.y);
            uint qb=(sub/2u)*32u,qs=(sub&1u)*4u;
            float acc=0.f,sx=0.f;
            for(uint l=0u;l<32u;l++){
                float q=float((blk.qs[qb+l]>>qs)&0xFu)
                       +float((blk.qh[l]>>sub)&1u)*16.f;
                acc+=q*xv[l];
                sx+=xv[l];
            }
            s[r]+=ds*acc-dm*sx;
        }
    }
    float bv=-MAXFLOAT; uint bi=0;
    for(uint r=0;r<NROWS_Q5K;r++){
        uint row=br+r; if(row>=rows)continue;
        float t=tg_sum(scr, s[r], tiitg, sw);
        if(t>bv){bv=t;bi=row;}
    }
    if(tiitg==0){P[tgid].val=bv;P[tgid].idx=bi;}
}

kernel void q3_K_matvec_argmax_amd(
    device const block_q3_K *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device argmax_partial *P [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float scr[64];
    const uint br=tgid*NROWS_Q3K, bpr=cols/256u;
    float s[NROWS_Q3K]; for(uint r=0;r<NROWS_Q3K;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){
        uint xb=b*256u;
        for(uint r=0;r<NROWS_Q3K;r++){
            uint row=br+r; if(row>=rows)continue;
            device const block_q3_K&blk=W[row*bpr+b];
            float d=float(blk.d);
            float ds[16];
            for(uint ss=0u;ss<16u;ss++) ds[ss]=d*(float(q3K_sc(blk.scales,ss))-32.f);
            float acc=0.f;
            for(uint ni=0u;ni<2u;ni++){
                for(uint j=0u;j<4u;j++){
                    uint sh=2u*j,hs=j+4u*ni,qb=ni*32u,xo=xb+ni*128u+j*32u,sb0=8u*ni+2u*j;
                    for(uint k=0u;k<16u;k++){
                        int lo=(int)((blk.qs[qb+k]>>sh)&3u);
                        int hb=(int)((blk.hmask[k]>>hs)&1u);
                        acc+=ds[sb0]*float(lo-(hb?0:4))*x[xo+k];
                    }
                    for(uint k=0u;k<16u;k++){
                        int lo=(int)((blk.qs[qb+16u+k]>>sh)&3u);
                        int hb=(int)((blk.hmask[16u+k]>>hs)&1u);
                        acc+=ds[sb0+1u]*float(lo-(hb?0:4))*x[xo+16u+k];
                    }
                }
            }
            s[r]+=acc;
        }
    }
    float bv=-MAXFLOAT; uint bi=0;
    for(uint r=0;r<NROWS_Q3K;r++){
        uint row=br+r; if(row>=rows)continue;
        float t=tg_sum(scr, s[r], tiitg, sw);
        if(t>bv){bv=t;bi=row;}
    }
    if(tiitg==0){P[tgid].val=bv;P[tgid].idx=bi;}
}

kernel void rms_norm_amd(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *o [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float scr[64];
    float ss=0.f;
    for(uint i=lane;i<n;i+=sw) ss+=x[i]*x[i];
    ss = tg_sum(scr, ss, tiitg, sw);
    float inv = rsqrt(ss/float(n)+eps);
    for(uint i=lane;i<n;i+=sw) o[i]=x[i]*inv*w[i];
}

kernel void attention_gqa_amd(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &sl [[buffer(4)]],
    constant uint &nqh [[buffer(5)]],
    constant uint &nkh [[buffer(6)]],
    constant uint &hd [[buffer(7)]],
    constant float &sc [[buffer(8)]],
    uint hid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float scr[64];
    if(hid>=nqh) return;
    uint kh = (nqh == nkh) ? hid : (hid * nkh / nqh);
    uint kvd = nkh * hd;
    device const float *q = Q + hid * hd;
    threadgroup float scores[MAX_CTX];

    for(uint p=0;p<sl;p++){
        device const half *k = K + p*kvd + kh*hd;
        float part = 0.f;
        for(uint d=lane; d<hd; d+=sw) part += q[d] * float(k[d]);
        float dd = tg_sum(scr, part, tiitg, sw);
        if(tiitg==0) scores[p] = dd * sc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mx = -MAXFLOAT;
    for(uint p=lane; p<sl; p+=sw) mx = max(mx, scores[p]);
    mx = tg_max(scr, mx, tiitg, sw);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sm = 0.f;
    for(uint p=lane; p<sl; p+=sw){
        float e = exp(scores[p] - mx);
        scores[p] = e;
        sm += e;
    }
    sm = tg_sum(scr, sm, tiitg, sw);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv = 1.f / sm;
    for(uint p=lane; p<sl; p+=sw) scores[p] *= inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(uint d=lane; d<hd; d+=sw){
        float s = 0.f;
        for(uint p=0; p<sl; p++) s += scores[p] * float(V[p*kvd + kh*hd + d]);
        out[hid*hd + d] = s;
    }
}

kernel void argmax_reduce_amd(
    device const argmax_partial *P [[buffer(0)]],
    device uint *result [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float sv[64];
    threadgroup uint  si[64];
    float bv=-MAXFLOAT; uint bi=0;
    for(uint i=lane;i<n;i+=sw){
        float v=P[i].val;
        if(v>bv){bv=v;bi=P[i].idx;}
    }
    float ov; uint oi;
    tg_argmax(sv, si, bv, bi, ov, oi, tiitg, sw);
    if(tiitg==0) result[0]=oi;
}

kernel void argmax_write_token_amd(
    device const float *logits [[buffer(0)]],
    device uint32_t *output_token [[buffer(1)]],
    constant uint &vocab_size [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]])
{
    threadgroup float sv[64];
    threadgroup uint  si[64];
    float best_val=-MAXFLOAT; uint best_idx=0;
    for(uint i=lane;i<vocab_size;i+=sw){
        float v=logits[i];
        if(v>best_val){best_val=v;best_idx=i;}
    }
    float ov; uint oi;
    tg_argmax(sv, si, best_val, best_idx, ov, oi, tiitg, sw);
    if(tiitg==0) *output_token=oi;
}

struct block_q4_1 { half d; half m; uchar qs[16]; };

kernel void q4_1_extract_row(device const block_q4_1 *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid>=dim) return;
    block_q4_1 b=B[row*(dim/32)+tid/32];
    uint ib=tid%32; uchar q=b.qs[ib/2];
    float qv=float((ib&1u)?(q>>4u):(q&0xFu));
    o[tid]=float(b.d)*qv+float(b.m);
}

kernel void q4_1_matvec_multi(device const block_q4_1 *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q4_1&blk=W[row*bpr+b];
            float d=float(blk.d),m=float(blk.m),acc=0.f,sx=0.f;
            for(uint i=0;i<16;i++){uchar q=blk.qs[i];
                acc+=float(q&0xFu)*xv[i*2]+float(q>>4u)*xv[i*2+1];
                sx +=xv[i*2]+xv[i*2+1];}
            s[r]+=d*acc+m*sx;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=simd_sum(s[r]);if(lane==0)y[br+r]=t;}}
}

kernel void q4_1_matvec_multi_amd(device const block_q4_1 *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint tiitg [[thread_index_in_threadgroup]], uint sw [[threads_per_simdgroup]]) {
    threadgroup float scr[64];
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q4_1&blk=W[row*bpr+b];
            float d=float(blk.d),m=float(blk.m),acc=0.f,sx=0.f;
            for(uint i=0;i<16;i++){uchar q=blk.qs[i];
                acc+=float(q&0xFu)*xv[i*2]+float(q>>4u)*xv[i*2+1];
                sx +=xv[i*2]+xv[i*2+1];}
            s[r]+=d*acc+m*sx;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=tg_sum(scr,s[r],tiitg,sw);if(tiitg==0)y[br+r]=t;}}
}

struct block_q5_0 { half d; uchar qh[4]; uchar qs[16]; };

kernel void q5_0_extract_row(device const block_q5_0 *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid>=dim) return;
    block_q5_0 b=B[row*(dim/32)+tid/32];
    uint ib=tid%32; uchar q=b.qs[ib/2];
    float lo=float((ib&1u)?(q>>4u):(q&0xFu));
    float hi=float((b.qh[ib/8]>>(ib%8))&1u);
    o[tid]=float(b.d)*((lo+hi*16.f)-16.f);
}

kernel void q5_0_matvec_multi(device const block_q5_0 *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q5_0&blk=W[row*bpr+b];
            float d=float(blk.d),acc=0.f;
            uint qh=uint(blk.qh[0])|(uint(blk.qh[1])<<8)|(uint(blk.qh[2])<<16)|(uint(blk.qh[3])<<24);
            for(uint i=0;i<16;i++){uchar q=blk.qs[i];
                uint i0=i*2u,i1=i*2u+1u;
                float q0=float(q&0xFu)+float((qh>>i0)&1u)*16.f-16.f;
                float q1=float(q>>4u)+float((qh>>i1)&1u)*16.f-16.f;
                acc+=q0*xv[i0]+q1*xv[i1];}
            s[r]+=d*acc;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=simd_sum(s[r]);if(lane==0)y[br+r]=t;}}
}

kernel void q5_0_matvec_multi_amd(device const block_q5_0 *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint tiitg [[thread_index_in_threadgroup]], uint sw [[threads_per_simdgroup]]) {
    threadgroup float scr[64];
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q5_0&blk=W[row*bpr+b];
            float d=float(blk.d),acc=0.f;
            uint qh=uint(blk.qh[0])|(uint(blk.qh[1])<<8)|(uint(blk.qh[2])<<16)|(uint(blk.qh[3])<<24);
            for(uint i=0;i<16;i++){uchar q=blk.qs[i];
                uint i0=i*2u,i1=i*2u+1u;
                float q0=float(q&0xFu)+float((qh>>i0)&1u)*16.f-16.f;
                float q1=float(q>>4u)+float((qh>>i1)&1u)*16.f-16.f;
                acc+=q0*xv[i0]+q1*xv[i1];}
            s[r]+=d*acc;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=tg_sum(scr,s[r],tiitg,sw);if(tiitg==0)y[br+r]=t;}}
}

struct block_q5_1 { half d; half m; uchar qh[4]; uchar qs[16]; };

kernel void q5_1_extract_row(device const block_q5_1 *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid>=dim) return;
    block_q5_1 b=B[row*(dim/32)+tid/32];
    uint ib=tid%32; uchar q=b.qs[ib/2];
    float lo=float((ib&1u)?(q>>4u):(q&0xFu));
    float hi=float((b.qh[ib/8]>>(ib%8))&1u);
    o[tid]=float(b.d)*(lo+hi*16.f)+float(b.m);
}

kernel void q5_1_matvec_multi(device const block_q5_1 *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q5_1&blk=W[row*bpr+b];
            float d=float(blk.d),m=float(blk.m),acc=0.f,sx=0.f;
            uint qh=uint(blk.qh[0])|(uint(blk.qh[1])<<8)|(uint(blk.qh[2])<<16)|(uint(blk.qh[3])<<24);
            for(uint i=0;i<16;i++){uchar q=blk.qs[i];
                uint i0=i*2u,i1=i*2u+1u;
                float q0=float(q&0xFu)+float((qh>>i0)&1u)*16.f;
                float q1=float(q>>4u)+float((qh>>i1)&1u)*16.f;
                acc+=q0*xv[i0]+q1*xv[i1];
                sx +=xv[i0]+xv[i1];}
            s[r]+=d*acc+m*sx;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=simd_sum(s[r]);if(lane==0)y[br+r]=t;}}
}

kernel void q5_1_matvec_multi_amd(device const block_q5_1 *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint tiitg [[thread_index_in_threadgroup]], uint sw [[threads_per_simdgroup]]) {
    threadgroup float scr[64];
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q5_1&blk=W[row*bpr+b];
            float d=float(blk.d),m=float(blk.m),acc=0.f,sx=0.f;
            uint qh=uint(blk.qh[0])|(uint(blk.qh[1])<<8)|(uint(blk.qh[2])<<16)|(uint(blk.qh[3])<<24);
            for(uint i=0;i<16;i++){uchar q=blk.qs[i];
                uint i0=i*2u,i1=i*2u+1u;
                float q0=float(q&0xFu)+float((qh>>i0)&1u)*16.f;
                float q1=float(q>>4u)+float((qh>>i1)&1u)*16.f;
                acc+=q0*xv[i0]+q1*xv[i1];
                sx +=xv[i0]+xv[i1];}
            s[r]+=d*acc+m*sx;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=tg_sum(scr,s[r],tiitg,sw);if(tiitg==0)y[br+r]=t;}}
}

struct block_q6_0 { uchar ql[16]; uchar qh[8]; };

kernel void q6_0_extract_row(device const block_q6_0 *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid>=dim) return;
    block_q6_0 b=B[row*(dim/32)+tid/32];
    uint ib=tid%32;
    uchar qq=b.ql[ib/2];
    uint lo=(uint(qq)>>((ib%2u)*4u))&0xFu;
    uint hi=(uint(b.qh[ib/4])>>((ib%4u)*2u))&3u;
    o[tid]=float(int(lo|(hi<<4u))-32);
}

kernel void q6_0_matvec_multi(device const block_q6_0 *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q6_0&blk=W[row*bpr+b];
            float acc=0.f;
            for(uint i=0;i<16;i++){uchar qq=blk.ql[i];uchar hh=blk.qh[i/2];
                uint sh=(i%2u)*4u;
                uint lo0=uint(qq)&0xFu, lo1=(uint(qq)>>4u)&0xFu;
                uint hi0=(uint(hh)>>sh)&3u, hi1=(uint(hh)>>(sh+2u))&3u;
                acc+=float(int(lo0|(hi0<<4u))-32)*xv[i*2]
                    +float(int(lo1|(hi1<<4u))-32)*xv[i*2+1];}
            s[r]+=acc;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=simd_sum(s[r]);if(lane==0)y[br+r]=t;}}
}

kernel void q6_0_matvec_multi_amd(device const block_q6_0 *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint tiitg [[thread_index_in_threadgroup]], uint sw [[threads_per_simdgroup]]) {
    threadgroup float scr[64];
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i];
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q6_0&blk=W[row*bpr+b];
            float acc=0.f;
            for(uint i=0;i<16;i++){uchar qq=blk.ql[i];uchar hh=blk.qh[i/2];
                uint sh=(i%2u)*4u;
                uint lo0=uint(qq)&0xFu, lo1=(uint(qq)>>4u)&0xFu;
                uint hi0=(uint(hh)>>sh)&3u, hi1=(uint(hh)>>(sh+2u))&3u;
                acc+=float(int(lo0|(hi0<<4u))-32)*xv[i*2]
                    +float(int(lo1|(hi1<<4u))-32)*xv[i*2+1];}
            s[r]+=acc;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=tg_sum(scr,s[r],tiitg,sw);if(tiitg==0)y[br+r]=t;}}
}

struct block_q2_K { half d; half dmin; uchar scales[16]; uchar qs[64]; };

kernel void q2_K_extract_row(device const block_q2_K *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid>=dim) return;
    uint bpr=dim/256u; device const block_q2_K &b=B[row*bpr+tid/256u];
    uint e=tid%256u, sub=e/16u, pos=e%16u;
    uint grp=sub/8u, half_=(sub%8u)%2u, sh_in=(sub%8u)/2u;
    uint qidx=grp*32u+half_*16u+pos, sh=sh_in*2u;
    uint q=(uint(b.qs[qidx])>>sh)&3u;
    uint sc=uint(b.scales[sub]);
    float dl=float(b.d)*float(sc&0xFu);
    float ml=float(b.dmin)*float(sc>>4u);
    o[tid]=dl*float(q)-ml;
}

kernel void q2_K_matvec_multi(device const block_q2_K *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    const uint br=tgid*NROWS, bpr=cols/256u; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){uint xb=b*256u;
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q2_K&blk=W[row*bpr+b];
            float local=0.f;
            for(uint sub=0u;sub<16u;sub++){
                uint sc=uint(blk.scales[sub]);
                float dl=float(blk.d)*float(sc&0xFu);
                float ml=float(blk.dmin)*float(sc>>4u);
                uint grp=sub/8u, half_=(sub%8u)%2u, sh_in=(sub%8u)/2u;
                uint sh=sh_in*2u, xo=xb+sub*16u;
                for(uint k=0u;k<16u;k++){
                    uint q=(uint(blk.qs[grp*32u+half_*16u+k])>>sh)&3u;
                    local+=(dl*float(q)-ml)*x[xo+k];
                }
            }
            s[r]+=local;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=simd_sum(s[r]);if(lane==0)y[br+r]=t;}}
}

kernel void q2_K_matvec_multi_amd(device const block_q2_K *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint tiitg [[thread_index_in_threadgroup]], uint sw [[threads_per_simdgroup]]) {
    threadgroup float scr[64];
    const uint br=tgid*NROWS, bpr=cols/256u; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f;
    for(uint b=lane;b<bpr;b+=sw){uint xb=b*256u;
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const block_q2_K&blk=W[row*bpr+b];
            float local=0.f;
            for(uint sub=0u;sub<16u;sub++){
                uint sc=uint(blk.scales[sub]);
                float dl=float(blk.d)*float(sc&0xFu);
                float ml=float(blk.dmin)*float(sc>>4u);
                uint grp=sub/8u, half_=(sub%8u)%2u, sh_in=(sub%8u)/2u;
                uint sh=sh_in*2u, xo=xb+sub*16u;
                for(uint k=0u;k<16u;k++){
                    uint q=(uint(blk.qs[grp*32u+half_*16u+k])>>sh)&3u;
                    local+=(dl*float(q)-ml)*x[xo+k];
                }
            }
            s[r]+=local;}}
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=tg_sum(scr,s[r],tiitg,sw);if(tiitg==0)y[br+r]=t;}}
}

struct block_q3_0 { half d; uchar ql[8]; uchar qh[4]; uchar sc[2];  };
struct block_q3_1 { half d; uchar ql[8]; uchar qh[4]; uchar sc[6];  };
struct block_q3_2 { half d; uchar ql[8]; uchar qh[4]; uchar sc[10]; };
struct block_q3_3 { half d; uchar ql[8]; uchar qh[4]; uchar sc[14]; };

#define Q3_DECODE(BLK, I, OUT) do { \
    uchar _qq = (BLK).ql[(I)/4u]; \
    uint _lo = (uint(_qq) >> (((I)%4u)*2u)) & 0x3u; \
    uint _hi = (uint((BLK).qh[(I)/8u]) >> ((I)%8u)) & 0x1u; \
    OUT = float((int)(_lo | (_hi << 2u)) - 4); \
} while(0)

#define Q3_EXTRACT(NAME, TYPE) \
kernel void NAME(device const TYPE *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) { \
    if (tid>=dim) return; \
    TYPE b=B[row*(dim/32)+tid/32]; \
    float v; Q3_DECODE(b, tid%32u, v); \
    o[tid]=float(b.d)*v; \
}

Q3_EXTRACT(q3_0_extract_row, block_q3_0)
Q3_EXTRACT(q3_1_extract_row, block_q3_1)
Q3_EXTRACT(q3_2_extract_row, block_q3_2)
Q3_EXTRACT(q3_3_extract_row, block_q3_3)

#define Q3_MATVEC(NAME, TYPE) \
kernel void NAME(device const TYPE *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) { \
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f; \
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i]; \
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const TYPE&blk=W[row*bpr+b]; \
            float d=float(blk.d),acc=0.f; \
            for(uint i=0;i<8;i++){ \
                uchar qq=blk.ql[i]; uchar hh=blk.qh[i/2]; uint sh=(i%2u)*4u; \
                uint lo0=uint(qq)&0x3u,lo1=(uint(qq)>>2u)&0x3u,lo2=(uint(qq)>>4u)&0x3u,lo3=(uint(qq)>>6u)&0x3u; \
                uint hi0=(uint(hh)>>sh)&1u,hi1=(uint(hh)>>(sh+1u))&1u,hi2=(uint(hh)>>(sh+2u))&1u,hi3=(uint(hh)>>(sh+3u))&1u; \
                acc+=float((int)(lo0|(hi0<<2u))-4)*xv[i*4] \
                    +float((int)(lo1|(hi1<<2u))-4)*xv[i*4+1] \
                    +float((int)(lo2|(hi2<<2u))-4)*xv[i*4+2] \
                    +float((int)(lo3|(hi3<<2u))-4)*xv[i*4+3]; \
            } \
            s[r]+=d*acc;}} \
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=simd_sum(s[r]);if(lane==0)y[br+r]=t;}} \
}

Q3_MATVEC(q3_0_matvec_multi, block_q3_0)
Q3_MATVEC(q3_1_matvec_multi, block_q3_1)
Q3_MATVEC(q3_2_matvec_multi, block_q3_2)
Q3_MATVEC(q3_3_matvec_multi, block_q3_3)

#define Q3_MATVEC_AMD(NAME, TYPE) \
kernel void NAME(device const TYPE *W [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint tiitg [[thread_index_in_threadgroup]], uint sw [[threads_per_simdgroup]]) { \
    threadgroup float scr[64]; \
    const uint br=tgid*NROWS, bpr=cols/32; float s[NROWS]; for(uint r=0;r<NROWS;r++) s[r]=0.f; \
    for(uint b=lane;b<bpr;b+=sw){float xv[32];for(uint i=0;i<32;i++)xv[i]=x[b*32+i]; \
        for(uint r=0;r<NROWS;r++){uint row=br+r;if(row>=rows)continue;device const TYPE&blk=W[row*bpr+b]; \
            float d=float(blk.d),acc=0.f; \
            for(uint i=0;i<8;i++){ \
                uchar qq=blk.ql[i]; uchar hh=blk.qh[i/2]; uint sh=(i%2u)*4u; \
                uint lo0=uint(qq)&0x3u,lo1=(uint(qq)>>2u)&0x3u,lo2=(uint(qq)>>4u)&0x3u,lo3=(uint(qq)>>6u)&0x3u; \
                uint hi0=(uint(hh)>>sh)&1u,hi1=(uint(hh)>>(sh+1u))&1u,hi2=(uint(hh)>>(sh+2u))&1u,hi3=(uint(hh)>>(sh+3u))&1u; \
                acc+=float((int)(lo0|(hi0<<2u))-4)*xv[i*4] \
                    +float((int)(lo1|(hi1<<2u))-4)*xv[i*4+1] \
                    +float((int)(lo2|(hi2<<2u))-4)*xv[i*4+2] \
                    +float((int)(lo3|(hi3<<2u))-4)*xv[i*4+3]; \
            } \
            s[r]+=d*acc;}} \
    for(uint r=0;r<NROWS;r++){if(br+r<rows){float t=tg_sum(scr,s[r],tiitg,sw);if(tiitg==0)y[br+r]=t;}} \
}

Q3_MATVEC_AMD(q3_0_matvec_multi_amd, block_q3_0)
Q3_MATVEC_AMD(q3_1_matvec_multi_amd, block_q3_1)
Q3_MATVEC_AMD(q3_2_matvec_multi_amd, block_q3_2)
Q3_MATVEC_AMD(q3_3_matvec_multi_amd, block_q3_3)
