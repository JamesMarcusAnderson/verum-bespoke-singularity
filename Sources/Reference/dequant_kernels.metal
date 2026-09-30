// dequant_kernels.metal — REFERENCE ONLY (not build input)
//
// Recovered from the 2026-09-19 iMac revision of the Verum inference tree
// (newest known revision; supersedes the 2026-09-16 reference). 35 kernels:
// dequantizing matvec / row-extract for F32/F16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/
// Q2_K..Q6_K (plus _amd discrete-GPU variants), rms_norm, rope, kv_cache_store,
// GQA attention, silu_mul, residual_add, embed/argmax, Gumbel-max sampling.
//
// NOT numerically validated against ggml — ground-truth candidate for the
// runtime MSL generator to match. Parity tests required before any
// correctness claim.

#include <metal_stdlib>
using namespace metal;

constant uint MAX_CTX = 2048;

struct block_q8_0 { half d; char qs[32]; };
struct block_q4_0 { half d; uchar qs[16]; };
struct block_q4_1 { half d; half m; uchar qs[16]; };
struct block_q5_0 { half d; uchar qh[4]; uchar qs[16]; };
struct block_q5_1 { half d; half m; uchar qh[4]; uchar qs[16]; };
struct block_q2_K { uchar scales[16]; uchar qs[64]; half d; half dmin; };
struct block_q3_K { uchar hmask[32]; uchar qs[64]; uchar scales[12]; half d; };
struct block_q4_K { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct block_q5_K { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; };
struct block_q6_K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };

static inline uint2 qK_scale_min(device const uchar *sc, uint j) {
    uint s, mn;
    if (j < 4) {
        s = sc[j] & 0x3Fu;
        mn = sc[j + 4] & 0x3Fu;
    } else {
        s = (sc[j + 4] & 0x0Fu) | ((sc[j - 4] >> 6u) << 4u);
        mn = (sc[j + 4] >> 4u) | ((sc[j] >> 6u) << 4u);
    }
    return uint2(s, mn);
}

static inline int q3K_sc(device const uchar *sc, uint sb) {
    const uint m1 = 0x03030303u, m2 = 0x0f0f0f0fu;
    uint a0 = uint(sc[0]) | (uint(sc[1]) << 8) | (uint(sc[2]) << 16) | (uint(sc[3]) << 24);
    uint a1 = uint(sc[4]) | (uint(sc[5]) << 8) | (uint(sc[6]) << 16) | (uint(sc[7]) << 24);
    uint a2 = uint(sc[8]) | (uint(sc[9]) << 8) | (uint(sc[10]) << 16) | (uint(sc[11]) << 24);
    uint r[4];
    r[0] = (a0 & m2) | (((a2 >> 0) & m1) << 4);
    r[1] = (a1 & m2) | (((a2 >> 2) & m1) << 4);
    r[2] = ((a0 >> 4) & m2) | (((a2 >> 4) & m1) << 4);
    r[3] = ((a1 >> 4) & m2) | (((a2 >> 6) & m1) << 4);
    return (int)(char)((r[sb >> 2] >> (8u * (sb & 3u))) & 0xFFu);
}

kernel void q8_extract_row(device const block_q8_0 *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    block_q8_0 b = B[(size_t)row * (dim / 32u) + tid / 32u];
    o[tid] = float(b.d) * float(b.qs[tid % 32u]);
}

kernel void q4_0_extract_row(device const block_q4_0 *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    block_q4_0 b = B[(size_t)row * (dim / 32u) + tid / 32u];
    uint ib = tid % 32u;
    uchar q = b.qs[ib / 2u];
    float qv = float((ib & 1u) ? (q >> 4u) : (q & 0xFu));
    o[tid] = float(b.d) * (qv - 8.0f);
}

kernel void q4_1_extract_row(device const block_q4_1 *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    block_q4_1 b = B[(size_t)row * (dim / 32u) + tid / 32u];
    uint ib = tid % 32u;
    uchar q = b.qs[ib / 2u];
    float qv = float((ib & 1u) ? (q >> 4u) : (q & 0xFu));
    o[tid] = float(b.d) * qv + float(b.m);
}

kernel void q5_0_extract_row(device const block_q5_0 *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    block_q5_0 b = B[(size_t)row * (dim / 32u) + tid / 32u];
    uint qh32 = uint(b.qh[0]) | (uint(b.qh[1]) << 8) | (uint(b.qh[2]) << 16) | (uint(b.qh[3]) << 24);
    uint ib = tid % 32u;
    uint hb = (qh32 >> ib) & 1u;
    uchar q = b.qs[ib / 2u];
    float qv = float(((ib & 1u) ? (q >> 4u) : (q & 0xFu)) | (hb << 4u));
    o[tid] = float(b.d) * (qv - 16.0f);
}

kernel void q5_1_extract_row(device const block_q5_1 *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    block_q5_1 b = B[(size_t)row * (dim / 32u) + tid / 32u];
    uint qh32 = uint(b.qh[0]) | (uint(b.qh[1]) << 8) | (uint(b.qh[2]) << 16) | (uint(b.qh[3]) << 24);
    uint ib = tid % 32u;
    uint hb = (qh32 >> ib) & 1u;
    uchar q = b.qs[ib / 2u];
    float qv = float(((ib & 1u) ? (q >> 4u) : (q & 0xFu)) | (hb << 4u));
    o[tid] = float(b.d) * qv + float(b.m);
}

kernel void q2_K_extract_row(device const block_q2_K *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    uint bpr = dim / 256u;
    device const block_q2_K &b = B[(size_t)row * bpr + tid / 256u];
    uint el = tid % 256u;
    uint h = el / 128u, m = (el % 128u) / 32u, l = el % 32u;
    uint q = (uint(b.qs[h * 32u + l]) >> (2u * m)) & 3u;
    uint si = h * 8u + 2u * m + l / 16u;
    float sc = float(b.scales[si] & 0xFu);
    float mn = float(b.scales[si] >> 4u);
    o[tid] = float(b.d) * sc * float(q) - float(b.dmin) * mn;
}

kernel void q3_K_extract_row(device const block_q3_K *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    uint bpr = dim / 256u;
    device const block_q3_K &b = B[(size_t)row * bpr + tid / 256u];
    uint i = tid % 256u, ni = i >> 7, j = (i >> 5) & 3u, l = i & 31u;
    int lo = (int)((b.qs[ni * 32u + l] >> (2u * j)) & 3u);
    int hb = (int)((b.hmask[l] >> (j + 4u * ni)) & 1u);
    uint sb = 8u * ni + 2u * j + (l >= 16u ? 1u : 0u);
    o[tid] = float(b.d) * (float(q3K_sc(b.scales, sb)) - 32.0f) * float(lo - (hb ? 0 : 4));
}

kernel void q4_K_extract_row(device const block_q4_K *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    uint bpr = dim / 256u, sb = (tid % 256u) / 32u, pos = (tid % 256u) % 32u;
    device const block_q4_K &b = B[(size_t)row * bpr + tid / 256u];
    uint2 sm = qK_scale_min(b.scales, sb);
    uint qsb = (sb / 2u) * 32u + pos, qss = (sb & 1u) * 4u;
    float q4 = float((b.qs[qsb] >> qss) & 0xFu);
    o[tid] = float(b.d) * float(sm.x) * q4 - float(b.dmin) * float(sm.y);
}

kernel void q5_K_extract_row(device const block_q5_K *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    uint bpr = dim / 256u, sb = (tid % 256u) / 32u, pos = (tid % 256u) % 32u;
    device const block_q5_K &b = B[(size_t)row * bpr + tid / 256u];
    uint2 sm = qK_scale_min(b.scales, sb);
    uint qsb = (sb / 2u) * 32u + pos, qss = (sb & 1u) * 4u;
    float q4 = float((b.qs[qsb] >> qss) & 0xFu);
    float qh = float((b.qh[pos] >> sb) & 1u) * 16.0f;
    o[tid] = float(b.d) * float(sm.x) * (q4 + qh) - float(b.dmin) * float(sm.y);
}

kernel void q6_K_extract_row(device const block_q6_K *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    uint bpr = dim / 256u;
    device const block_q6_K &b = B[(size_t)row * bpr + tid / 256u];
    uint el = tid % 256u, h = el / 128u, q = (el % 128u) / 32u, l = el % 32u;
    uint qi = (q < 2u) ? (h * 64u + q * 32u + l) : (h * 64u + (q - 2u) * 32u + l);
    uint qs = (q < 2u) ? 0u : 4u;
    uint lo = (uint(b.ql[qi]) >> qs) & 0xFu;
    uint hi = (uint(b.qh[h * 32u + l]) >> (q * 2u)) & 3u;
    o[tid] = float(b.d) * float(b.scales[h * 8u + q * 2u + l / 16u]) * float(int(lo | (hi << 4u)) - 32);
}

kernel void f16_extract_row(device const half *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    o[tid] = float(B[(size_t)row * dim + tid]);
}

kernel void f32_extract_row(device const float *B [[buffer(0)]], device float *o [[buffer(1)]], constant uint &row [[buffer(2)]], constant uint &dim [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim) return;
    o[tid] = B[(size_t)row * dim + tid];
}
kernel void q8_matvec(device const block_q8_0 *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 8u;
    const uint br = tgid.x * RPT;
    const uint bpr = cols / 32u;
    device const float *x = X + (size_t)tgid.y * cols;
    float s[RPT];
    for (uint r = 0; r < RPT; r++) s[r] = 0.0f;
    for (uint bb = lane; bb < bpr; bb += sw) {
        float xv[32];
        for (uint i = 0; i < 32u; i++) xv[i] = x[bb * 32u + i];
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            device const block_q8_0 &blk = W[(size_t)row * bpr + bb];
            float acc = 0.0f;
            for (uint i = 0; i < 32u; i += 4)
                acc += dot(float4(float(blk.qs[i]), float(blk.qs[i + 1]), float(blk.qs[i + 2]), float(blk.qs[i + 3])),
                           float4(xv[i], xv[i + 1], xv[i + 2], xv[i + 3]));
            s[r] += float(blk.d) * acc;
        }
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(s[r]);
            if (lane == 0) y[row] = t;
        }
    }
}

kernel void q4_0_matvec(device const block_q4_0 *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 8u;
    const uint br = tgid.x * RPT;
    const uint bpr = cols / 32u;
    device const float *x = X + (size_t)tgid.y * cols;
    float s[RPT];
    for (uint r = 0; r < RPT; r++) s[r] = 0.0f;
    for (uint bb = lane; bb < bpr; bb += sw) {
        float xv[32];
        for (uint i = 0; i < 32u; i++) xv[i] = x[bb * 32u + i];
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            device const block_q4_0 &blk = W[(size_t)row * bpr + bb];
            float acc = 0.0f;
            for (uint i = 0; i < 16u; i++) {
                uchar q = blk.qs[i];
                acc += (float(q & 0xFu) - 8.0f) * xv[i * 2u] + (float(q >> 4u) - 8.0f) * xv[i * 2u + 1u];
            }
            s[r] += float(blk.d) * acc;
        }
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(s[r]);
            if (lane == 0) y[row] = t;
        }
    }
}

kernel void q4_1_matvec(device const block_q4_1 *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 8u;
    const uint br = tgid.x * RPT;
    const uint bpr = cols / 32u;
    device const float *x = X + (size_t)tgid.y * cols;
    float s[RPT];
    for (uint r = 0; r < RPT; r++) s[r] = 0.0f;
    for (uint bb = lane; bb < bpr; bb += sw) {
        float xv[32];
        float xs = 0.0f;
        for (uint i = 0; i < 32u; i++) { xv[i] = x[bb * 32u + i]; xs += xv[i]; }
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            device const block_q4_1 &blk = W[(size_t)row * bpr + bb];
            float acc = 0.0f;
            for (uint i = 0; i < 16u; i++) {
                uchar q = blk.qs[i];
                acc += float(q & 0xFu) * xv[i * 2u] + float(q >> 4u) * xv[i * 2u + 1u];
            }
            s[r] += float(blk.d) * acc + float(blk.m) * xs;
        }
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(s[r]);
            if (lane == 0) y[row] = t;
        }
    }
}

kernel void q5_0_matvec(device const block_q5_0 *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 8u;
    const uint br = tgid.x * RPT;
    const uint bpr = cols / 32u;
    device const float *x = X + (size_t)tgid.y * cols;
    float s[RPT];
    for (uint r = 0; r < RPT; r++) s[r] = 0.0f;
    for (uint bb = lane; bb < bpr; bb += sw) {
        float xv[32];
        for (uint i = 0; i < 32u; i++) xv[i] = x[bb * 32u + i];
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            device const block_q5_0 &blk = W[(size_t)row * bpr + bb];
            uint qh32 = uint(blk.qh[0]) | (uint(blk.qh[1]) << 8) | (uint(blk.qh[2]) << 16) | (uint(blk.qh[3]) << 24);
            float acc = 0.0f;
            for (uint l = 0; l < 16u; l++) {
                uint q0 = uint(blk.qs[l] & 0xFu) | (((qh32 >> l) & 1u) << 4u);
                uint q1 = uint(blk.qs[l] >> 4u) | (((qh32 >> (l + 16u)) & 1u) << 4u);
                acc += (float(q0) - 16.0f) * xv[l * 2u] + (float(q1) - 16.0f) * xv[l * 2u + 1u];
            }
            s[r] += float(blk.d) * acc;
        }
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(s[r]);
            if (lane == 0) y[row] = t;
        }
    }
}

kernel void q5_1_matvec(device const block_q5_1 *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 8u;
    const uint br = tgid.x * RPT;
    const uint bpr = cols / 32u;
    device const float *x = X + (size_t)tgid.y * cols;
    float s[RPT];
    for (uint r = 0; r < RPT; r++) s[r] = 0.0f;
    for (uint bb = lane; bb < bpr; bb += sw) {
        float xv[32];
        float xs = 0.0f;
        for (uint i = 0; i < 32u; i++) { xv[i] = x[bb * 32u + i]; xs += xv[i]; }
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            device const block_q5_1 &blk = W[(size_t)row * bpr + bb];
            uint qh32 = uint(blk.qh[0]) | (uint(blk.qh[1]) << 8) | (uint(blk.qh[2]) << 16) | (uint(blk.qh[3]) << 24);
            float acc = 0.0f;
            for (uint l = 0; l < 16u; l++) {
                uint q0 = uint(blk.qs[l] & 0xFu) | (((qh32 >> l) & 1u) << 4u);
                uint q1 = uint(blk.qs[l] >> 4u) | (((qh32 >> (l + 16u)) & 1u) << 4u);
                acc += float(q0) * xv[l * 2u] + float(q1) * xv[l * 2u + 1u];
            }
            s[r] += float(blk.d) * acc + float(blk.m) * xs;
        }
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(s[r]);
            if (lane == 0) y[row] = t;
        }
    }
}

kernel void f16_matvec(device const half *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 8u;
    const uint br = tgid.x * RPT;
    const uint bpr = cols / 32u;
    device const float *x = X + (size_t)tgid.y * cols;
    float s[RPT];
    for (uint r = 0; r < RPT; r++) s[r] = 0.0f;
    for (uint bb = lane; bb < bpr; bb += sw) {
        float xv[32];
        for (uint i = 0; i < 32u; i++) xv[i] = x[bb * 32u + i];
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            float acc = 0.0f;
            for (uint i = 0; i < 32u; i++)
                acc += float(W[(size_t)row * cols + bb * 32u + i]) * xv[i];
            s[r] += acc;
        }
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(s[r]);
            if (lane == 0) y[row] = t;
        }
    }
}

kernel void f32_matvec(device const float *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 8u;
    const uint br = tgid.x * RPT;
    const uint bpr = cols / 32u;
    device const float *x = X + (size_t)tgid.y * cols;
    float s[RPT];
    for (uint r = 0; r < RPT; r++) s[r] = 0.0f;
    for (uint bb = lane; bb < bpr; bb += sw) {
        float xv[32];
        for (uint i = 0; i < 32u; i++) xv[i] = x[bb * 32u + i];
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            float acc = 0.0f;
            for (uint i = 0; i < 32u; i++)
                acc += W[(size_t)row * cols + bb * 32u + i] * xv[i];
            s[r] += acc;
        }
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(s[r]);
            if (lane == 0) y[row] = t;
        }
    }
}
kernel void q2_K_matvec(device const block_q2_K *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 4u;
    const uint br = tgid.x * RPT;
    const uint bpr = cols / 256u;
    const uint ns = bpr * 8u;
    device const float *x = X + (size_t)tgid.y * cols;
    float s[RPT];
    for (uint r = 0; r < RPT; r++) s[r] = 0.0f;
    for (uint sb = lane; sb < ns; sb += sw) {
        uint bb = sb / 8u, si = sb % 8u;
        uint h = si / 4u, m = si % 4u;
        uint xb = bb * 256u + h * 128u + m * 32u;
        float xv[32];
        for (uint l = 0; l < 32u; l++) xv[l] = x[xb + l];
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            device const block_q2_K &blk = W[(size_t)row * bpr + bb];
            float acc = 0.0f;
            for (uint l = 0; l < 32u; l++) {
                uint q = (uint(blk.qs[h * 32u + l]) >> (2u * m)) & 3u;
                uint idx = h * 8u + 2u * m + l / 16u;
                float sc = float(blk.scales[idx] & 0xFu);
                float mn = float(blk.scales[idx] >> 4u);
                acc += (float(blk.d) * sc * float(q) - float(blk.dmin) * mn) * xv[l];
            }
            s[r] += acc;
        }
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(s[r]);
            if (lane == 0) y[row] = t;
        }
    }
}

kernel void q3_K_matvec(device const block_q3_K *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 4u;
    const uint br = tgid.x * RPT;
    const uint bpr = cols / 256u;
    device const float *x = X + (size_t)tgid.y * cols;
    float s[RPT];
    for (uint r = 0; r < RPT; r++) s[r] = 0.0f;
    for (uint bb = lane; bb < bpr; bb += sw) {
        uint xb = bb * 256u;
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            device const block_q3_K &blk = W[(size_t)row * bpr + bb];
            float d = float(blk.d);
            float ds[16];
            for (uint ss = 0; ss < 16u; ss++) ds[ss] = d * (float(q3K_sc(blk.scales, ss)) - 32.0f);
            float acc = 0.0f;
            for (uint ni = 0; ni < 2u; ni++) {
                for (uint j = 0; j < 4u; j++) {
                    uint sh = 2u * j, hs = j + 4u * ni, qb = ni * 32u;
                    uint xo = xb + ni * 128u + j * 32u, sb0 = 8u * ni + 2u * j;
                    for (uint k = 0; k < 16u; k++) {
                        int lo = (int)((blk.qs[qb + k] >> sh) & 3u);
                        int hb = (int)((blk.hmask[k] >> hs) & 1u);
                        acc += ds[sb0] * float(lo - (hb ? 0 : 4)) * x[xo + k];
                    }
                    for (uint k = 0; k < 16u; k++) {
                        int lo = (int)((blk.qs[qb + 16u + k] >> sh) & 3u);
                        int hb = (int)((blk.hmask[16u + k] >> hs) & 1u);
                        acc += ds[sb0 + 1u] * float(lo - (hb ? 0 : 4)) * x[xo + 16u + k];
                    }
                }
            }
            s[r] += acc;
        }
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(s[r]);
            if (lane == 0) y[row] = t;
        }
    }
}

kernel void q4_K_matvec(device const block_q4_K *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 4u;
    const uint br = tgid.x * RPT;
    const uint n_blocks = cols / 256u;
    device const float *x = X + (size_t)tgid.y * cols;
    threadgroup float sx[256];
    float rs[RPT];
    for (uint r = 0; r < RPT; r++) rs[r] = 0.0f;
    for (uint blk = 0; blk < n_blocks; blk++) {
        for (uint i = tiisg; i < 256u; i += sw) sx[i] = x[blk * 256u + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            device const block_q4_K &w = W[(size_t)row * n_blocks + blk];
            for (uint sub = tiisg / 8u; sub < 8u; sub += sw / 8u) {
                uint ln = tiisg % 8u;
                uint2 sm = qK_scale_min(w.scales, sub);
                float d = float(w.d) * float(sm.x);
                float dm = float(w.dmin) * float(sm.y);
                uint qb = (sub / 2u) * 32u;
                uint qs = (sub & 1u) * 4u;
                float acc = 0.0f, xs = 0.0f;
                for (uint k = 0; k < 4u; k++) {
                    uint idx = ln * 4u + k;
                    float qv = float((w.qs[qb + idx] >> qs) & 0xFu);
                    float xv = sx[sub * 32u + idx];
                    acc += qv * xv;
                    xs += xv;
                }
                rs[r] += d * acc - dm * xs;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(rs[r]);
            if (tiisg == 0) y[row] = t;
        }
    }
}

kernel void q5_K_matvec(device const block_q5_K *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 4u;
    const uint br = tgid.x * RPT;
    const uint n_blocks = cols / 256u;
    device const float *x = X + (size_t)tgid.y * cols;
    threadgroup float sx[256];
    float rs[RPT];
    for (uint r = 0; r < RPT; r++) rs[r] = 0.0f;
    for (uint blk = 0; blk < n_blocks; blk++) {
        for (uint i = tiisg; i < 256u; i += sw) sx[i] = x[blk * 256u + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            device const block_q5_K &w = W[(size_t)row * n_blocks + blk];
            for (uint sub = tiisg; sub < 8u; sub += sw) {
                uint2 sm = qK_scale_min(w.scales, sub);
                float ds = float(w.d) * float(sm.x);
                float dm = float(w.dmin) * float(sm.y);
                uint qb = (sub / 2u) * 32u;
                uint qs = (sub & 1u) * 4u;
                float acc = 0.0f, xs = 0.0f;
                for (uint l = 0; l < 32u; l++) {
                    float q = float((w.qs[qb + l] >> qs) & 0xFu) + float((w.qh[l] >> sub) & 1u) * 16.0f;
                    acc += q * sx[sub * 32u + l];
                    xs += sx[sub * 32u + l];
                }
                rs[r] += ds * acc - dm * xs;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(rs[r]);
            if (tiisg == 0) y[row] = t;
        }
    }
}

kernel void q6_K_matvec(device const block_q6_K *W [[buffer(0)]], device const float *X [[buffer(1)]], device float *Y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint tiisg [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    constexpr uint RPT = 4u;
    const uint br = tgid.x * RPT;
    const uint n_blocks = cols / 256u;
    device const float *x = X + (size_t)tgid.y * cols;
    threadgroup float sx[256];
    float rs[RPT];
    for (uint r = 0; r < RPT; r++) rs[r] = 0.0f;
    for (uint blk = 0; blk < n_blocks; blk++) {
        for (uint i = tiisg; i < 256u; i += sw) sx[i] = x[blk * 256u + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < RPT; r++) {
            uint row = br + r;
            if (row >= rows) continue;
            device const block_q6_K &w = W[(size_t)row * n_blocks + blk];
            float d = float(w.d);
            float acc = 0.0f;
            for (uint i = tiisg; i < 256u; i += sw) {
                uint h = i / 128u, q = (i % 128u) / 32u, l = i % 32u;
                uint qi = (q < 2u) ? (h * 64u + q * 32u + l) : (h * 64u + (q - 2u) * 32u + l);
                uint qs = (q < 2u) ? 0u : 4u;
                uint lo = (uint(w.ql[qi]) >> qs) & 0xFu;
                uint hi = (uint(w.qh[h * 32u + l]) >> (q * 2u)) & 3u;
                float sc = float(w.scales[h * 8u + q * 2u + l / 16u]);
                acc += sc * float(int(lo | (hi << 4u)) - 32) * sx[i];
            }
            rs[r] += d * acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *y = Y + (size_t)tgid.y * rows;
    for (uint r = 0; r < RPT; r++) {
        uint row = br + r;
        if (row < rows) {
            float t = simd_sum(rs[r]);
            if (tiisg == 0) y[row] = t;
        }
    }
}
kernel void rms_norm_b(device const float *X [[buffer(0)]], device const float *W [[buffer(1)]], device float *O [[buffer(2)]], constant uint &n [[buffer(3)]], constant float &eps [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    device const float *x = X + (size_t)tgid.y * n;
    device float *o = O + (size_t)tgid.y * n;
    float ss = 0.0f;
    for (uint i = lane; i < n; i += sw) ss += x[i] * x[i];
    ss = simd_sum(ss);
    float inv = rsqrt(ss / float(n) + eps);
    for (uint i = lane; i < n; i += sw) o[i] = x[i] * inv * W[i];
}

kernel void rms_norm_heads(device float *X [[buffer(0)]], device const float *W [[buffer(1)]], constant uint &nh [[buffer(2)]], constant uint &hd [[buffer(3)]], constant float &eps [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    device float *x = X + ((size_t)tgid.y * nh + tgid.x) * hd;
    float ss = 0.0f;
    for (uint i = lane; i < hd; i += sw) ss += x[i] * x[i];
    ss = simd_sum(ss);
    float inv = rsqrt(ss / float(hd) + eps);
    for (uint i = lane; i < hd; i += sw) x[i] = x[i] * inv * W[i];
}

kernel void bias_add_b(device float *X [[buffer(0)]], device const float *B [[buffer(1)]], constant uint &n [[buffer(2)]], uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_grid]]) {
    if (tid >= n) return;
    X[(size_t)tgid.y * n + tid] += B[tid];
}

kernel void rope_b(device float *V [[buffer(0)]], constant uint &pos0 [[buffer(1)]], constant uint &hd [[buffer(2)]], constant uint &nh [[buffer(3)]], constant float &theta [[buffer(4)]], uint tid [[thread_position_in_grid]], uint3 tgid [[threadgroup_position_in_grid]]) {
    uint hp = nh * (hd / 2u);
    if (tid >= hp) return;
    device float *v = V + (size_t)tgid.y * (nh * hd);
    uint pos = pos0 + tgid.y;
    uint h = tid / (hd / 2u), p = tid % (hd / 2u);
    uint b1 = h * hd + p * 2u, b2 = b1 + 1u;
    float freq = pow(theta, -(float)(p * 2u) / (float)hd);
    float a = (float)pos * freq;
    float cv = cos(a), sv = sin(a);
    float v0 = v[b1], v1 = v[b2];
    v[b1] = v0 * cv - v1 * sv;
    v[b2] = v1 * cv + v0 * sv;
}

kernel void kv_cache_store_b(device const float *K [[buffer(0)]], device const float *V [[buffer(1)]], device half *KC [[buffer(2)]], device half *VC [[buffer(3)]], constant uint &pos0 [[buffer(4)]], constant uint &kvd [[buffer(5)]], uint tid [[thread_position_in_grid]], uint3 tgid [[threadgroup_position_in_grid]]) {
    if (tid >= kvd) return;
    uint pos = pos0 + tgid.y;
    KC[(size_t)pos * kvd + tid] = half(K[(size_t)tgid.y * kvd + tid]);
    VC[(size_t)pos * kvd + tid] = half(V[(size_t)tgid.y * kvd + tid]);
}

kernel void attention_gqa(device const float *Q [[buffer(0)]], device const half *K [[buffer(1)]], device const half *V [[buffer(2)]], device float *out [[buffer(3)]], constant uint &sl [[buffer(4)]], constant uint &nqh [[buffer(5)]], constant uint &nkh [[buffer(6)]], constant uint &hd [[buffer(7)]], constant float &sc [[buffer(8)]], uint hid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    if (hid >= nqh) return;
    uint kh = hid / (nqh / nkh);
    uint kvd = nkh * hd;
    device const float *q = Q + hid * hd;
    threadgroup float scores[MAX_CTX];
    for (uint p = 0; p < sl; p++) {
        device const half *k = K + (size_t)p * kvd + kh * hd;
        float part = 0.0f;
        for (uint d = lane; d < hd; d += sw) part += q[d] * float(k[d]);
        float dd = simd_sum(part);
        if (lane == 0) scores[p] = dd * sc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -MAXFLOAT;
    for (uint p = lane; p < sl; p += sw) mx = max(mx, scores[p]);
    mx = simd_max(mx);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sm = 0.0f;
    for (uint p = lane; p < sl; p += sw) {
        float e = exp(scores[p] - mx);
        scores[p] = e;
        sm += e;
    }
    sm = simd_sum(sm);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / sm;
    for (uint p = lane; p < sl; p += sw) scores[p] *= inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = lane; d < hd; d += sw) {
        float s = 0.0f;
        for (uint p = 0; p < sl; p++) s += scores[p] * float(V[(size_t)p * kvd + kh * hd + d]);
        out[hid * hd + d] = s;
    }
}

kernel void silu_mul(device const float *G [[buffer(0)]], device const float *U [[buffer(1)]], device float *O [[buffer(2)]], constant uint &n [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= n) return;
    float gv = G[tid];
    O[tid] = (gv / (1.0f + exp(-gv))) * U[tid];
}

kernel void residual_add(device float *X [[buffer(0)]], device const float *Y [[buffer(1)]], constant uint &n [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= n) return;
    X[tid] += Y[tid];
}

kernel void argmax_write_token(device const float *logits [[buffer(0)]], device uint32_t *output_token [[buffer(1)]], constant uint &vocab_size [[buffer(2)]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
    float best_val = -MAXFLOAT;
    uint best_idx = 0;
    for (uint i = lane; i < vocab_size; i += sw) {
        float v = logits[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    for (uint s = sw / 2u; s > 0; s >>= 1) {
        float other_val = simd_shuffle_down(best_val, s);
        uint other_idx = simd_shuffle_down(best_idx, s);
        if (lane + s < sw && other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }
    if (lane == 0) *output_token = best_idx;
}

kernel void min_p_mask(device float *logits [[buffer(0)]],
                       constant uint &vocab_size [[buffer(1)]],
                       constant float &min_p [[buffer(2)]],
                       threadgroup float *red [[threadgroup(0)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint nt [[threads_per_threadgroup]]) {
    float m = -MAXFLOAT;
    for (uint i = tid; i < vocab_size; i += nt) m = fmax(m, logits[i]);
    red[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2u; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmax(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float thr = min_p * red[0];
    for (uint i = tid; i < vocab_size; i += nt)
        if (logits[i] < thr) logits[i] = -MAXFLOAT;
}

kernel void sample_gumbel(device const float *logits [[buffer(0)]],
                          device uint32_t *output_token [[buffer(1)]],
                          constant uint &vocab_size [[buffer(2)]],
                          constant float &temperature [[buffer(3)]],
                          constant uint &seed [[buffer(4)]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sw [[threads_per_simdgroup]]) {
    float best_val = -MAXFLOAT;
    uint best_idx = 0;
    for (uint i = lane; i < vocab_size; i += sw) {
        uint h = seed ^ (i * 2654435761u);
        h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
        float u = ((float)(h & 0x7FFFFFu) + 1.0f) / 8388609.0f;
        float g = -log(-log(u));
        float v = logits[i] / temperature + g;
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    for (uint s = sw / 2u; s > 0; s >>= 1) {
        float other_val = simd_shuffle_down(best_val, s);
        uint other_idx = simd_shuffle_down(best_idx, s);
        if (lane + s < sw && other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }
    if (lane == 0) *output_token = best_idx;
}

kernel void min_p_mask(device float *logits [[buffer(0)]],
                       constant uint &vocab_size [[buffer(1)]],
                       constant float &min_p [[buffer(2)]],
                       threadgroup float *red [[threadgroup(0)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint nt [[threads_per_threadgroup]]) {
    float m = -MAXFLOAT;
    for (uint i = tid; i < vocab_size; i += nt) m = fmax(m, logits[i]);
    red[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2u; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmax(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float thr = min_p * red[0];
    for (uint i = tid; i < vocab_size; i += nt)
        if (logits[i] < thr) logits[i] = -MAXFLOAT;
}

kernel void sample_gumbel(device const float *logits [[buffer(0)]],
                          device uint32_t *output_token [[buffer(1)]],
                          constant uint &vocab_size [[buffer(2)]],
                          constant float &temperature [[buffer(3)]],
                          constant uint &seed [[buffer(4)]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sw [[threads_per_simdgroup]]) {
    float best_val = -MAXFLOAT;
    uint best_idx = 0;
    for (uint i = lane; i < vocab_size; i += sw) {
        uint h = seed ^ (i * 2654435761u);
        h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
        float u = ((float)(h & 0x7FFFFFu) + 1.0f) / 8388609.0f;
        float g = -log(-log(u));
        float v = logits[i] / temperature + g;
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    for (uint s = sw / 2u; s > 0; s >>= 1) {
        float other_val = simd_shuffle_down(best_val, s);
        uint other_idx = simd_shuffle_down(best_idx, s);
        if (lane + s < sw && other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }
    if (lane == 0) *output_token = best_idx;
}
