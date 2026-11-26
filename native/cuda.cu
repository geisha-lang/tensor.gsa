// CUDA + cuBLAS backend for tensor.gsa (`[[target.native.when]] tool = "nvcc"`).

#ifdef GS_TENSOR_CUDA

#include "backend.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdlib.h>

static cublasHandle_t g_cublas = NULL;

static cublasHandle_t get_cublas(int32_t dev_id) {
    cudaSetDevice(dev_id);
    if (!g_cublas) cublasCreate(&g_cublas);
    return g_cublas;
}

static int32_t cu_device_count(void) { int c = 0; return cudaGetDeviceCount(&c) == cudaSuccess ? (int32_t)c : 0; }

static void *cu_alloc(int32_t dev_id, int64_t nbytes) {
    cudaSetDevice(dev_id);
    void *p = NULL;
    size_t sz = nbytes > 0 ? (size_t)nbytes : 64;
    if (cudaMalloc(&p, sz) != cudaSuccess) return NULL;
    cudaMemset(p, 0, sz);
    return p;
}

static void cu_free(void *ptr) { if (ptr) cudaFree(ptr); }
static void cu_h2d(int32_t d, void *dst, const void *src, int64_t n) { cudaSetDevice(d); if (n > 0) cudaMemcpy(dst, src, (size_t)n, cudaMemcpyHostToDevice); }
static void cu_d2h(int32_t d, void *dst, const void *src, int64_t n) { cudaSetDevice(d); if (n > 0) cudaMemcpy(dst, src, (size_t)n, cudaMemcpyDeviceToHost); }
static void cu_d2d(int32_t d, void *dst, const void *src, int64_t n) { cudaSetDevice(d); if (n > 0 && dst != src) cudaMemcpy(dst, src, (size_t)n, cudaMemcpyDeviceToDevice); }

static void cu_sgemm(int32_t d, int32_t ta, int32_t tb, int32_t M, int32_t N, int32_t K, float alpha, const float *A, const float *B, float beta, float *C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    cublasSgemm(get_cublas(d), tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
                N, M, K, &alpha, B, tb ? K : N, A, ta ? M : K, &beta, C, N);
}

static void cu_bgemm(int32_t d, int32_t batch, int32_t ta, int32_t tb, int32_t M, int32_t N, int32_t K, float alpha, const float *A, const float *B, float beta, float *C) {
    if (batch <= 0 || M <= 0 || N <= 0 || K <= 0) return;
    cublasSgemmStridedBatched(get_cublas(d), tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
        N, M, K, &alpha, B, tb ? K : N, (long long)K * N, A, ta ? M : K, (long long)M * K, &beta, C, N, (long long)M * N, batch);
}

static inline int grid1d(int64_t n) { return (int)((n + 255) / 256); }

__global__ void k_fill(float *d, int64_t n, float v) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) d[i] = v; }
__global__ void k_add(const float *a, const float *b, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = a[i] + b[i]; }
__global__ void k_sub(const float *a, const float *b, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = a[i] - b[i]; }
__global__ void k_mul(const float *a, const float *b, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = a[i] * b[i]; }
__global__ void k_div(const float *a, const float *b, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = a[i] / b[i]; }
__global__ void k_scale(const float *a, float s, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = a[i] * s; }
__global__ void k_add_s(const float *a, float s, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = a[i] + s; }
__global__ void k_pow_f(const float *a, float p, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = powf(a[i], p); }
__global__ void k_pow_b(const float *a, const float *gy, float p, float *gx, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) gx[i] = gy[i] * p * powf(a[i], p - 1.0f); }
__global__ void k_clamp_f(const float *a, float lo, float hi, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) { float v = a[i]; o[i] = v < lo ? lo : (v > hi ? hi : v); } }
__global__ void k_clamp_b(const float *a, const float *gy, float lo, float hi, float *gx, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) { float v = a[i]; gx[i] = (v >= lo && v <= hi) ? gy[i] : 0.0f; } }
__global__ void k_lrelu_f(const float *a, float s, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = a[i] > 0.0f ? a[i] : s * a[i]; }
__global__ void k_lrelu_b(const float *a, const float *gy, float s, float *gx, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) gx[i] = a[i] > 0.0f ? gy[i] : s * gy[i]; }
__global__ void k_axpy(float a, const float *x, float *y, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) y[i] += a * x[i]; }

typedef struct { int32_t sh[8], sa[8], sb[8]; } BcastMeta;

__global__ void k_bcast_binop(int32_t op, int32_t ndim, int64_t total, BcastMeta m, const float *a, const float *b, float *out) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= total) return;
    int64_t rem = idx, oa = 0, ob = 0;
    for (int32_t d = ndim - 1; d >= 0; d--) {
        int32_t c = (int32_t)(rem % m.sh[d]);
        rem /= m.sh[d];
        oa += (int64_t)c * m.sa[d];
        ob += (int64_t)c * m.sb[d];
    }
    float va = a[oa], vb = b[ob];
    out[idx] = (op == GS_BOP_ADD) ? (va + vb) : (op == GS_BOP_SUB) ? (va - vb) : (op == GS_BOP_MUL) ? (va * vb) : (va / vb);
}

__global__ void k_bcast_reduce_sum(int32_t ndim, int64_t src_total, BcastMeta m, const float *src, float *dst) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= src_total) return;
    int64_t rem = idx, od = 0;
    for (int32_t d = ndim - 1; d >= 0; d--) {
        int32_t c = (int32_t)(rem % m.sh[d]);
        rem /= m.sh[d];
        od += (int64_t)c * m.sa[d];
    }
    atomicAdd(&dst[od], src[idx]);
}

__global__ void k_unary_fwd(int32_t op, const float *x, float *o, int64_t n) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    const float gc = 0.7978845608028654f;
    switch (op) {
        case GS_UOP_EXP:     o[i] = expf(v); break;
        case GS_UOP_LOG:     o[i] = logf(v); break;
        case GS_UOP_SQRT:    o[i] = sqrtf(v); break;
        case GS_UOP_RSQRT:   o[i] = rsqrtf(v); break;
        case GS_UOP_ABS:     o[i] = fabsf(v); break;
        case GS_UOP_SQUARE:  o[i] = v * v; break;
        case GS_UOP_TANH:    o[i] = tanhf(v); break;
        case GS_UOP_SIGMOID: o[i] = 1.0f / (1.0f + expf(-v)); break;
        case GS_UOP_ROUND:   o[i] = roundf(v); break;
        case GS_UOP_SILU:    o[i] = v / (1.0f + expf(-v)); break;
        case GS_UOP_GELU:    o[i] = 0.5f * v * (1.0f + tanhf(gc * (v + 0.044715f * v * v * v))); break;
        case GS_UOP_RELU:    o[i] = v > 0.0f ? v : 0.0f; break;
    }
}

__global__ void k_unary_bwd(int32_t op, const float *x, const float *y, const float *gy, float *gx, int64_t n) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i >= n) return;
    float v = x[i], yv = y[i], g = gy[i];
    const float gc = 0.7978845608028654f;
    switch (op) {
        case GS_UOP_EXP:     gx[i] = g * yv; break;
        case GS_UOP_LOG:     gx[i] = g / v; break;
        case GS_UOP_SQRT:    gx[i] = g / (2.0f * yv + 1e-12f); break;
        case GS_UOP_RSQRT:   gx[i] = -0.5f * g * yv * yv * yv; break;
        case GS_UOP_ABS:     gx[i] = v > 0.0f ? g : (v < 0.0f ? -g : 0.0f); break;
        case GS_UOP_SQUARE:  gx[i] = 2.0f * v * g; break;
        case GS_UOP_TANH:    gx[i] = g * (1.0f - yv * yv); break;
        case GS_UOP_SIGMOID: gx[i] = g * yv * (1.0f - yv); break;
        case GS_UOP_ROUND:   gx[i] = 0.0f; break;
        case GS_UOP_SILU:    { float s = 1.0f / (1.0f + expf(-v)); gx[i] = g * s * (1.0f + v * (1.0f - s)); break; }
        case GS_UOP_GELU:    { float v2 = v * v; float th = tanhf(gc * (v + 0.044715f * v * v2)); gx[i] = g * (0.5f * (1.0f + th) + 0.5f * v * (1.0f - th * th) * gc * (1.0f + 0.134145f * v2)); break; }
        case GS_UOP_RELU:    gx[i] = v > 0.0f ? g : 0.0f; break;
    }
}

__global__ void k_slice_f(const float *x, float *o, int32_t outer, int32_t in_dim, int32_t inner, int32_t start, int32_t len) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= (int64_t)outer * len * inner) return;
    int32_t k = (int32_t)(idx % inner), l = (int32_t)((idx / inner) % len), o0 = (int32_t)(idx / ((int64_t)len * inner));
    o[idx] = x[((int64_t)o0 * in_dim + start + l) * inner + k];
}
__global__ void k_slice_b(const float *gy, float *gx, int32_t outer, int32_t in_dim, int32_t inner, int32_t start, int32_t len) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= (int64_t)outer * len * inner) return;
    int32_t k = (int32_t)(idx % inner), l = (int32_t)((idx / inner) % len), o0 = (int32_t)(idx / ((int64_t)len * inner));
    gx[((int64_t)o0 * in_dim + start + l) * inner + k] = gy[idx];
}
__global__ void k_sum_axis(const float *x, float *o, int32_t outer, int32_t dim, int32_t inner) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= (int64_t)outer * inner) return;
    int32_t o0 = (int32_t)(idx / inner), k = (int32_t)(idx % inner);
    float acc = 0.0f;
    for (int32_t d = 0; d < dim; d++) acc += x[((int64_t)o0 * dim + d) * inner + k];
    o[idx] = acc;
}
__global__ void k_max_axis(const float *x, float *o, int32_t outer, int32_t dim, int32_t inner) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= (int64_t)outer * inner) return;
    int32_t o0 = (int32_t)(idx / inner), k = (int32_t)(idx % inner);
    float mx = x[((int64_t)o0 * dim) * inner + k];
    for (int32_t d = 1; d < dim; d++) { float v = x[((int64_t)o0 * dim + d) * inner + k]; if (v > mx) mx = v; }
    o[idx] = mx;
}

__global__ void k_rms_f(const float *x, const float *w, float *o, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim; float *orow = o + (int64_t)r * dim;
    float ss = 0.0f; for (int32_t c = 0; c < dim; c++) ss += xr[c] * xr[c];
    float inv = rsqrtf(ss / (float)dim + eps);
    for (int32_t c = threadIdx.x; c < dim; c += blockDim.x) orow[c] = xr[c] * inv * w[c];
}
__global__ void k_rms_b(const float *x, const float *w, const float *gy, float *gx, float *gw, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim, *gyr = gy + (int64_t)r * dim; float *gxr = gx + (int64_t)r * dim;
    float ss = 0.0f, dot = 0.0f; for (int32_t c = 0; c < dim; c++) { ss += xr[c] * xr[c]; dot += gyr[c] * w[c] * xr[c]; }
    float inv = rsqrtf(ss / (float)dim + eps), coeff = dot * (inv * inv * inv) / (float)dim;
    for (int32_t c = threadIdx.x; c < dim; c += blockDim.x) { atomicAdd(&gw[c], gyr[c] * xr[c] * inv); gxr[c] = gyr[c] * w[c] * inv - xr[c] * coeff; }
}
__global__ void k_ln_f(const float *x, const float *w, const float *b, float *o, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim; float *orow = o + (int64_t)r * dim;
    float mean = 0.0f; for (int32_t c = 0; c < dim; c++) mean += xr[c]; mean /= (float)dim;
    float var = 0.0f; for (int32_t c = 0; c < dim; c++) { float d = xr[c] - mean; var += d * d; }
    float inv = rsqrtf(var / (float)dim + eps);
    for (int32_t c = threadIdx.x; c < dim; c += blockDim.x) orow[c] = (xr[c] - mean) * inv * w[c] + (b ? b[c] : 0.0f);
}
__global__ void k_ln_b(const float *x, const float *w, const float *gy, float *gx, float *gw, float *gb, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim, *gyr = gy + (int64_t)r * dim; float *gxr = gx + (int64_t)r * dim;
    float mean = 0.0f; for (int32_t c = 0; c < dim; c++) mean += xr[c]; mean /= (float)dim;
    float var = 0.0f; for (int32_t c = 0; c < dim; c++) { float d = xr[c] - mean; var += d * d; }
    float inv = rsqrtf(var / (float)dim + eps);
    float s1 = 0.0f, s2 = 0.0f;
    for (int32_t c = 0; c < dim; c++) { float xh = (xr[c] - mean) * inv; float dx = gyr[c] * w[c]; s1 += dx; s2 += dx * xh; }
    for (int32_t c = threadIdx.x; c < dim; c += blockDim.x) {
        float xh = (xr[c] - mean) * inv;
        atomicAdd(&gw[c], gyr[c] * xh);
        if (gb) atomicAdd(&gb[c], gyr[c]);
        gxr[c] = inv * (gyr[c] * w[c] - (s1 + xh * s2) / (float)dim);
    }
}
__global__ void k_l2_f(const float *x, float *o, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim; float *orow = o + (int64_t)r * dim;
    float ss = 0.0f; for (int32_t c = 0; c < dim; c++) ss += xr[c] * xr[c];
    float norm = sqrtf(ss); float inv = 1.0f / (norm > eps ? norm : eps);
    for (int32_t c = threadIdx.x; c < dim; c += blockDim.x) orow[c] = xr[c] * inv;
}
__global__ void k_l2_b(const float *x, const float *gy, float *gx, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim, *gyr = gy + (int64_t)r * dim; float *gxr = gx + (int64_t)r * dim;
    float ss = 0.0f, dot = 0.0f; for (int32_t c = 0; c < dim; c++) { ss += xr[c] * xr[c]; dot += xr[c] * gyr[c]; }
    float norm = sqrtf(ss);
    if (norm <= eps) { float inv = 1.0f / eps; for (int32_t c = threadIdx.x; c < dim; c += blockDim.x) gxr[c] = gyr[c] * inv; }
    else { float inv = 1.0f / norm, coeff = dot * inv * inv * inv; for (int32_t c = threadIdx.x; c < dim; c += blockDim.x) gxr[c] = gyr[c] * inv - xr[c] * coeff; }
}
__global__ void k_sm_f(const float *x, float *o, int32_t rows, int32_t cols) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * cols; float *orow = o + (int64_t)r * cols;
    float mx = xr[0]; for (int32_t c = 1; c < cols; c++) if (xr[c] > mx) mx = xr[c];
    float sum = 0.0f; for (int32_t c = 0; c < cols; c++) { float e = expf(xr[c] - mx); orow[c] = e; sum += e; }
    float inv = 1.0f / sum; for (int32_t c = 0; c < cols; c++) orow[c] *= inv;
}
__global__ void k_sm_b(const float *y, const float *gy, float *gx, int32_t rows, int32_t cols) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *yr = y + (int64_t)r * cols, *gyr = gy + (int64_t)r * cols; float *gxr = gx + (int64_t)r * cols;
    float dot = 0.0f; for (int32_t c = 0; c < cols; c++) dot += yr[c] * gyr[c];
    for (int32_t c = 0; c < cols; c++) gxr[c] = yr[c] * (gyr[c] - dot);
}
__global__ void k_lse_f(const float *x, float *o, int32_t rows, int32_t cols) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * cols;
    float mx = xr[0]; for (int32_t c = 1; c < cols; c++) if (xr[c] > mx) mx = xr[c];
    float sum = 0.0f; for (int32_t c = 0; c < cols; c++) sum += expf(xr[c] - mx);
    o[r] = mx + logf(sum);
}
__global__ void k_lse_b(const float *x, const float *lse, const float *gy, float *gx, int32_t rows, int32_t cols) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= (int64_t)rows * cols) return;
    int32_t r = (int32_t)(idx / cols);
    gx[idx] = gy[r] * expf(x[idx] - lse[r]);
}
__global__ void k_ce_bwd(const float *probs, const int32_t *targets, float scale, float *gl, int32_t rows, int32_t vocab, int32_t ignore_idx) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= (int64_t)rows * vocab) return;
    int32_t r = (int32_t)(idx / vocab), c = (int32_t)(idx % vocab), tgt = targets[r];
    gl[idx] = (tgt == ignore_idx || tgt < 0 || tgt >= vocab) ? 0.0f : (probs[idx] * scale - ((c == tgt) ? scale : 0.0f));
}
__global__ void k_emb_f(const float *tab, const int32_t *ids, float *o, int32_t n_ids, int32_t dim) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)n_ids * dim) o[idx] = tab[(int64_t)ids[idx / dim] * dim + (idx % dim)];
}
__global__ void k_emb_b(const int32_t *ids, const float *gy, float *gtab, int32_t n_ids, int32_t vocab, int32_t dim) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)n_ids * dim) { int32_t id = ids[idx / dim]; if (id >= 0 && id < vocab) atomicAdd(&gtab[(int64_t)id * dim + (idx % dim)], gy[idx]); }
}
__global__ void k_rope(const float *in, float *out, int32_t batch, int32_t seq, int32_t heads, int32_t hd, float theta, int32_t bwd) {
    int32_t half = hd / 2; int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= (int64_t)batch * seq * heads * half) return;
    int32_t i = (int32_t)(idx % half); int64_t rem = idx / half;
    int32_t h = (int32_t)(rem % heads); rem /= heads; int32_t t = (int32_t)(rem % seq); int32_t b = (int32_t)(rem / seq);
    int64_t base = (((int64_t)b * seq + t) * heads + h) * hd;
    float ang = (float)t / powf(theta, (float)(2 * i) / (float)hd), cs = cosf(ang), sn = sinf(ang);
    float v0 = in[base + i], v1 = in[base + half + i];
    if (!bwd) { out[base + i] = v0 * cs - v1 * sn; out[base + half + i] = v0 * sn + v1 * cs; }
    else      { out[base + i] = v0 * cs + v1 * sn; out[base + half + i] = -v0 * sn + v1 * cs; }
}
__global__ void k_t2d(const float *x, float *o, int32_t r, int32_t c) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < (int64_t)r * c) o[(i % c) * r + (i / c)] = x[i]; }
__global__ void k_t0213(const float *x, float *o, int32_t d0, int32_t d1, int32_t d2, int32_t d3) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x; if (idx >= (int64_t)d0 * d1 * d2 * d3) return;
    int32_t i3 = (int32_t)(idx % d3); int64_t rem = idx / d3; int32_t i2 = (int32_t)(rem % d2); rem /= d2; int32_t i1 = (int32_t)(rem % d1); int32_t i0 = (int32_t)(rem / d1);
    o[(((int64_t)i0 * d2 + i2) * d1 + i1) * d3 + i3] = x[idx];
}
__global__ void k_causal(float *s, int32_t bh, int32_t seq) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)bh * seq * seq) { if ((int32_t)(idx % seq) > (int32_t)((idx / seq) % seq)) s[idx] += -1.0e9f; }
}
__global__ void k_adamw(const float *p, const float *g, const float *m, const float *v, float *po, float *mo, float *vo, int64_t n, float lr, float b1, float b2, float eps, float wd, float bc1, float bc2) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i < n) { float gi = g[i], mi = b1 * m[i] + (1.0f - b1) * gi, vi = b2 * v[i] + (1.0f - b2) * gi * gi; mo[i] = mi; vo[i] = vi; po[i] = p[i] * (1.0f - lr * wd) - lr * ((mi / bc1) / (sqrtf(vi / bc2) + eps)); }
}

typedef struct {
    const float *p, *g, *m, *v;
    float *po, *mo, *vo;
    int64_t n;
    int64_t block_start;
} AdamwParamDesc;

// Single-launch multi-tensor fused AdamW across all model parameters
__global__ void k_adamw_multi(
    const AdamwParamDesc *descs, int32_t count,
    float lr, float b1, float b2, float eps, float wd, float bc1, float bc2
) {
    int64_t blk = (int64_t)blockIdx.x;
    int32_t lo = 0, hi = count - 1;
    while (lo < hi) {
        int32_t mid = (lo + hi + 1) >> 1;
        if (descs[mid].block_start <= blk) lo = mid;
        else hi = mid - 1;
    }
    const AdamwParamDesc d = descs[lo];
    int64_t i = (blk - d.block_start) * 256LL + (int64_t)threadIdx.x;
    if (i < d.n) {
        float gi = d.g[i];
        float mi = b1 * d.m[i] + (1.0f - b1) * gi;
        float vi = b2 * d.v[i] + (1.0f - b2) * gi * gi;
        d.mo[i] = mi;
        d.vo[i] = vi;
        d.po[i] = d.p[i] * (1.0f - lr * wd) - lr * ((mi / bc1) / (sqrtf(vi / bc2) + eps));
    }
}

static void cu_fill(int32_t d, float *dst, int64_t n, float v) { cudaSetDevice(d); k_fill<<<grid1d(n), 256>>>(dst, n, v); }
static void cu_copy(int32_t d, float *dst, const float *src, int64_t n) { cu_d2d(d, dst, src, n * (int64_t)sizeof(float)); }
static void cu_add(int32_t d, const float *a, const float *b, float *o, int64_t n) { cudaSetDevice(d); k_add<<<grid1d(n), 256>>>(a, b, o, n); }
static void cu_sub(int32_t d, const float *a, const float *b, float *o, int64_t n) { cudaSetDevice(d); k_sub<<<grid1d(n), 256>>>(a, b, o, n); }
static void cu_mul(int32_t d, const float *a, const float *b, float *o, int64_t n) { cudaSetDevice(d); k_mul<<<grid1d(n), 256>>>(a, b, o, n); }
static void cu_div_ew(int32_t d, const float *a, const float *b, float *o, int64_t n) { cudaSetDevice(d); k_div<<<grid1d(n), 256>>>(a, b, o, n); }
static void cu_scale(int32_t d, const float *a, float s, float *o, int64_t n) { cudaSetDevice(d); k_scale<<<grid1d(n), 256>>>(a, s, o, n); }
static void cu_add_s(int32_t d, const float *a, float s, float *o, int64_t n) { cudaSetDevice(d); k_add_s<<<grid1d(n), 256>>>(a, s, o, n); }
static void cu_pow_f(int32_t d, const float *a, float p, float *o, int64_t n) { cudaSetDevice(d); k_pow_f<<<grid1d(n), 256>>>(a, p, o, n); }
static void cu_pow_b(int32_t d, const float *a, const float *gy, float p, float *gx, int64_t n) { cudaSetDevice(d); k_pow_b<<<grid1d(n), 256>>>(a, gy, p, gx, n); }
static void cu_clamp_f(int32_t d, const float *a, float lo, float hi, float *o, int64_t n) { cudaSetDevice(d); k_clamp_f<<<grid1d(n), 256>>>(a, lo, hi, o, n); }
static void cu_clamp_b(int32_t d, const float *a, const float *gy, float lo, float hi, float *gx, int64_t n) { cudaSetDevice(d); k_clamp_b<<<grid1d(n), 256>>>(a, gy, lo, hi, gx, n); }
static void cu_lrelu_f(int32_t d, const float *a, float s, float *o, int64_t n) { cudaSetDevice(d); k_lrelu_f<<<grid1d(n), 256>>>(a, s, o, n); }
static void cu_lrelu_b(int32_t d, const float *a, const float *gy, float s, float *gx, int64_t n) { cudaSetDevice(d); k_lrelu_b<<<grid1d(n), 256>>>(a, gy, s, gx, n); }
static void cu_axpy(int32_t d, float a, const float *x, float *y, int64_t n) { cudaSetDevice(d); k_axpy<<<grid1d(n), 256>>>(a, x, y, n); }

static void cu_bcast_binop(int32_t d, int32_t op, int32_t ndim, int64_t total, const int32_t *sh, const int32_t *sa, const int32_t *sb, const float *a, const float *b, float *o) {
    BcastMeta m = {{0}, {0}, {0}};
    for (int32_t i = 0; i < ndim && i < 8; i++) { m.sh[i] = sh[i]; m.sa[i] = sa[i]; m.sb[i] = sb[i]; }
    cudaSetDevice(d);
    k_bcast_binop<<<grid1d(total), 256>>>(op, ndim, total, m, a, b, o);
}
static void cu_bcast_reduce_sum(int32_t d, int32_t ndim, int64_t src_total, int64_t dst_total, const int32_t *sh, const int32_t *sd, const float *src, float *dst) {
    BcastMeta m = {{0}, {0}, {0}};
    for (int32_t i = 0; i < ndim && i < 8; i++) { m.sh[i] = sh[i]; m.sa[i] = sd[i]; }
    cudaSetDevice(d);
    cudaMemset(dst, 0, (size_t)dst_total * sizeof(float));
    k_bcast_reduce_sum<<<grid1d(src_total), 256>>>(ndim, src_total, m, src, dst);
}
static void cu_unary_f(int32_t d, int32_t op, const float *x, float *o, int64_t n) { cudaSetDevice(d); k_unary_fwd<<<grid1d(n), 256>>>(op, x, o, n); }
static void cu_unary_b(int32_t d, int32_t op, const float *x, const float *y, const float *gy, float *gx, int64_t n) { cudaSetDevice(d); k_unary_bwd<<<grid1d(n), 256>>>(op, x, y, gy, gx, n); }
static void cu_slice_f(int32_t d, const float *x, float *o, int32_t out0, int32_t id, int32_t in0, int32_t st, int32_t len) { cudaSetDevice(d); k_slice_f<<<grid1d((int64_t)out0*len*in0), 256>>>(x, o, out0, id, in0, st, len); }
static void cu_slice_b(int32_t d, const float *gy, float *gx, int32_t out0, int32_t id, int32_t in0, int32_t st, int32_t len) { cudaSetDevice(d); cudaMemset(gx, 0, (size_t)((int64_t)out0*id*in0)*sizeof(float)); k_slice_b<<<grid1d((int64_t)out0*len*in0), 256>>>(gy, gx, out0, id, in0, st, len); }
static void cu_sum_ax(int32_t d, const float *x, float *o, int32_t out0, int32_t dim, int32_t in0) { cudaSetDevice(d); k_sum_axis<<<grid1d((int64_t)out0*in0), 256>>>(x, o, out0, dim, in0); }
static void cu_max_ax(int32_t d, const float *x, float *o, int32_t out0, int32_t dim, int32_t in0) { cudaSetDevice(d); k_max_axis<<<grid1d((int64_t)out0*in0), 256>>>(x, o, out0, dim, in0); }
static float cu_sum_all(int32_t d, const float *x, int64_t n) { float *h = (float *)malloc((size_t)n * sizeof(float)); cu_d2h(d, h, x, n * (int64_t)sizeof(float)); float r = gs_backend_cpu.sum_all(0, h, n); free(h); return r; }
static float cu_sum_sq(int32_t d, const float *x, int64_t n) { float *h = (float *)malloc((size_t)n * sizeof(float)); cu_d2h(d, h, x, n * (int64_t)sizeof(float)); float r = gs_backend_cpu.sum_sq(0, h, n); free(h); return r; }
static void cu_rms_f(int32_t d, const float *x, const float *w, float *o, int32_t r, int32_t dim, float eps) { cudaSetDevice(d); k_rms_f<<<r, 256>>>(x, w, o, r, dim, eps); }
static void cu_rms_b(int32_t d, const float *x, const float *w, const float *gy, float *gx, float *gw, int32_t r, int32_t dim, float eps) { cudaSetDevice(d); cudaMemset(gw, 0, (size_t)dim*sizeof(float)); k_rms_b<<<r, 256>>>(x, w, gy, gx, gw, r, dim, eps); }
static void cu_ln_f(int32_t d, const float *x, const float *w, const float *b, float *o, int32_t r, int32_t dim, float eps) { cudaSetDevice(d); k_ln_f<<<r, 256>>>(x, w, b, o, r, dim, eps); }
static void cu_ln_b(int32_t d, const float *x, const float *w, const float *gy, float *gx, float *gw, float *gb, int32_t r, int32_t dim, float eps) { cudaSetDevice(d); cudaMemset(gw, 0, (size_t)dim*sizeof(float)); if (gb) cudaMemset(gb, 0, (size_t)dim*sizeof(float)); k_ln_b<<<r, 256>>>(x, w, gy, gx, gw, gb, r, dim, eps); }
static void cu_l2_f(int32_t d, const float *x, float *o, int32_t r, int32_t dim, float eps) { cudaSetDevice(d); k_l2_f<<<r, 256>>>(x, o, r, dim, eps); }
static void cu_l2_b(int32_t d, const float *x, const float *gy, float *gx, int32_t r, int32_t dim, float eps) { cudaSetDevice(d); k_l2_b<<<r, 256>>>(x, gy, gx, r, dim, eps); }
static void cu_sm_f(int32_t d, const float *x, float *o, int32_t r, int32_t c) { cudaSetDevice(d); k_sm_f<<<r, 1>>>(x, o, r, c); }
static void cu_sm_b(int32_t d, const float *y, const float *gy, float *gx, int32_t r, int32_t c) { cudaSetDevice(d); k_sm_b<<<r, 1>>>(y, gy, gx, r, c); }
static void cu_lse_f(int32_t d, const float *x, float *o, int32_t r, int32_t c) { cudaSetDevice(d); k_lse_f<<<r, 1>>>(x, o, r, c); }
static void cu_lse_b(int32_t d, const float *x, const float *lse, const float *gy, float *gx, int32_t r, int32_t c) { cudaSetDevice(d); k_lse_b<<<grid1d((int64_t)r*c), 256>>>(x, lse, gy, gx, r, c); }
static float cu_ce_f(int32_t d, const float *logits, const int32_t *targets, float *probs_out, int32_t rows, int32_t vocab, int32_t ignore_idx) {
    cu_sm_f(d, logits, probs_out, rows, vocab);
    float *hp = (float *)malloc((size_t)((int64_t)rows * vocab) * sizeof(float));
    int32_t *ht = (int32_t *)malloc((size_t)rows * sizeof(int32_t));
    cu_d2h(d, hp, probs_out, (int64_t)rows * vocab * (int64_t)sizeof(float));
    cu_d2h(d, ht, targets, (int64_t)rows * (int64_t)sizeof(int32_t));
    double loss = 0.0; int32_t valid = 0;
    for (int32_t r = 0; r < rows; r++) { int32_t t = ht[r]; if (t != ignore_idx && t >= 0 && t < vocab) { loss -= log((double)hp[(int64_t)r * vocab + t] + 1e-12); valid++; } }
    free(hp); free(ht);
    return valid > 0 ? (float)(loss / (double)valid) : 0.0f;
}
static void cu_ce_b(int32_t d, const float *probs, const int32_t *targets, float g_loss, float *gl, int32_t rows, int32_t vocab, int32_t ignore_idx) {
    int32_t *ht = (int32_t *)malloc((size_t)rows * sizeof(int32_t));
    cu_d2h(d, ht, targets, (int64_t)rows * (int64_t)sizeof(int32_t));
    int32_t valid = 0;
    for (int32_t r = 0; r < rows; r++) if (ht[r] != ignore_idx && ht[r] >= 0 && ht[r] < vocab) valid++;
    free(ht);
    cudaSetDevice(d);
    if (valid <= 0) { cudaMemset(gl, 0, (size_t)((int64_t)rows * vocab) * sizeof(float)); return; }
    k_ce_bwd<<<grid1d((int64_t)rows*vocab), 256>>>(probs, targets, g_loss / (float)valid, gl, rows, vocab, ignore_idx);
}
static void cu_emb_f(int32_t d, const float *tab, const int32_t *ids, float *o, int32_t n, int32_t dim) { cudaSetDevice(d); k_emb_f<<<grid1d((int64_t)n*dim), 256>>>(tab, ids, o, n, dim); }
static void cu_emb_b(int32_t d, const int32_t *ids, const float *gy, float *gtab, int32_t n, int32_t v, int32_t dim) { cudaSetDevice(d); cudaMemset(gtab, 0, (size_t)((int64_t)v*dim)*sizeof(float)); k_emb_b<<<grid1d((int64_t)n*dim), 256>>>(ids, gy, gtab, n, v, dim); }
static void cu_rope_f(int32_t d, const float *x, float *o, int32_t b, int32_t s, int32_t h, int32_t hd, float th) { cudaSetDevice(d); k_rope<<<grid1d((int64_t)b*s*h*(hd/2)), 256>>>(x, o, b, s, h, hd, th, 0); }
static void cu_rope_b(int32_t d, const float *gy, float *gx, int32_t b, int32_t s, int32_t h, int32_t hd, float th) { cudaSetDevice(d); k_rope<<<grid1d((int64_t)b*s*h*(hd/2)), 256>>>(gy, gx, b, s, h, hd, th, 1); }
static void cu_t2d(int32_t d, const float *x, float *o, int32_t r, int32_t c) { cudaSetDevice(d); k_t2d<<<grid1d((int64_t)r*c), 256>>>(x, o, r, c); }
static void cu_t0213(int32_t d, const float *x, float *o, int32_t d0, int32_t d1, int32_t d2, int32_t d3) { cudaSetDevice(d); k_t0213<<<grid1d((int64_t)d0*d1*d2*d3), 256>>>(x, o, d0, d1, d2, d3); }
static void cu_causal(int32_t d, float *s, int32_t bh, int32_t seq) { cudaSetDevice(d); k_causal<<<grid1d((int64_t)bh*seq*seq), 256>>>(s, bh, seq); }
static void cu_randn(int32_t d, float *dst, int64_t n, int64_t seed, float std) { float *h = (float *)malloc((size_t)n * sizeof(float)); gs_backend_cpu.randn(0, h, n, seed, std); cu_h2d(d, dst, h, n * (int64_t)sizeof(float)); free(h); }
static void cu_adamw(int32_t d, const float *p, const float *g, const float *m, const float *v, float *po, float *mo, float *vo, int64_t n, float lr, float b1, float b2, float eps, float wd, float bc1, float bc2) { cudaSetDevice(d); k_adamw<<<grid1d(n), 256>>>(p, g, m, v, po, mo, vo, n, lr, b1, b2, eps, wd, bc1, bc2); }

static void cu_adamw_multi(
    int32_t d, int32_t count,
    const float *const *ps, const float *const *gs,
    const float *const *ms, const float *const *vs,
    float *const *p_outs, float *const *m_outs, float *const *v_outs,
    const int64_t *numels,
    float lr, float b1, float b2, float eps, float wd, float bc1, float bc2
) {
    if (count <= 0) return;
    AdamwParamDesc *h_descs = (AdamwParamDesc *)malloc((size_t)count * sizeof(AdamwParamDesc));
    if (!h_descs) return;
    int64_t total_blocks = 0;
    for (int32_t t = 0; t < count; t++) {
        h_descs[t].p = ps[t];
        h_descs[t].g = gs[t];
        h_descs[t].m = ms[t];
        h_descs[t].v = vs[t];
        h_descs[t].po = p_outs[t];
        h_descs[t].mo = m_outs[t];
        h_descs[t].vo = v_outs[t];
        h_descs[t].n = numels[t];
        h_descs[t].block_start = total_blocks;
        total_blocks += (numels[t] + 255) / 256;
    }
    if (total_blocks > 0) {
        cudaSetDevice(d);
        AdamwParamDesc *d_descs = NULL;
        size_t bytes = (size_t)count * sizeof(AdamwParamDesc);
        if (cudaMalloc(&d_descs, bytes) == cudaSuccess) {
            cudaMemcpy(d_descs, h_descs, bytes, cudaMemcpyHostToDevice);
            k_adamw_multi<<<(int)total_blocks, 256>>>(d_descs, count, lr, b1, b2, eps, wd, bc1, bc2);
            cudaFree(d_descs);
        }
    }
    free(h_descs);
}

const GsTensorBackend gs_backend_cuda = {
    .name = "cuda+cublas", .dev_kind = GS_DEV_CUDA, .device_count = cu_device_count,
    .alloc = cu_alloc, .free_ptr = cu_free, .copy_h2d = cu_h2d, .copy_d2h = cu_d2h, .copy_d2d = cu_d2d,
    .sgemm = cu_sgemm, .bgemm = cu_bgemm, .fill = cu_fill, .copy = cu_copy,
    .add = cu_add, .sub = cu_sub, .mul = cu_mul, .div_ew = cu_div_ew,
    .scale = cu_scale, .add_scalar = cu_add_s, .pow_scalar_fwd = cu_pow_f, .pow_scalar_bwd = cu_pow_b,
    .clamp_fwd = cu_clamp_f, .clamp_bwd = cu_clamp_b, .leaky_relu_fwd = cu_lrelu_f, .leaky_relu_bwd = cu_lrelu_b,
    .axpy = cu_axpy, .bcast_binop = cu_bcast_binop, .bcast_reduce_sum = cu_bcast_reduce_sum,
    .unary_fwd = cu_unary_f, .unary_bwd = cu_unary_b,
    .slice_axis_fwd = cu_slice_f, .slice_axis_bwd = cu_slice_b, .sum_axis = cu_sum_ax, .max_axis = cu_max_ax,
    .sum_all = cu_sum_all, .sum_sq = cu_sum_sq,
    .rms_norm_fwd = cu_rms_f, .rms_norm_bwd = cu_rms_b, .layer_norm_fwd = cu_ln_f, .layer_norm_bwd = cu_ln_b,
    .l2_norm_last_fwd = cu_l2_f, .l2_norm_last_bwd = cu_l2_b,
    .softmax_fwd = cu_sm_f, .softmax_bwd = cu_sm_b, .log_sum_exp_fwd = cu_lse_f, .log_sum_exp_bwd = cu_lse_b,
    .cross_entropy_fwd = cu_ce_f, .cross_entropy_bwd = cu_ce_b,
    .embedding_fwd = cu_emb_f, .embedding_bwd = cu_emb_b, .rope_fwd = cu_rope_f, .rope_bwd = cu_rope_b,
    .transpose_2d = cu_t2d, .transpose_0213 = cu_t0213, .causal_mask_add = cu_causal,
    .randn = cu_randn, .adamw_step = cu_adamw, .adamw_multi_step = cu_adamw_multi,
};

#endif // GS_TENSOR_CUDA
