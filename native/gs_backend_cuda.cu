// CUDA + cuBLAS backend for tensor.gsa (`-DGS_TENSOR_CUDA -lcudart -lcublas`).
// Compiled with `nvcc -O3 -DGS_TENSOR_CUDA -c native/gs_backend_cuda.cu -o build/gs_backend_cuda.o`.

#ifdef GS_TENSOR_CUDA

#include "gs_tensor_backend.h"
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

static int32_t cu_device_count(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return (int32_t)count;
}

static void *cu_alloc(int32_t dev_id, int64_t nbytes) {
    cudaSetDevice(dev_id);
    void *p = NULL;
    size_t sz = nbytes > 0 ? (size_t)nbytes : 64;
    if (cudaMalloc(&p, sz) != cudaSuccess) return NULL;
    cudaMemset(p, 0, sz);
    return p;
}

static void cu_free(void *ptr) { if (ptr) cudaFree(ptr); }

static void cu_h2d(int32_t dev_id, void *dst, const void *src, int64_t n) {
    cudaSetDevice(dev_id);
    if (n > 0) cudaMemcpy(dst, src, (size_t)n, cudaMemcpyHostToDevice);
}
static void cu_d2h(int32_t dev_id, void *dst, const void *src, int64_t n) {
    cudaSetDevice(dev_id);
    if (n > 0) cudaMemcpy(dst, src, (size_t)n, cudaMemcpyDeviceToHost);
}
static void cu_d2d(int32_t dev_id, void *dst, const void *src, int64_t n) {
    cudaSetDevice(dev_id);
    if (n > 0 && dst != src) cudaMemcpy(dst, src, (size_t)n, cudaMemcpyDeviceToDevice);
}

// Row-major C[M,N] = alpha * op(A) * op(B) + beta * C via col-major C^T = op(B)^T * op(A)^T
static void cu_sgemm(int32_t dev_id, int32_t ta, int32_t tb, int32_t M, int32_t N, int32_t K, float alpha, const float *A, const float *B, float beta, float *C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    cublasHandle_t h = get_cublas(dev_id);
    int lda = ta ? M : K;
    int ldb = tb ? K : N;
    cublasSgemm(h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
                N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
}

static void cu_bgemm(int32_t dev_id, int32_t batch, int32_t ta, int32_t tb, int32_t M, int32_t N, int32_t K, float alpha, const float *A, const float *B, float beta, float *C) {
    if (batch <= 0 || M <= 0 || N <= 0 || K <= 0) return;
    cublasHandle_t h = get_cublas(dev_id);
    int lda = ta ? M : K;
    int ldb = tb ? K : N;
    cublasSgemmStridedBatched(
        h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
        N, M, K, &alpha,
        B, ldb, (long long)K * N,
        A, lda, (long long)M * K,
        &beta, C, N, (long long)M * N, batch
    );
}

static inline int grid1d(int64_t n) { return (int)((n + 255) / 256); }

__global__ void k_fill(float *d, int64_t n, float v) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) d[i] = v; }
__global__ void k_add(const float *a, const float *b, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = a[i] + b[i]; }
__global__ void k_sub(const float *a, const float *b, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = a[i] - b[i]; }
__global__ void k_mul(const float *a, const float *b, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = a[i] * b[i]; }
__global__ void k_scale(const float *a, float s, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = a[i] * s; }
__global__ void k_axpy(float a, const float *x, float *y, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) y[i] += a * x[i]; }
__global__ void k_silu_fwd(const float *x, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) { float v = x[i]; o[i] = v / (1.0f + expf(-v)); } }
__global__ void k_silu_bwd(const float *x, const float *gy, float *gx, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) { float v = x[i]; float s = 1.0f / (1.0f + expf(-v)); gx[i] = gy[i] * s * (1.0f + v * (1.0f - s)); } }
__global__ void k_gelu_fwd(const float *x, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) { float v = x[i]; o[i] = 0.5f * v * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v))); } }
__global__ void k_gelu_bwd(const float *x, const float *gy, float *gx, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) { float v = x[i], v2 = v * v; float th = tanhf(0.79788456f * (v + 0.044715f * v * v2)); gx[i] = gy[i] * (0.5f * (1.0f + th) + 0.5f * v * (1.0f - th * th) * 0.79788456f * (1.0f + 0.134145f * v2)); } }
__global__ void k_relu_fwd(const float *x, float *o, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) o[i] = x[i] > 0.0f ? x[i] : 0.0f; }
__global__ void k_relu_bwd(const float *x, const float *gy, float *gx, int64_t n) { int64_t i = blockIdx.x * 256LL + threadIdx.x; if (i < n) gx[i] = x[i] > 0.0f ? gy[i] : 0.0f; }
__global__ void k_adamw(const float *p, const float *g, const float *m, const float *v, float *po, float *mo, float *vo, int64_t n, float lr, float b1, float b2, float eps, float wd, float bc1, float bc2) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i < n) {
        float gi = g[i];
        float mi = b1 * m[i] + (1.0f - b1) * gi;
        float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        mo[i] = mi; vo[i] = vi;
        po[i] = p[i] * (1.0f - lr * wd) - lr * ((mi / bc1) / (sqrtf(vi / bc2) + eps));
    }
}

__global__ void k_add_row_bias(const float *x, const float *b, float *o, int32_t rows, int32_t cols) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)rows * cols) o[idx] = x[idx] + b[idx % cols];
}

__global__ void k_sum_rows(const float *x, float *o, int32_t rows, int32_t cols) {
    int32_t c = blockIdx.x * 256 + threadIdx.x;
    if (c < cols) {
        float acc = 0.0f;
        for (int32_t r = 0; r < rows; r++) acc += x[(int64_t)r * cols + c];
        o[c] = acc;
    }
}

__global__ void k_rms_norm_fwd(const float *x, const float *w, float *o, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x;
    if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim;
    float *orow = o + (int64_t)r * dim;
    float ss = 0.0f;
    for (int32_t c = 0; c < dim; c++) ss += xr[c] * xr[c];
    float inv = rsqrtf(ss / (float)dim + eps);
    for (int32_t c = threadIdx.x; c < dim; c += blockDim.x) orow[c] = xr[c] * inv * w[c];
}

__global__ void k_rms_norm_bwd(const float *x, const float *w, const float *gy, float *gx, float *gw, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x;
    if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim;
    const float *gyr = gy + (int64_t)r * dim;
    float *gxr = gx + (int64_t)r * dim;
    float ss = 0.0f, dot = 0.0f;
    for (int32_t c = 0; c < dim; c++) { ss += xr[c] * xr[c]; dot += gyr[c] * w[c] * xr[c]; }
    float inv = rsqrtf(ss / (float)dim + eps);
    float coeff = dot * (inv * inv * inv) / (float)dim;
    for (int32_t c = threadIdx.x; c < dim; c += blockDim.x) {
        atomicAdd(&gw[c], gyr[c] * xr[c] * inv);
        gxr[c] = gyr[c] * w[c] * inv - xr[c] * coeff;
    }
}

__global__ void k_softmax_fwd(const float *x, float *o, int32_t rows, int32_t cols) {
    int32_t r = blockIdx.x;
    if (r >= rows) return;
    const float *xr = x + (int64_t)r * cols;
    float *orow = o + (int64_t)r * cols;
    float mx = xr[0];
    for (int32_t c = 1; c < cols; c++) if (xr[c] > mx) mx = xr[c];
    float sum = 0.0f;
    for (int32_t c = 0; c < cols; c++) { float e = expf(xr[c] - mx); orow[c] = e; sum += e; }
    float inv = 1.0f / sum;
    for (int32_t c = 0; c < cols; c++) orow[c] *= inv;
}

__global__ void k_softmax_bwd(const float *y, const float *gy, float *gx, int32_t rows, int32_t cols) {
    int32_t r = blockIdx.x;
    if (r >= rows) return;
    const float *yr = y + (int64_t)r * cols;
    const float *gyr = gy + (int64_t)r * cols;
    float *gxr = gx + (int64_t)r * cols;
    float dot = 0.0f;
    for (int32_t c = 0; c < cols; c++) dot += yr[c] * gyr[c];
    for (int32_t c = 0; c < cols; c++) gxr[c] = yr[c] * (gyr[c] - dot);
}

__global__ void k_ce_bwd(const float *probs, const int32_t *targets, float scale, float *gl, int32_t rows, int32_t vocab) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)rows * vocab) {
        int32_t r = (int32_t)(idx / vocab);
        int32_t c = (int32_t)(idx % vocab);
        gl[idx] = probs[idx] * scale - ((c == targets[r]) ? scale : 0.0f);
    }
}

__global__ void k_emb_fwd(const float *tab, const int32_t *ids, float *o, int32_t n_ids, int32_t dim) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)n_ids * dim) {
        int32_t i = (int32_t)(idx / dim), c = (int32_t)(idx % dim);
        o[idx] = tab[(int64_t)ids[i] * dim + c];
    }
}

__global__ void k_emb_bwd(const int32_t *ids, const float *gy, float *gtab, int32_t n_ids, int32_t vocab, int32_t dim) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)n_ids * dim) {
        int32_t i = (int32_t)(idx / dim), c = (int32_t)(idx % dim);
        int32_t id = ids[i];
        if (id >= 0 && id < vocab) atomicAdd(&gtab[(int64_t)id * dim + c], gy[idx]);
    }
}

__global__ void k_rope(const float *in, float *out, int32_t batch, int32_t seq, int32_t heads, int32_t hd, float theta, int32_t bwd) {
    int32_t half = hd / 2;
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    int64_t total = (int64_t)batch * seq * heads * half;
    if (idx >= total) return;
    int32_t i = (int32_t)(idx % half);
    int64_t rem = idx / half;
    int32_t h = (int32_t)(rem % heads); rem /= heads;
    int32_t t = (int32_t)(rem % seq);   rem /= seq;
    int32_t b = (int32_t)rem;
    int64_t base = (((int64_t)b * seq + t) * heads + h) * hd;
    float ang = (float)t / powf(theta, (float)(2 * i) / (float)hd);
    float cs = cosf(ang), sn = sinf(ang);
    float v0 = in[base + i], v1 = in[base + half + i];
    if (!bwd) { out[base + i] = v0 * cs - v1 * sn; out[base + half + i] = v0 * sn + v1 * cs; }
    else      { out[base + i] = v0 * cs + v1 * sn; out[base + half + i] = -v0 * sn + v1 * cs; }
}

__global__ void k_t2d(const float *x, float *o, int32_t rows, int32_t cols) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)rows * cols) { int32_t r = (int32_t)(idx / cols), c = (int32_t)(idx % cols); o[(int64_t)c * rows + r] = x[idx]; }
}

__global__ void k_t0213(const float *x, float *o, int32_t d0, int32_t d1, int32_t d2, int32_t d3) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    int64_t tot = (int64_t)d0 * d1 * d2 * d3;
    if (idx >= tot) return;
    int32_t i3 = (int32_t)(idx % d3); int64_t rem = idx / d3;
    int32_t i2 = (int32_t)(rem % d2); rem /= d2;
    int32_t i1 = (int32_t)(rem % d1); rem /= d1;
    int32_t i0 = (int32_t)rem;
    o[(((int64_t)i0 * d2 + i2) * d1 + i1) * d3 + i3] = x[idx];
}

__global__ void k_causal(float *s, int32_t bh, int32_t seq) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)bh * seq * seq) {
        int32_t j = (int32_t)(idx % seq);
        int32_t i = (int32_t)((idx / seq) % seq);
        if (j > i) s[idx] += -1.0e9f;
    }
}

static void cu_fill(int32_t d, float *dst, int64_t n, float v) { cudaSetDevice(d); k_fill<<<grid1d(n), 256>>>(dst, n, v); }
static void cu_copy(int32_t d, float *dst, const float *src, int64_t n) { cu_d2d(d, dst, src, n * (int64_t)sizeof(float)); }
static void cu_add(int32_t d, const float *a, const float *b, float *o, int64_t n) { cudaSetDevice(d); k_add<<<grid1d(n), 256>>>(a, b, o, n); }
static void cu_sub(int32_t d, const float *a, const float *b, float *o, int64_t n) { cudaSetDevice(d); k_sub<<<grid1d(n), 256>>>(a, b, o, n); }
static void cu_mul(int32_t d, const float *a, const float *b, float *o, int64_t n) { cudaSetDevice(d); k_mul<<<grid1d(n), 256>>>(a, b, o, n); }
static void cu_scale(int32_t d, const float *a, float s, float *o, int64_t n) { cudaSetDevice(d); k_scale<<<grid1d(n), 256>>>(a, s, o, n); }
static void cu_axpy(int32_t d, float a, const float *x, float *y, int64_t n) { cudaSetDevice(d); k_axpy<<<grid1d(n), 256>>>(a, x, y, n); }
static void cu_add_row_bias(int32_t d, const float *x, const float *b, float *o, int32_t r, int32_t c) { cudaSetDevice(d); k_add_row_bias<<<grid1d((int64_t)r*c), 256>>>(x, b, o, r, c); }
static void cu_sum_rows(int32_t d, const float *x, float *o, int32_t r, int32_t c) { cudaSetDevice(d); k_sum_rows<<<grid1d(c), 256>>>(x, o, r, c); }
static void cu_silu_fwd(int32_t d, const float *x, float *o, int64_t n) { cudaSetDevice(d); k_silu_fwd<<<grid1d(n), 256>>>(x, o, n); }
static void cu_silu_bwd(int32_t d, const float *x, const float *gy, float *gx, int64_t n) { cudaSetDevice(d); k_silu_bwd<<<grid1d(n), 256>>>(x, gy, gx, n); }
static void cu_gelu_fwd(int32_t d, const float *x, float *o, int64_t n) { cudaSetDevice(d); k_gelu_fwd<<<grid1d(n), 256>>>(x, o, n); }
static void cu_gelu_bwd(int32_t d, const float *x, const float *gy, float *gx, int64_t n) { cudaSetDevice(d); k_gelu_bwd<<<grid1d(n), 256>>>(x, gy, gx, n); }
static void cu_relu_fwd(int32_t d, const float *x, float *o, int64_t n) { cudaSetDevice(d); k_relu_fwd<<<grid1d(n), 256>>>(x, o, n); }
static void cu_relu_bwd(int32_t d, const float *x, const float *gy, float *gx, int64_t n) { cudaSetDevice(d); k_relu_bwd<<<grid1d(n), 256>>>(x, gy, gx, n); }

static float cu_sum_all(int32_t d, const float *x, int64_t n) {
    float *h = (float *)malloc((size_t)n * sizeof(float));
    cu_d2h(d, h, x, n * (int64_t)sizeof(float));
    float res = gs_backend_cpu.sum_all(0, h, n);
    free(h);
    return res;
}
static float cu_sum_sq(int32_t d, const float *x, int64_t n) {
    float *h = (float *)malloc((size_t)n * sizeof(float));
    cu_d2h(d, h, x, n * (int64_t)sizeof(float));
    float res = gs_backend_cpu.sum_sq(0, h, n);
    free(h);
    return res;
}
static void cu_rms_fwd(int32_t d, const float *x, const float *w, float *o, int32_t r, int32_t dim, float eps) { cudaSetDevice(d); k_rms_norm_fwd<<<r, 256>>>(x, w, o, r, dim, eps); }
static void cu_rms_bwd(int32_t d, const float *x, const float *w, const float *gy, float *gx, float *gw, int32_t r, int32_t dim, float eps) { cudaSetDevice(d); cudaMemset(gw, 0, (size_t)dim * sizeof(float)); k_rms_norm_bwd<<<r, 256>>>(x, w, gy, gx, gw, r, dim, eps); }
static void cu_sm_fwd(int32_t d, const float *x, float *o, int32_t r, int32_t c) { cudaSetDevice(d); k_softmax_fwd<<<r, 1>>>(x, o, r, c); }
static void cu_sm_bwd(int32_t d, const float *y, const float *gy, float *gx, int32_t r, int32_t c) { cudaSetDevice(d); k_softmax_bwd<<<r, 1>>>(y, gy, gx, r, c); }
static float cu_ce_fwd(int32_t d, const float *logits, const int32_t *targets, float *probs_out, int32_t rows, int32_t vocab) {
    cu_sm_fwd(d, logits, probs_out, rows, vocab);
    float *hp = (float *)malloc((size_t)((int64_t)rows * vocab) * sizeof(float));
    int32_t *ht = (int32_t *)malloc((size_t)rows * sizeof(int32_t));
    cu_d2h(d, hp, probs_out, (int64_t)rows * vocab * (int64_t)sizeof(float));
    cu_d2h(d, ht, targets, (int64_t)rows * (int64_t)sizeof(int32_t));
    double loss = 0.0;
    for (int32_t r = 0; r < rows; r++) { int32_t t = ht[r]; if (t >= 0 && t < vocab) loss -= log((double)hp[(int64_t)r * vocab + t] + 1e-12); }
    free(hp); free(ht);
    return rows > 0 ? (float)(loss / (double)rows) : 0.0f;
}
static void cu_ce_bwd(int32_t d, const float *probs, const int32_t *targets, float g_loss, float *gl, int32_t r, int32_t v) { cudaSetDevice(d); k_ce_bwd<<<grid1d((int64_t)r*v), 256>>>(probs, targets, g_loss / (float)r, gl, r, v); }
static void cu_emb_fwd(int32_t d, const float *tab, const int32_t *ids, float *o, int32_t n, int32_t dim) { cudaSetDevice(d); k_emb_fwd<<<grid1d((int64_t)n*dim), 256>>>(tab, ids, o, n, dim); }
static void cu_emb_bwd(int32_t d, const int32_t *ids, const float *gy, float *gtab, int32_t n, int32_t v, int32_t dim) { cudaSetDevice(d); cudaMemset(gtab, 0, (size_t)((int64_t)v*dim)*sizeof(float)); k_emb_bwd<<<grid1d((int64_t)n*dim), 256>>>(ids, gy, gtab, n, v, dim); }
static void cu_rope_fwd(int32_t d, const float *x, float *o, int32_t b, int32_t s, int32_t h, int32_t hd, float th) { cudaSetDevice(d); k_rope<<<grid1d((int64_t)b*s*h*(hd/2)), 256>>>(x, o, b, s, h, hd, th, 0); }
static void cu_rope_bwd(int32_t d, const float *gy, float *gx, int32_t b, int32_t s, int32_t h, int32_t hd, float th) { cudaSetDevice(d); k_rope<<<grid1d((int64_t)b*s*h*(hd/2)), 256>>>(gy, gx, b, s, h, hd, th, 1); }
static void cu_t2d(int32_t d, const float *x, float *o, int32_t r, int32_t c) { cudaSetDevice(d); k_t2d<<<grid1d((int64_t)r*c), 256>>>(x, o, r, c); }
static void cu_t0213(int32_t d, const float *x, float *o, int32_t d0, int32_t d1, int32_t d2, int32_t d3) { cudaSetDevice(d); k_t0213<<<grid1d((int64_t)d0*d1*d2*d3), 256>>>(x, o, d0, d1, d2, d3); }
static void cu_causal(int32_t d, float *s, int32_t bh, int32_t seq) { cudaSetDevice(d); k_causal<<<grid1d((int64_t)bh*seq*seq), 256>>>(s, bh, seq); }
static void cu_randn(int32_t d, float *dst, int64_t n, int64_t seed, float std) {
    float *h = (float *)malloc((size_t)n * sizeof(float));
    gs_backend_cpu.randn(0, h, n, seed, std);
    cu_h2d(d, dst, h, n * (int64_t)sizeof(float));
    free(h);
}
static void cu_adamw(int32_t d, const float *p, const float *g, const float *m, const float *v, float *po, float *mo, float *vo, int64_t n, float lr, float b1, float b2, float eps, float wd, float bc1, float bc2) {
    cudaSetDevice(d);
    k_adamw<<<grid1d(n), 256>>>(p, g, m, v, po, mo, vo, n, lr, b1, b2, eps, wd, bc1, bc2);
}

const GsTensorBackend gs_backend_cuda = {
    .name = "cuda+cublas",
    .dev_kind = GS_DEV_CUDA,
    .device_count = cu_device_count,
    .alloc = cu_alloc,
    .free_ptr = cu_free,
    .copy_h2d = cu_h2d,
    .copy_d2h = cu_d2h,
    .copy_d2d = cu_d2d,
    .sgemm = cu_sgemm,
    .bgemm = cu_bgemm,
    .fill = cu_fill,
    .copy = cu_copy,
    .add = cu_add,
    .sub = cu_sub,
    .mul = cu_mul,
    .scale = cu_scale,
    .axpy = cu_axpy,
    .add_row_bias = cu_add_row_bias,
    .sum_rows = cu_sum_rows,
    .silu_fwd = cu_silu_fwd,
    .silu_bwd = cu_silu_bwd,
    .gelu_fwd = cu_gelu_fwd,
    .gelu_bwd = cu_gelu_bwd,
    .relu_fwd = cu_relu_fwd,
    .relu_bwd = cu_relu_bwd,
    .sum_all = cu_sum_all,
    .sum_sq = cu_sum_sq,
    .rms_norm_fwd = cu_rms_fwd,
    .rms_norm_bwd = cu_rms_bwd,
    .softmax_fwd = cu_sm_fwd,
    .softmax_bwd = cu_sm_bwd,
    .cross_entropy_fwd = cu_ce_fwd,
    .cross_entropy_bwd = cu_ce_bwd,
    .embedding_fwd = cu_emb_fwd,
    .embedding_bwd = cu_emb_bwd,
    .rope_fwd = cu_rope_fwd,
    .rope_bwd = cu_rope_bwd,
    .transpose_2d = cu_t2d,
    .transpose_0213 = cu_t0213,
    .causal_mask_add = cu_causal,
    .randn = cu_randn,
    .adamw_step = cu_adamw,
};

#endif // GS_TENSOR_CUDA
