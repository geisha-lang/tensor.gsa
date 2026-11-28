// CUDA + cuBLAS backend for tensor.gsa (`[[target.native.when]] tool = "nvcc"`).
// Exports `extern "C" gs_cu_*` symbols overriding weak stubs in `cuda_weak_stubs.c`.

#ifdef GS_TENSOR_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define GS_BOP_ADD 0
#define GS_BOP_SUB 1
#define GS_BOP_MUL 2
#define GS_BOP_DIV 3

#define GS_UOP_EXP     0
#define GS_UOP_LOG     1
#define GS_UOP_SQRT    2
#define GS_UOP_RSQRT   3
#define GS_UOP_ABS     4
#define GS_UOP_SQUARE  5
#define GS_UOP_TANH    6
#define GS_UOP_SIGMOID 7
#define GS_UOP_ROUND   8
#define GS_UOP_SILU    9
#define GS_UOP_GELU    10
#define GS_UOP_RELU    11

static cublasHandle_t g_cublas[16] = {NULL};
static int32_t g_cuda_bf16_gemm = 0;
static int32_t g_pool_inited[16] = {0};

static void ensure_mempool(int32_t dev_id) {
    if (dev_id >= 0 && dev_id < 16 && !g_pool_inited[dev_id]) {
        g_pool_inited[dev_id] = 1;
        cudaMemPool_t pool = NULL;
        if (cudaDeviceGetDefaultMemPool(&pool, dev_id) == cudaSuccess && pool != NULL) {
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
        }
    }
}

static cublasHandle_t get_cublas(int32_t dev_id) {
    int32_t slot = (dev_id >= 0 && dev_id < 16) ? dev_id : 0;
    cudaSetDevice(slot);
    if (!g_cublas[slot]) {
        cublasCreate(&g_cublas[slot]);
        cublasSetMathMode(g_cublas[slot], CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return g_cublas[slot];
}

static inline int grid1d(int64_t n) { return (int)((n + 255) / 256); }

// ============================================================================
// Device Memory (Async Pool), Sync & BLAS (TF32 + Optional BF16 Tensor Core)
// ============================================================================

static void cu_free_impl(void *ptr) {
    if (ptr) {
        if (cudaFreeAsync(ptr, 0) != cudaSuccess) cudaFree(ptr);
    }
}

extern "C" {

int32_t gs_cu_device_count(void) {
    int c = 0;
    return cudaGetDeviceCount(&c) == cudaSuccess ? (int32_t)c : 0;
}

void gs_cu_set_bf16(int32_t enabled) {
    g_cuda_bf16_gemm = enabled ? 1 : 0;
}

void *gs_cu_free_fn(void) {
    return (void *)cu_free_impl;
}

void *gs_cu_alloc(int32_t dev_id, int64_t nbytes) {
    cudaSetDevice(dev_id);
    ensure_mempool(dev_id);
    void *p = NULL;
    size_t sz = nbytes > 0 ? (size_t)nbytes : 64;
    if (cudaMallocAsync(&p, sz, 0) != cudaSuccess) {
        if (cudaMalloc(&p, sz) != cudaSuccess) return NULL;
    }
    return p;
}

void gs_cu_zero(int32_t dev_id, void *ptr, int64_t nbytes) {
    cudaSetDevice(dev_id);
    if (ptr && nbytes > 0) cudaMemsetAsync(ptr, 0, (size_t)nbytes, 0);
}

void gs_cu_free(void *ptr) { cu_free_impl(ptr); }

void gs_cu_sync(int32_t dev_id) {
    cudaSetDevice(dev_id);
    cudaDeviceSynchronize();
}

void gs_cu_h2d(int32_t d, void *dst, const void *src, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) cudaMemcpy(dst, src, (size_t)n, cudaMemcpyHostToDevice);
}

void gs_cu_d2h(int32_t d, void *dst, const void *src, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) cudaMemcpy(dst, src, (size_t)n, cudaMemcpyDeviceToHost);
}

void gs_cu_d2d(int32_t d, void *dst, const void *src, int64_t n) {
    cudaSetDevice(d);
    if (n > 0 && dst != src) cudaMemcpyAsync(dst, src, (size_t)n, cudaMemcpyDeviceToDevice, 0);
}

void gs_cu_sgemm(
    int32_t d, int32_t ta, int32_t tb,
    int32_t M, int32_t N, int32_t K,
    float alpha, const float *A, const float *B, float beta, float *C
) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    cublasHandle_t h = get_cublas(d);
    cublasOperation_t opB = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opA = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldb = tb ? K : N;
    int lda = ta ? M : K;
    if (g_cuda_bf16_gemm) {
        cublasGemmEx(
            h, opB, opA, N, M, K, &alpha,
            B, CUDA_R_32F, ldb,
            A, CUDA_R_32F, lda,
            &beta, C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    } else {
        cublasSgemm(h, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
    }
}

void gs_cu_bgemm(
    int32_t d, int32_t batch, int32_t ta, int32_t tb,
    int32_t M, int32_t N, int32_t K,
    float alpha, const float *A, const float *B, float beta, float *C
) {
    if (batch <= 0 || M <= 0 || N <= 0 || K <= 0) return;
    cublasHandle_t h = get_cublas(d);
    cublasOperation_t opB = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opA = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldb = tb ? K : N;
    int lda = ta ? M : K;
    if (g_cuda_bf16_gemm) {
        cublasGemmStridedBatchedEx(
            h, opB, opA, N, M, K, &alpha,
            B, CUDA_R_32F, ldb, (long long)K * N,
            A, CUDA_R_32F, lda, (long long)M * K,
            &beta, C, CUDA_R_32F, N, (long long)M * N, batch,
            CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    } else {
        cublasSgemmStridedBatched(
            h, opB, opA, N, M, K, &alpha,
            B, ldb, (long long)K * N,
            A, lda, (long long)M * K,
            &beta, C, N, (long long)M * N, batch
        );
    }
}

} // extern "C"

// ============================================================================
// Elementwise, Broadcast & Scalar Kernels
// ============================================================================

__global__ void k_fill(float *d, int64_t n, float v) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i < n) d[i] = v;
}
__global__ void k_binop_same(int32_t op, const float *a, const float *b, float *o, int64_t n) {
    int64_t i4 = blockIdx.x * 256LL + threadIdx.x;
    int64_t n4 = n >> 2;
    if (i4 < n4) {
        float4 va = ((const float4 *)a)[i4];
        float4 vb = ((const float4 *)b)[i4];
        float4 vo;
        if (op == 0)      { vo.x = va.x + vb.x; vo.y = va.y + vb.y; vo.z = va.z + vb.z; vo.w = va.w + vb.w; }
        else if (op == 1) { vo.x = va.x - vb.x; vo.y = va.y - vb.y; vo.z = va.z - vb.z; vo.w = va.w - vb.w; }
        else if (op == 2) { vo.x = va.x * vb.x; vo.y = va.y * vb.y; vo.z = va.z * vb.z; vo.w = va.w * vb.w; }
        else              { vo.x = va.x / vb.x; vo.y = va.y / vb.y; vo.z = va.z / vb.z; vo.w = va.w / vb.w; }
        ((float4 *)o)[i4] = vo;
    } else if (i4 == n4) {
        for (int64_t i = n4 << 2; i < n; i++) {
            float va = a[i], vb = b[i];
            o[i] = (op == 0) ? (va + vb) : (op == 1) ? (va - vb) : (op == 2) ? (va * vb) : (va / vb);
        }
    }
}
__global__ void k_scale(const float *a, float s, float *o, int64_t n) {
    int64_t i4 = blockIdx.x * 256LL + threadIdx.x;
    int64_t n4 = n >> 2;
    if (i4 < n4) {
        float4 va = ((const float4 *)a)[i4];
        ((float4 *)o)[i4] = make_float4(va.x * s, va.y * s, va.z * s, va.w * s);
    } else if (i4 == n4) {
        for (int64_t i = n4 << 2; i < n; i++) o[i] = a[i] * s;
    }
}
__global__ void k_add_s(const float *a, float s, float *o, int64_t n) {
    int64_t i4 = blockIdx.x * 256LL + threadIdx.x;
    int64_t n4 = n >> 2;
    if (i4 < n4) {
        float4 va = ((const float4 *)a)[i4];
        ((float4 *)o)[i4] = make_float4(va.x + s, va.y + s, va.z + s, va.w + s);
    } else if (i4 == n4) {
        for (int64_t i = n4 << 2; i < n; i++) o[i] = a[i] + s;
    }
}
__global__ void k_pow_f(const float *a, float p, float *o, int64_t n) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i < n) o[i] = powf(a[i], p);
}
__global__ void k_pow_b(const float *a, const float *gy, float p, float *gx, int64_t n) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i < n) gx[i] = gy[i] * p * powf(a[i], p - 1.0f);
}
__global__ void k_clamp_f(const float *a, float lo, float hi, float *o, int64_t n) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i < n) { float v = a[i]; o[i] = v < lo ? lo : (v > hi ? hi : v); }
}
__global__ void k_clamp_b(const float *a, const float *gy, float lo, float hi, float *gx, int64_t n) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i < n) { float v = a[i]; gx[i] = (v >= lo && v <= hi) ? gy[i] : 0.0f; }
}
__global__ void k_lrelu_f(const float *a, float s, float *o, int64_t n) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i < n) o[i] = a[i] > 0.0f ? a[i] : s * a[i];
}
__global__ void k_lrelu_b(const float *a, const float *gy, float s, float *gx, int64_t n) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i < n) gx[i] = a[i] > 0.0f ? gy[i] : s * gy[i];
}
__global__ void k_axpy(float a, const float *x, float *y, int64_t n) {
    int64_t i4 = blockIdx.x * 256LL + threadIdx.x;
    int64_t n4 = n >> 2;
    if (i4 < n4) {
        float4 vx = ((const float4 *)x)[i4];
        float4 vy = ((float4 *)y)[i4];
        ((float4 *)y)[i4] = make_float4(vy.x + a * vx.x, vy.y + a * vx.y, vy.z + a * vx.z, vy.w + a * vx.w);
    } else if (i4 == n4) {
        for (int64_t i = n4 << 2; i < n; i++) y[i] += a * x[i];
    }
}

typedef struct { int32_t sh[16], sa[16], sb[16]; } BcastMeta;

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
__global__ void k_max_axis_bwd(const float *x, const float *gy, float *gx, int32_t outer, int32_t dim, int32_t inner) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= (int64_t)outer * inner) return;
    int32_t o0 = (int32_t)(idx / inner), k = (int32_t)(idx % inner);
    int32_t best_d = 0;
    float mx = x[((int64_t)o0 * dim) * inner + k];
    for (int32_t d = 1; d < dim; d++) {
        float v = x[((int64_t)o0 * dim + d) * inner + k];
        if (v > mx) { mx = v; best_d = d; }
    }
    gx[((int64_t)o0 * dim + best_d) * inner + k] = gy[idx];
}

// ============================================================================
// Block-Parallel Reductions (sum_all, sum_sq, RMSNorm, LayerNorm, L2, Softmax, LSE, CE)
// ============================================================================

__device__ inline float block_reduce_sum(float v) {
    __shared__ float sh[8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float res = (tid < 8) ? sh[tid] : 0.0f;
    if (warp == 0) {
        for (int off = 4; off > 0; off >>= 1) res += __shfl_down_sync(0xffffffffu, res, off);
        if (tid == 0) sh[0] = res;
    }
    __syncthreads();
    res = sh[0];
    __syncthreads();
    return res;
}

__device__ inline float block_reduce_max(float v) {
    __shared__ float sh[8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_down_sync(0xffffffffu, v, off);
        if (o > v) v = o;
    }
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float res = (tid < 8) ? sh[tid] : -1.0e38f;
    if (warp == 0) {
        for (int off = 4; off > 0; off >>= 1) {
            float o = __shfl_down_sync(0xffffffffu, res, off);
            if (o > res) res = o;
        }
        if (tid == 0) sh[0] = res;
    }
    __syncthreads();
    res = sh[0];
    __syncthreads();
    return res;
}

__global__ void k_reduce_sum_all(const float *x, float *out, int64_t n, int32_t sq) {
    float acc = 0.0f;
    for (int64_t i = blockIdx.x * 256LL + threadIdx.x; i < n; i += (int64_t)gridDim.x * 256LL) {
        float v = x[i];
        acc += sq ? (v * v) : v;
    }
    float bs = block_reduce_sum(acc);
    if (threadIdx.x == 0) atomicAdd(out, bs);
}

__global__ void k_rms_f(const float *x, const float *w, float *o, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim; float *orow = o + (int64_t)r * dim;
    float ss = 0.0f;
    for (int32_t c = threadIdx.x; c < dim; c += 256) ss += xr[c] * xr[c];
    float inv = rsqrtf(block_reduce_sum(ss) / (float)dim + eps);
    for (int32_t c = threadIdx.x; c < dim; c += 256) orow[c] = xr[c] * inv * w[c];
}

__global__ void k_rms_b(const float *x, const float *w, const float *gy, float *gx, float *gw, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim, *gyr = gy + (int64_t)r * dim; float *gxr = gx + (int64_t)r * dim;
    float ss = 0.0f, dot = 0.0f;
    for (int32_t c = threadIdx.x; c < dim; c += 256) {
        ss += xr[c] * xr[c];
        dot += gyr[c] * w[c] * xr[c];
    }
    float inv = rsqrtf(block_reduce_sum(ss) / (float)dim + eps);
    float coeff = block_reduce_sum(dot) * (inv * inv * inv) / (float)dim;
    for (int32_t c = threadIdx.x; c < dim; c += 256) {
        atomicAdd(&gw[c], gyr[c] * xr[c] * inv);
        gxr[c] = gyr[c] * w[c] * inv - xr[c] * coeff;
    }
}

__global__ void k_ln_f(const float *x, const float *w, const float *b, float *o, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim; float *orow = o + (int64_t)r * dim;
    float s = 0.0f;
    for (int32_t c = threadIdx.x; c < dim; c += 256) s += xr[c];
    float mean = block_reduce_sum(s) / (float)dim;
    float v = 0.0f;
    for (int32_t c = threadIdx.x; c < dim; c += 256) { float d = xr[c] - mean; v += d * d; }
    float inv = rsqrtf(block_reduce_sum(v) / (float)dim + eps);
    for (int32_t c = threadIdx.x; c < dim; c += 256) orow[c] = (xr[c] - mean) * inv * w[c] + (b ? b[c] : 0.0f);
}

__global__ void k_ln_b(const float *x, const float *w, const float *gy, float *gx, float *gw, float *gb, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim, *gyr = gy + (int64_t)r * dim; float *gxr = gx + (int64_t)r * dim;
    float s = 0.0f;
    for (int32_t c = threadIdx.x; c < dim; c += 256) s += xr[c];
    float mean = block_reduce_sum(s) / (float)dim;
    float v = 0.0f;
    for (int32_t c = threadIdx.x; c < dim; c += 256) { float d = xr[c] - mean; v += d * d; }
    float inv = rsqrtf(block_reduce_sum(v) / (float)dim + eps);
    float l1 = 0.0f, l2 = 0.0f;
    for (int32_t c = threadIdx.x; c < dim; c += 256) {
        float xh = (xr[c] - mean) * inv;
        float dx = gyr[c] * w[c];
        l1 += dx; l2 += dx * xh;
    }
    float s1 = block_reduce_sum(l1);
    float s2 = block_reduce_sum(l2);
    for (int32_t c = threadIdx.x; c < dim; c += 256) {
        float xh = (xr[c] - mean) * inv;
        atomicAdd(&gw[c], gyr[c] * xh);
        if (gb) atomicAdd(&gb[c], gyr[c]);
        gxr[c] = inv * (gyr[c] * w[c] - (s1 + xh * s2) / (float)dim);
    }
}

__global__ void k_l2_f(const float *x, float *o, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim; float *orow = o + (int64_t)r * dim;
    float ss = 0.0f;
    for (int32_t c = threadIdx.x; c < dim; c += 256) ss += xr[c] * xr[c];
    float norm = sqrtf(block_reduce_sum(ss));
    float inv = 1.0f / (norm > eps ? norm : eps);
    for (int32_t c = threadIdx.x; c < dim; c += 256) orow[c] = xr[c] * inv;
}

__global__ void k_l2_b(const float *x, const float *gy, float *gx, int32_t rows, int32_t dim, float eps) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * dim, *gyr = gy + (int64_t)r * dim; float *gxr = gx + (int64_t)r * dim;
    float ss = 0.0f, dot = 0.0f;
    for (int32_t c = threadIdx.x; c < dim; c += 256) { ss += xr[c] * xr[c]; dot += xr[c] * gyr[c]; }
    float norm = sqrtf(block_reduce_sum(ss));
    float tot_dot = block_reduce_sum(dot);
    if (norm <= eps) {
        float inv = 1.0f / eps;
        for (int32_t c = threadIdx.x; c < dim; c += 256) gxr[c] = gyr[c] * inv;
    } else {
        float inv = 1.0f / norm, coeff = tot_dot * inv * inv * inv;
        for (int32_t c = threadIdx.x; c < dim; c += 256) gxr[c] = gyr[c] * inv - xr[c] * coeff;
    }
}

__global__ void k_sm_f(const float *x, float *o, int32_t rows, int32_t cols) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * cols; float *orow = o + (int64_t)r * cols;
    float lmx = -1.0e38f;
    for (int32_t c = threadIdx.x; c < cols; c += 256) if (xr[c] > lmx) lmx = xr[c];
    float mx = block_reduce_max(lmx);
    float lsum = 0.0f;
    for (int32_t c = threadIdx.x; c < cols; c += 256) { float e = expf(xr[c] - mx); orow[c] = e; lsum += e; }
    float inv = 1.0f / block_reduce_sum(lsum);
    for (int32_t c = threadIdx.x; c < cols; c += 256) orow[c] *= inv;
}

__global__ void k_sm_b(const float *y, const float *gy, float *gx, int32_t rows, int32_t cols) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *yr = y + (int64_t)r * cols, *gyr = gy + (int64_t)r * cols; float *gxr = gx + (int64_t)r * cols;
    float ldot = 0.0f;
    for (int32_t c = threadIdx.x; c < cols; c += 256) ldot += yr[c] * gyr[c];
    float dot = block_reduce_sum(ldot);
    for (int32_t c = threadIdx.x; c < cols; c += 256) gxr[c] = yr[c] * (gyr[c] - dot);
}

__global__ void k_lse_f(const float *x, float *o, int32_t rows, int32_t cols) {
    int32_t r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (int64_t)r * cols;
    float lmx = -1.0e38f;
    for (int32_t c = threadIdx.x; c < cols; c += 256) if (xr[c] > lmx) lmx = xr[c];
    float mx = block_reduce_max(lmx);
    float lsum = 0.0f;
    for (int32_t c = threadIdx.x; c < cols; c += 256) lsum += expf(xr[c] - mx);
    float sum = block_reduce_sum(lsum);
    if (threadIdx.x == 0) o[r] = mx + logf(sum);
}

__global__ void k_lse_b(const float *x, const float *lse, const float *gy, float *gx, int32_t rows, int32_t cols) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= (int64_t)rows * cols) return;
    int32_t r = (int32_t)(idx / cols);
    gx[idx] = gy[r] * expf(x[idx] - lse[r]);
}

__global__ void k_ce_fused_fwd(
    const float *logits, const int32_t *targets,
    float *lse_out, float *loss_sum, int32_t *valid_cnt,
    int32_t rows, int32_t vocab, int32_t ignore_idx
) {
    int32_t r = blockIdx.x;
    if (r >= rows) return;
    int32_t t = targets[r];
    if (t == ignore_idx || t < 0 || t >= vocab) {
        if (threadIdx.x == 0) lse_out[r] = 0.0f;
        return;
    }
    const float *xr = logits + (int64_t)r * vocab;
    float lmx = -1.0e38f;
    for (int32_t c = threadIdx.x; c < vocab; c += 256) {
        if (xr[c] > lmx) lmx = xr[c];
    }
    float mx = block_reduce_max(lmx);
    float lsum = 0.0f;
    for (int32_t c = threadIdx.x; c < vocab; c += 256) {
        lsum += expf(xr[c] - mx);
    }
    float sum = block_reduce_sum(lsum);
    if (threadIdx.x == 0) {
        float lse = mx + logf(sum);
        lse_out[r] = lse;
        atomicAdd(loss_sum, lse - xr[t]);
        atomicAdd(valid_cnt, 1);
    }
}

__global__ void k_ce_fused_bwd(
    const float *logits, const float *lse, const int32_t *targets,
    float scale, float *gl, int32_t rows, int32_t vocab, int32_t ignore_idx
) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= (int64_t)rows * vocab) return;
    int32_t r = (int32_t)(idx / vocab), c = (int32_t)(idx % vocab), tgt = targets[r];
    if (tgt == ignore_idx || tgt < 0 || tgt >= vocab) {
        gl[idx] = 0.0f;
    } else {
        float p = expf(logits[idx] - lse[r]);
        gl[idx] = p * scale - ((c == tgt) ? scale : 0.0f);
    }
}

__global__ void k_emb_f(const float *tab, const int32_t *ids, float *o, int32_t n_ids, int32_t dim) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)n_ids * dim) o[idx] = tab[(int64_t)ids[idx / dim] * dim + (idx % dim)];
}
__global__ void k_emb_b(const int32_t *ids, const float *gy, float *gtab, int32_t n_ids, int32_t vocab, int32_t dim) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)n_ids * dim) {
        int32_t id = ids[idx / dim];
        if (id >= 0 && id < vocab) atomicAdd(&gtab[(int64_t)id * dim + (idx % dim)], gy[idx]);
    }
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
__global__ void k_t2d(const float *x, float *o, int32_t r, int32_t c) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i < (int64_t)r * c) o[(i % c) * r + (i / c)] = x[i];
}
__global__ void k_t0213(const float *x, float *o, int32_t d0, int32_t d1, int32_t d2, int32_t d3) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx >= (int64_t)d0 * d1 * d2 * d3) return;
    int32_t i3 = (int32_t)(idx % d3); int64_t rem = idx / d3;
    int32_t i2 = (int32_t)(rem % d2); rem /= d2;
    int32_t i1 = (int32_t)(rem % d1); int32_t i0 = (int32_t)(rem / d1);
    o[(((int64_t)i0 * d2 + i2) * d1 + i1) * d3 + i3] = x[idx];
}
__global__ void k_causal(float *s, int32_t bh, int32_t seq) {
    int64_t idx = blockIdx.x * 256LL + threadIdx.x;
    if (idx < (int64_t)bh * seq * seq) {
        if ((int32_t)(idx % seq) > (int32_t)((idx / seq) % seq)) s[idx] += -1.0e9f;
    }
}
__global__ void k_adamw(
    const float *p, const float *g, const float *m, const float *v,
    float *po, float *mo, float *vo, int64_t n,
    float lr, float b1, float b2, float eps, float wd, float bc1, float bc2
) {
    int64_t i = blockIdx.x * 256LL + threadIdx.x;
    if (i < n) {
        float gi = g[i];
        float mi = b1 * m[i] + (1.0f - b1) * gi;
        float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        mo[i] = mi; vo[i] = vi;
        po[i] = p[i] * (1.0f - lr * wd) - lr * ((mi / bc1) / (sqrtf(vi / bc2) + eps));
    }
}

__global__ void k_adamw_multi(
    int32_t count,
    const float *const *ps, const float *const *gs,
    const float *const *ms, const float *const *vs,
    float *const *pos, float *const *mos, float *const *vos,
    const int64_t *numels,
    float lr, float b1, float b2, float eps, float wd, float bc1, float bc2
) {
    int32_t t = blockIdx.y;
    if (t >= count) return;
    int64_t n = numels[t];
    const float *p = ps[t], *g = gs[t], *m = ms[t], *v = vs[t];
    float *po = pos[t], *mo = mos[t], *vo = vos[t];
    for (int64_t i = blockIdx.x * 256LL + threadIdx.x; i < n; i += (int64_t)gridDim.x * 256LL) {
        float gi = g[i];
        float mi = b1 * m[i] + (1.0f - b1) * gi;
        float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        mo[i] = mi; vo[i] = vi;
        po[i] = p[i] * (1.0f - lr * wd) - lr * ((mi / bc1) / (sqrtf(vi / bc2) + eps));
    }
}

__global__ void k_sum_sq_multi(
    int32_t count,
    const float *const *gs,
    const int64_t *numels,
    float *out_sum_sq
) {
    int32_t t = blockIdx.y;
    if (t >= count) return;
    int64_t n = numels[t];
    const float *g = gs[t];
    float acc = 0.0f;
    int64_t n4 = n >> 2;
    for (int64_t i4 = blockIdx.x * 256LL + threadIdx.x; i4 < n4; i4 += (int64_t)gridDim.x * 256LL) {
        float4 v = ((const float4 *)g)[i4];
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int64_t i = n4 << 2; i < n; i++) {
            float v = g[i];
            acc += v * v;
        }
    }
    float bs = block_reduce_sum(acc);
    if (threadIdx.x == 0 && bs != 0.0f) atomicAdd(out_sum_sq, bs);
}

__global__ void k_clip_scale_multi(
    int32_t count,
    const float *const *gs,
    float *const *gos,
    const int64_t *numels,
    const float *sum_sq_ptr,
    float max_norm
) {
    int32_t t = blockIdx.y;
    if (t >= count) return;
    float gn = sqrtf(*sum_sq_ptr);
    float s = (gn > max_norm && gn > 0.0f) ? (max_norm / (gn + 1.0e-6f)) : 1.0f;
    int64_t n = numels[t];
    const float *g = gs[t];
    float *go = gos[t];
    int64_t n4 = n >> 2;
    for (int64_t i4 = blockIdx.x * 256LL + threadIdx.x; i4 < n4; i4 += (int64_t)gridDim.x * 256LL) {
        float4 v = ((const float4 *)g)[i4];
        ((float4 *)go)[i4] = make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int64_t i = n4 << 2; i < n; i++) go[i] = g[i] * s;
    }
}

// ============================================================================
// Exported C ABI Wrappers for `src/cuda.gsa`
// ============================================================================

extern "C" {

void gs_cu_fill(int32_t d, float *dst, int64_t n, float v) {
    cudaSetDevice(d);
    if (n > 0) k_fill<<<grid1d(n), 256>>>(dst, n, v);
}
void gs_cu_copy(int32_t d, float *dst, const float *src, int64_t n) {
    gs_cu_d2d(d, dst, src, n * (int64_t)sizeof(float));
}
void gs_cu_copy_at(int32_t d, float *dst, int64_t dst_off, const float *src, int64_t src_off, int64_t n) {
    if (n > 0) gs_cu_d2d(d, dst + dst_off, src + src_off, n * (int64_t)sizeof(float));
}
void gs_cu_binop_same(int32_t d, int32_t op, const float *a, const float *b, float *o, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_binop_same<<<grid1d((n >> 2) + 1), 256>>>(op, a, b, o, n);
}
void gs_cu_scale(int32_t d, const float *a, float s, float *o, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_scale<<<grid1d((n >> 2) + 1), 256>>>(a, s, o, n);
}
void gs_cu_add_s(int32_t d, const float *a, float s, float *o, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_add_s<<<grid1d((n >> 2) + 1), 256>>>(a, s, o, n);
}
void gs_cu_pow_f(int32_t d, const float *a, float p, float *o, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_pow_f<<<grid1d(n), 256>>>(a, p, o, n);
}
void gs_cu_pow_b(int32_t d, const float *a, const float *gy, float p, float *gx, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_pow_b<<<grid1d(n), 256>>>(a, gy, p, gx, n);
}
void gs_cu_clamp_f(int32_t d, const float *a, float lo, float hi, float *o, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_clamp_f<<<grid1d(n), 256>>>(a, lo, hi, o, n);
}
void gs_cu_clamp_b(int32_t d, const float *a, const float *gy, float lo, float hi, float *gx, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_clamp_b<<<grid1d(n), 256>>>(a, gy, lo, hi, gx, n);
}
void gs_cu_lrelu_f(int32_t d, const float *a, float s, float *o, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_lrelu_f<<<grid1d(n), 256>>>(a, s, o, n);
}
void gs_cu_lrelu_b(int32_t d, const float *a, const float *gy, float s, float *gx, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_lrelu_b<<<grid1d(n), 256>>>(a, gy, s, gx, n);
}
void gs_cu_axpy(int32_t d, float a, const float *x, float *y, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_axpy<<<grid1d((n >> 2) + 1), 256>>>(a, x, y, n);
}

void gs_cu_bcast_binop(int32_t d, int32_t op, int32_t ndim, int64_t total, const int32_t *sh, const int32_t *sa, const int32_t *sb, const float *a, const float *b, float *o) {
    if (total <= 0 || ndim <= 0 || ndim > 16) return;
    BcastMeta m = {{0}, {0}, {0}};
    for (int32_t i = 0; i < ndim; i++) { m.sh[i] = sh[i]; m.sa[i] = sa[i]; m.sb[i] = sb[i]; }
    cudaSetDevice(d);
    k_bcast_binop<<<grid1d(total), 256>>>(op, ndim, total, m, a, b, o);
}
void gs_cu_bcast_reduce_sum(int32_t d, int32_t ndim, int64_t src_total, int64_t dst_total, const int32_t *sh, const int32_t *sd, const float *src, float *dst) {
    if (ndim <= 0 || ndim > 16) return;
    BcastMeta m = {{0}, {0}, {0}};
    for (int32_t i = 0; i < ndim; i++) { m.sh[i] = sh[i]; m.sa[i] = sd[i]; }
    cudaSetDevice(d);
    if (dst_total > 0) cudaMemsetAsync(dst, 0, (size_t)dst_total * sizeof(float), 0);
    if (src_total > 0) k_bcast_reduce_sum<<<grid1d(src_total), 256>>>(ndim, src_total, m, src, dst);
}
void gs_cu_unary_f(int32_t d, int32_t op, const float *x, float *o, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_unary_fwd<<<grid1d(n), 256>>>(op, x, o, n);
}
void gs_cu_unary_b(int32_t d, int32_t op, const float *x, const float *y, const float *gy, float *gx, int64_t n) {
    cudaSetDevice(d);
    if (n > 0) k_unary_bwd<<<grid1d(n), 256>>>(op, x, y, gy, gx, n);
}
void gs_cu_slice_f(int32_t d, const float *x, float *o, int32_t out0, int32_t id, int32_t in0, int32_t st, int32_t len) {
    cudaSetDevice(d);
    int64_t tot = (int64_t)out0 * len * in0;
    if (tot > 0) k_slice_f<<<grid1d(tot), 256>>>(x, o, out0, id, in0, st, len);
}
void gs_cu_slice_b(int32_t d, const float *gy, float *gx, int32_t out0, int32_t id, int32_t in0, int32_t st, int32_t len) {
    cudaSetDevice(d);
    cudaMemsetAsync(gx, 0, (size_t)((int64_t)out0 * id * in0) * sizeof(float), 0);
    int64_t tot = (int64_t)out0 * len * in0;
    if (tot > 0) k_slice_b<<<grid1d(tot), 256>>>(gy, gx, out0, id, in0, st, len);
}
void gs_cu_sum_ax(int32_t d, const float *x, float *o, int32_t out0, int32_t dim, int32_t in0) {
    cudaSetDevice(d);
    int64_t tot = (int64_t)out0 * in0;
    if (tot > 0) k_sum_axis<<<grid1d(tot), 256>>>(x, o, out0, dim, in0);
}
void gs_cu_max_ax(int32_t d, const float *x, float *o, int32_t out0, int32_t dim, int32_t in0) {
    cudaSetDevice(d);
    int64_t tot = (int64_t)out0 * in0;
    if (tot > 0 && dim > 0) k_max_axis<<<grid1d(tot), 256>>>(x, o, out0, dim, in0);
}
void gs_cu_max_ax_b(int32_t d, const float *x, const float *gy, float *gx, int32_t out0, int32_t dim, int32_t in0) {
    cudaSetDevice(d);
    int64_t in_tot = (int64_t)out0 * dim * in0;
    if (in_tot > 0) cudaMemsetAsync(gx, 0, (size_t)in_tot * sizeof(float), 0);
    int64_t tot = (int64_t)out0 * in0;
    if (tot > 0 && dim > 0) k_max_axis_bwd<<<grid1d(tot), 256>>>(x, gy, gx, out0, dim, in0);
}

static float *g_reduce_buf = NULL;
static int32_t *g_valid_buf = NULL;

static void ensure_scratch(int32_t d) {
    cudaSetDevice(d);
    if (!g_reduce_buf) cudaMalloc(&g_reduce_buf, sizeof(float));
    if (!g_valid_buf) cudaMalloc(&g_valid_buf, sizeof(int32_t));
}

double gs_cu_sum_all(int32_t d, const float *x, int64_t n) {
    if (n <= 0) return 0.0;
    ensure_scratch(d);
    cudaMemsetAsync(g_reduce_buf, 0, sizeof(float), 0);
    int blocks = (int)((n + 255) / 256);
    if (blocks > 1024) blocks = 1024;
    k_reduce_sum_all<<<blocks, 256>>>(x, g_reduce_buf, n, 0);
    float res = 0.0f;
    cudaMemcpy(&res, g_reduce_buf, sizeof(float), cudaMemcpyDeviceToHost);
    return (double)res;
}

double gs_cu_sum_sq(int32_t d, const float *x, int64_t n) {
    if (n <= 0) return 0.0;
    ensure_scratch(d);
    cudaMemsetAsync(g_reduce_buf, 0, sizeof(float), 0);
    int blocks = (int)((n + 255) / 256);
    if (blocks > 1024) blocks = 1024;
    k_reduce_sum_all<<<blocks, 256>>>(x, g_reduce_buf, n, 1);
    float res = 0.0f;
    cudaMemcpy(&res, g_reduce_buf, sizeof(float), cudaMemcpyDeviceToHost);
    return (double)res;
}

void gs_cu_rms_f(int32_t d, const float *x, const float *w, float *o, int32_t r, int32_t dim, float eps) {
    cudaSetDevice(d);
    if (r > 0) k_rms_f<<<r, 256>>>(x, w, o, r, dim, eps);
}
void gs_cu_rms_b(int32_t d, const float *x, const float *w, const float *gy, float *gx, float *gw, int32_t r, int32_t dim, float eps) {
    cudaSetDevice(d);
    cudaMemsetAsync(gw, 0, (size_t)dim * sizeof(float), 0);
    if (r > 0) k_rms_b<<<r, 256>>>(x, w, gy, gx, gw, r, dim, eps);
}
void gs_cu_ln_f(int32_t d, const float *x, const float *w, const float *b, float *o, int32_t r, int32_t dim, float eps) {
    cudaSetDevice(d);
    if (r > 0) k_ln_f<<<r, 256>>>(x, w, b, o, r, dim, eps);
}
void gs_cu_ln_b(int32_t d, const float *x, const float *w, const float *gy, float *gx, float *gw, float *gb, int32_t r, int32_t dim, float eps) {
    cudaSetDevice(d);
    cudaMemsetAsync(gw, 0, (size_t)dim * sizeof(float), 0);
    if (gb) cudaMemsetAsync(gb, 0, (size_t)dim * sizeof(float), 0);
    if (r > 0) k_ln_b<<<r, 256>>>(x, w, gy, gx, gw, gb, r, dim, eps);
}
void gs_cu_l2_f(int32_t d, const float *x, float *o, int32_t r, int32_t dim, float eps) {
    cudaSetDevice(d);
    if (r > 0) k_l2_f<<<r, 256>>>(x, o, r, dim, eps);
}
void gs_cu_l2_b(int32_t d, const float *x, const float *gy, float *gx, int32_t r, int32_t dim, float eps) {
    cudaSetDevice(d);
    if (r > 0) k_l2_b<<<r, 256>>>(x, gy, gx, r, dim, eps);
}
void gs_cu_sm_f(int32_t d, const float *x, float *o, int32_t r, int32_t c) {
    cudaSetDevice(d);
    if (r > 0) k_sm_f<<<r, 256>>>(x, o, r, c);
}
void gs_cu_sm_b(int32_t d, const float *y, const float *gy, float *gx, int32_t r, int32_t c) {
    cudaSetDevice(d);
    if (r > 0) k_sm_b<<<r, 256>>>(y, gy, gx, r, c);
}
void gs_cu_lse_f(int32_t d, const float *x, float *o, int32_t r, int32_t c) {
    cudaSetDevice(d);
    if (r > 0) k_lse_f<<<r, 256>>>(x, o, r, c);
}
void gs_cu_lse_b(int32_t d, const float *x, const float *lse, const float *gy, float *gx, int32_t r, int32_t c) {
    cudaSetDevice(d);
    int64_t tot = (int64_t)r * c;
    if (tot > 0) k_lse_b<<<grid1d(tot), 256>>>(x, lse, gy, gx, r, c);
}

double gs_cu_ce_f(int32_t d, const float *logits, const int32_t *targets, float *lse_out, int32_t *valid_out, int32_t rows, int32_t vocab, int32_t ignore_idx) {
    if (rows <= 0) {
        if (valid_out) *valid_out = 0;
        return 0.0;
    }
    ensure_scratch(d);
    cudaMemsetAsync(g_reduce_buf, 0, sizeof(float), 0);
    cudaMemsetAsync(g_valid_buf, 0, sizeof(int32_t), 0);
    k_ce_fused_fwd<<<rows, 256>>>(logits, targets, lse_out, g_reduce_buf, g_valid_buf, rows, vocab, ignore_idx);
    float loss_sum = 0.0f;
    int32_t valid = 0;
    cudaMemcpy(&loss_sum, g_reduce_buf, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&valid, g_valid_buf, sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (valid_out) *valid_out = valid;
    return valid > 0 ? (double)loss_sum / (double)valid : 0.0;
}

void gs_cu_ce_b(int32_t d, const float *logits, const float *lse, const int32_t *targets, float g_loss, float *gl, int32_t rows, int32_t vocab, int32_t ignore_idx, int32_t valid) {
    if (rows <= 0) return;
    cudaSetDevice(d);
    if (valid <= 0) {
        cudaMemsetAsync(gl, 0, (size_t)((int64_t)rows * vocab) * sizeof(float), 0);
        return;
    }
    k_ce_fused_bwd<<<grid1d((int64_t)rows * vocab), 256>>>(logits, lse, targets, g_loss / (float)valid, gl, rows, vocab, ignore_idx);
}

void gs_cu_emb_f(int32_t d, const float *tab, const int32_t *ids, float *o, int32_t n, int32_t dim) {
    cudaSetDevice(d);
    int64_t tot = (int64_t)n * dim;
    if (tot > 0) k_emb_f<<<grid1d(tot), 256>>>(tab, ids, o, n, dim);
}
void gs_cu_emb_b(int32_t d, const int32_t *ids, const float *gy, float *gtab, int32_t n, int32_t v, int32_t dim) {
    cudaSetDevice(d);
    cudaMemsetAsync(gtab, 0, (size_t)((int64_t)v * dim) * sizeof(float), 0);
    int64_t tot = (int64_t)n * dim;
    if (tot > 0) k_emb_b<<<grid1d(tot), 256>>>(ids, gy, gtab, n, v, dim);
}
void gs_cu_rope_f(int32_t d, const float *x, float *o, int32_t b, int32_t s, int32_t h, int32_t hd, float th) {
    cudaSetDevice(d);
    int64_t tot = (int64_t)b * s * h * (hd / 2);
    if (tot > 0) k_rope<<<grid1d(tot), 256>>>(x, o, b, s, h, hd, th, 0);
}
void gs_cu_rope_b(int32_t d, const float *gy, float *gx, int32_t b, int32_t s, int32_t h, int32_t hd, float th) {
    cudaSetDevice(d);
    int64_t tot = (int64_t)b * s * h * (hd / 2);
    if (tot > 0) k_rope<<<grid1d(tot), 256>>>(gy, gx, b, s, h, hd, th, 1);
}
void gs_cu_t2d(int32_t d, const float *x, float *o, int32_t r, int32_t c) {
    cudaSetDevice(d);
    int64_t tot = (int64_t)r * c;
    if (tot > 0) k_t2d<<<grid1d(tot), 256>>>(x, o, r, c);
}
void gs_cu_t0213(int32_t d, const float *x, float *o, int32_t d0, int32_t d1, int32_t d2, int32_t d3) {
    cudaSetDevice(d);
    int64_t tot = (int64_t)d0 * d1 * d2 * d3;
    if (tot > 0) k_t0213<<<grid1d(tot), 256>>>(x, o, d0, d1, d2, d3);
}
void gs_cu_causal(int32_t d, float *s, int32_t bh, int32_t seq) {
    cudaSetDevice(d);
    int64_t tot = (int64_t)bh * seq * seq;
    if (tot > 0) k_causal<<<grid1d(tot), 256>>>(s, bh, seq);
}

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void gs_cu_randn(int32_t d, float *dst, int64_t n, int64_t seed, float std) {
    if (n <= 0) return;
    float *h = (float *)malloc((size_t)n * sizeof(float));
    if (!h) return;
    uint64_t st = (uint64_t)seed ^ 0x517cc1b727220a95ULL;
    for (int64_t i = 0; i < n; i += 2) {
        double u1 = ((double)(splitmix64(&st) >> 11) + 0.5) / 9007199254740992.0;
        double u2 = ((double)(splitmix64(&st) >> 11) + 0.5) / 9007199254740992.0;
        if (u1 < 1e-12) u1 = 1e-12;
        double mag = sqrt(-2.0 * log(u1)) * (double)std;
        double ang = 6.283185307179586 * u2;
        h[i] = (float)(mag * cos(ang));
        if (i + 1 < n) h[i + 1] = (float)(mag * sin(ang));
    }
    gs_cu_h2d(d, dst, h, n * (int64_t)sizeof(float));
    free(h);
}

void gs_cu_adamw(
    int32_t d, const float *p, const float *g, const float *m, const float *v,
    float *po, float *mo, float *vo, int64_t n,
    float lr, float b1, float b2, float eps, float wd, float bc1, float bc2
) {
    cudaSetDevice(d);
    if (n > 0) k_adamw<<<grid1d(n), 256>>>(p, g, m, v, po, mo, vo, n, lr, b1, b2, eps, wd, bc1, bc2);
}

static void *g_adamw_meta_buf = NULL;
static size_t g_adamw_meta_cap = 0;

typedef struct {
    const float *p, *g, *m, *v;
    float *po, *mo, *vo;
    int64_t n;
} AdamwItem;

static AdamwItem *g_adamw_items = NULL;
static int32_t g_adamw_count = 0;
static int32_t g_adamw_cap = 0;

void gs_cu_adamw_batch_reset(void) {
    g_adamw_count = 0;
}

void gs_cu_adamw_batch_push(
    const float *p, const float *g, const float *m, const float *v,
    float *po, float *mo, float *vo, int64_t n
) {
    if (n <= 0) return;
    if (g_adamw_count >= g_adamw_cap) {
        int32_t new_cap = g_adamw_cap > 0 ? g_adamw_cap * 2 : 128;
        AdamwItem *nb = (AdamwItem *)realloc(g_adamw_items, (size_t)new_cap * sizeof(AdamwItem));
        if (!nb) return;
        g_adamw_items = nb;
        g_adamw_cap = new_cap;
    }
    AdamwItem *it = &g_adamw_items[g_adamw_count++];
    it->p = p; it->g = g; it->m = m; it->v = v;
    it->po = po; it->mo = mo; it->vo = vo; it->n = n;
}

void gs_cu_adamw_batch_run(
    int32_t d, float lr, float b1, float b2, float eps, float wd, float bc1, float bc2
) {
    int32_t count = g_adamw_count;
    g_adamw_count = 0;
    if (count <= 0) return;
    cudaSetDevice(d);
    if (count == 1) {
        AdamwItem *it = &g_adamw_items[0];
        k_adamw<<<grid1d(it->n), 256>>>(it->p, it->g, it->m, it->v, it->po, it->mo, it->vo, it->n, lr, b1, b2, eps, wd, bc1, bc2);
        return;
    }
    int64_t max_n = 0;
    for (int32_t i = 0; i < count; i++) {
        if (g_adamw_items[i].n > max_n) max_n = g_adamw_items[i].n;
    }
    if (max_n <= 0) return;
    size_t chunk = (size_t)count * sizeof(void *);
    size_t need = 8 * chunk;
    void *host_pack = malloc(need);
    if (!host_pack) {
        for (int32_t i = 0; i < count; i++) {
            AdamwItem *it = &g_adamw_items[i];
            k_adamw<<<grid1d(it->n), 256>>>(it->p, it->g, it->m, it->v, it->po, it->mo, it->vo, it->n, lr, b1, b2, eps, wd, bc1, bc2);
        }
        return;
    }
    if (need > g_adamw_meta_cap) {
        if (g_adamw_meta_buf) cudaFree(g_adamw_meta_buf);
        if (cudaMalloc(&g_adamw_meta_buf, need) != cudaSuccess) {
            g_adamw_meta_buf = NULL;
            g_adamw_meta_cap = 0;
            free(host_pack);
            for (int32_t i = 0; i < count; i++) {
                AdamwItem *it = &g_adamw_items[i];
                k_adamw<<<grid1d(it->n), 256>>>(it->p, it->g, it->m, it->v, it->po, it->mo, it->vo, it->n, lr, b1, b2, eps, wd, bc1, bc2);
            }
            return;
        }
        g_adamw_meta_cap = need;
    }
    char *hb = (char *)host_pack;
    const float **hps = (const float **)(hb + 0 * chunk);
    const float **hgs = (const float **)(hb + 1 * chunk);
    const float **hms = (const float **)(hb + 2 * chunk);
    const float **hvs = (const float **)(hb + 3 * chunk);
    float **hpos = (float **)(hb + 4 * chunk);
    float **hmos = (float **)(hb + 5 * chunk);
    float **hvos = (float **)(hb + 6 * chunk);
    int64_t *hnum = (int64_t *)(hb + 7 * chunk);
    for (int32_t i = 0; i < count; i++) {
        hps[i] = g_adamw_items[i].p;
        hgs[i] = g_adamw_items[i].g;
        hms[i] = g_adamw_items[i].m;
        hvs[i] = g_adamw_items[i].v;
        hpos[i] = g_adamw_items[i].po;
        hmos[i] = g_adamw_items[i].mo;
        hvos[i] = g_adamw_items[i].vo;
        hnum[i] = g_adamw_items[i].n;
    }
    cudaMemcpy(g_adamw_meta_buf, host_pack, need, cudaMemcpyHostToDevice);
    free(host_pack);
    char *db = (char *)g_adamw_meta_buf;
    const float **d_ps = (const float **)(db + 0 * chunk);
    const float **d_gs = (const float **)(db + 1 * chunk);
    const float **d_ms = (const float **)(db + 2 * chunk);
    const float **d_vs = (const float **)(db + 3 * chunk);
    float **d_pos = (float **)(db + 4 * chunk);
    float **d_mos = (float **)(db + 5 * chunk);
    float **d_vos = (float **)(db + 6 * chunk);
    int64_t *d_numels = (int64_t *)(db + 7 * chunk);
    int bx = grid1d(max_n);
    if (bx > 4096) bx = 4096;
    int32_t off = 0;
    while (off < count) {
        int32_t batch = count - off;
        if (batch > 65535) batch = 65535;
        dim3 grid((unsigned int)bx, (unsigned int)batch);
        k_adamw_multi<<<grid, 256>>>(
            batch, d_ps + off, d_gs + off, d_ms + off, d_vs + off,
            d_pos + off, d_mos + off, d_vos + off, d_numels + off,
            lr, b1, b2, eps, wd, bc1, bc2
        );
        off += batch;
    }
}

// ============================================================================
// Runtime NVRTC JIT Compilation & Kernel Launch (zero extra link dependencies)
// ============================================================================

typedef void *GsNvrtcProg;
typedef int (*FnNvrtcCreate)(GsNvrtcProg *, const char *, const char *, int, const char *const *, const char *const *);
typedef int (*FnNvrtcCompile)(GsNvrtcProg, int, const char *const *);
typedef int (*FnNvrtcGetPtxSize)(GsNvrtcProg, size_t *);
typedef int (*FnNvrtcGetPtx)(GsNvrtcProg, char *);
typedef int (*FnNvrtcDestroy)(GsNvrtcProg *);
typedef int (*FnCuModLoadData)(void **, const void *);
typedef int (*FnCuModUnload)(void *);
typedef int (*FnCuModGetFn)(void **, void *, const char *);
typedef int (*FnCuLaunch)(void *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void *, void **, void **);

static int g_jit_probed = 0;
static int g_jit_ready = 0;
static FnNvrtcCreate p_nvrtcCreate = NULL;
static FnNvrtcCompile p_nvrtcCompile = NULL;
static FnNvrtcGetPtxSize p_nvrtcGetPtxSize = NULL;
static FnNvrtcGetPtx p_nvrtcGetPtx = NULL;
static FnNvrtcDestroy p_nvrtcDestroy = NULL;
static FnCuModLoadData p_cuModLoadData = NULL;
static FnCuModUnload p_cuModUnload = NULL;
static FnCuModGetFn p_cuModGetFn = NULL;
static FnCuLaunch p_cuLaunch = NULL;

typedef struct { int32_t dev_id; void *mod; } JitOwnedMod;
static JitOwnedMod *g_jit_mods = NULL;
static int32_t g_jit_mod_count = 0;
static int32_t g_jit_mod_cap = 0;

static void track_jit_mod(int32_t dev_id, void *mod) {
    if (!mod) return;
    if (g_jit_mod_count >= g_jit_mod_cap) {
        int32_t nc = g_jit_mod_cap > 0 ? g_jit_mod_cap * 2 : 64;
        JitOwnedMod *nb = (JitOwnedMod *)realloc(g_jit_mods, (size_t)nc * sizeof(JitOwnedMod));
        if (!nb) return;
        g_jit_mods = nb;
        g_jit_mod_cap = nc;
    }
    g_jit_mods[g_jit_mod_count].dev_id = dev_id;
    g_jit_mods[g_jit_mod_count].mod = mod;
    g_jit_mod_count++;
}

static int ensure_nvrtc_loaded(void) {
    if (g_jit_probed) return g_jit_ready;
    g_jit_probed = 1;
    const char *nvrtc_names[] = {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2", NULL};
    void *h_nvrtc = NULL;
    for (int i = 0; nvrtc_names[i] && !h_nvrtc; i++) {
        h_nvrtc = dlopen(nvrtc_names[i], RTLD_LAZY | RTLD_LOCAL);
    }
    void *h_cuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!h_nvrtc || !h_cuda) return 0;
    p_nvrtcCreate = (FnNvrtcCreate)dlsym(h_nvrtc, "nvrtcCreateProgram");
    p_nvrtcCompile = (FnNvrtcCompile)dlsym(h_nvrtc, "nvrtcCompileProgram");
    p_nvrtcGetPtxSize = (FnNvrtcGetPtxSize)dlsym(h_nvrtc, "nvrtcGetPTXSize");
    p_nvrtcGetPtx = (FnNvrtcGetPtx)dlsym(h_nvrtc, "nvrtcGetPTX");
    p_nvrtcDestroy = (FnNvrtcDestroy)dlsym(h_nvrtc, "nvrtcDestroyProgram");
    p_cuModLoadData = (FnCuModLoadData)dlsym(h_cuda, "cuModuleLoadData");
    p_cuModUnload = (FnCuModUnload)dlsym(h_cuda, "cuModuleUnload");
    p_cuModGetFn = (FnCuModGetFn)dlsym(h_cuda, "cuModuleGetFunction");
    p_cuLaunch = (FnCuLaunch)dlsym(h_cuda, "cuLaunchKernel");
    if (p_nvrtcCreate && p_nvrtcCompile && p_nvrtcGetPtxSize && p_nvrtcGetPtx && p_nvrtcDestroy &&
        p_cuModLoadData && p_cuModGetFn && p_cuLaunch) {
        g_jit_ready = 1;
    }
    return g_jit_ready;
}

int32_t gs_cu_jit_available(void) {
    return ensure_nvrtc_loaded() ? 1 : 0;
}

void *gs_cu_jit_compile(int32_t dev_id, const char *src, const char *fn_name) {
    if (!ensure_nvrtc_loaded() || !src || !fn_name) return NULL;
    cudaSetDevice(dev_id);
    cudaFree(0);
    GsNvrtcProg prog = NULL;
    if (p_nvrtcCreate(&prog, src, "jit.cu", 0, NULL, NULL) != 0) return NULL;
    if (p_nvrtcCompile(prog, 0, NULL) != 0) {
        p_nvrtcDestroy(&prog);
        return NULL;
    }
    size_t ptx_sz = 0;
    if (p_nvrtcGetPtxSize(prog, &ptx_sz) != 0 || ptx_sz == 0) {
        p_nvrtcDestroy(&prog);
        return NULL;
    }
    char *ptx = (char *)malloc(ptx_sz);
    if (!ptx) {
        p_nvrtcDestroy(&prog);
        return NULL;
    }
    if (p_nvrtcGetPtx(prog, ptx) != 0) {
        free(ptx);
        p_nvrtcDestroy(&prog);
        return NULL;
    }
    p_nvrtcDestroy(&prog);
    void *mod = NULL;
    void *fn = NULL;
    int rc = p_cuModLoadData(&mod, ptx);
    free(ptx);
    if (rc != 0 || !mod) return NULL;
    if (p_cuModGetFn(&fn, mod, fn_name) != 0) {
        if (p_cuModUnload) p_cuModUnload(mod);
        return NULL;
    }
    track_jit_mod(dev_id, mod);
    return fn;
}

void gs_cu_jit_clear_cache(void) {
    for (int32_t i = 0; i < g_jit_mod_count; i++) {
        cudaSetDevice(g_jit_mods[i].dev_id);
        cudaDeviceSynchronize();
        if (p_cuModUnload && g_jit_mods[i].mod) p_cuModUnload(g_jit_mods[i].mod);
    }
    g_jit_mod_count = 0;
}

static const void *g_jit_ins[16];
static int32_t g_jit_n_in = 0;
static float *g_jit_scs = NULL;
static int32_t g_jit_n_sc = 0;
static int32_t g_jit_sc_host_cap = 0;
static float *g_jit_sc_dev = NULL;
static int32_t g_jit_sc_cap = 0;

void gs_cu_jit_launch_reset(void) {
    g_jit_n_in = 0;
    g_jit_n_sc = 0;
}

void gs_cu_jit_launch_push_in(const void *ptr) {
    if (g_jit_n_in < 16) g_jit_ins[g_jit_n_in++] = ptr;
}

void gs_cu_jit_launch_push_sc(float v) {
    if (g_jit_n_sc >= g_jit_sc_host_cap) {
        int32_t nc = g_jit_sc_host_cap > 0 ? g_jit_sc_host_cap * 2 : 64;
        float *nb = (float *)realloc(g_jit_scs, (size_t)nc * sizeof(float));
        if (!nb) return;
        g_jit_scs = nb;
        g_jit_sc_host_cap = nc;
    }
    g_jit_scs[g_jit_n_sc++] = v;
}

int32_t gs_cu_jit_launch_run(int32_t dev_id, void *fn, void *out_ptr, int32_t n) {
    if (!fn || n <= 0 || !p_cuLaunch) return 0;
    cudaSetDevice(dev_id);
    int32_t sc_cnt = g_jit_n_sc > 0 ? g_jit_n_sc : 1;
    if (sc_cnt > g_jit_sc_cap) {
        if (g_jit_sc_dev) cudaFree(g_jit_sc_dev);
        if (cudaMalloc(&g_jit_sc_dev, (size_t)sc_cnt * sizeof(float)) != cudaSuccess) {
            g_jit_sc_dev = NULL;
            g_jit_sc_cap = 0;
            return 0;
        }
        g_jit_sc_cap = sc_cnt;
    }
    if (g_jit_n_sc > 0) {
        if (cudaMemcpyAsync(g_jit_sc_dev, g_jit_scs, (size_t)g_jit_n_sc * sizeof(float), cudaMemcpyHostToDevice, 0) != cudaSuccess) {
            return 0;
        }
    }
    void *args[20];
    int32_t k = 0;
    for (int32_t i = 0; i < g_jit_n_in; i++) args[k++] = (void *)&g_jit_ins[i];
    args[k++] = (void *)&out_ptr;
    args[k++] = (void *)&g_jit_sc_dev;
    args[k++] = (void *)&n;
    int rc = p_cuLaunch(fn, (unsigned int)grid1d(n), 1, 1, 256, 1, 1, 0, NULL, args, NULL);
    return rc == 0 ? 1 : 0;
}

// ============================================================================
// Zero-Sync Multi-Tensor Grad Norm Clipping
// ============================================================================

typedef struct {
    const float *g;
    float *go;
    int64_t n;
} GradClipItem;

static GradClipItem *g_clip_items = NULL;
static int32_t g_clip_count = 0;
static int32_t g_clip_cap = 0;
static void *g_clip_meta_buf = NULL;
static size_t g_clip_meta_cap = 0;

void gs_cu_gradclip_batch_reset(void) {
    g_clip_count = 0;
}

void gs_cu_gradclip_batch_push(const float *g, float *go, int64_t n) {
    if (n <= 0) return;
    if (g_clip_count >= g_clip_cap) {
        int32_t nc = g_clip_cap > 0 ? g_clip_cap * 2 : 128;
        GradClipItem *nb = (GradClipItem *)realloc(g_clip_items, (size_t)nc * sizeof(GradClipItem));
        if (!nb) return;
        g_clip_items = nb;
        g_clip_cap = nc;
    }
    GradClipItem *it = &g_clip_items[g_clip_count++];
    it->g = g; it->go = go; it->n = n;
}

void gs_cu_gradclip_batch_run(int32_t d, float max_norm) {
    int32_t count = g_clip_count;
    g_clip_count = 0;
    if (count <= 0) return;
    ensure_scratch(d);
    int64_t max_n = 0;
    for (int32_t i = 0; i < count; i++) {
        if (g_clip_items[i].n > max_n) max_n = g_clip_items[i].n;
    }
    if (max_n <= 0) return;
    size_t chunk = (size_t)count * sizeof(void *);
    size_t need = 3 * chunk;
    void *host_pack = malloc(need);
    if (!host_pack) return;
    if (need > g_clip_meta_cap) {
        if (g_clip_meta_buf) cudaFree(g_clip_meta_buf);
        if (cudaMalloc(&g_clip_meta_buf, need) != cudaSuccess) {
            g_clip_meta_buf = NULL;
            g_clip_meta_cap = 0;
            free(host_pack);
            return;
        }
        g_clip_meta_cap = need;
    }
    char *hb = (char *)host_pack;
    const float **hgs = (const float **)(hb + 0 * chunk);
    float **hgos = (float **)(hb + 1 * chunk);
    int64_t *hnum = (int64_t *)(hb + 2 * chunk);
    for (int32_t i = 0; i < count; i++) {
        hgs[i] = g_clip_items[i].g;
        hgos[i] = g_clip_items[i].go;
        hnum[i] = g_clip_items[i].n;
    }
    cudaMemcpy(g_clip_meta_buf, host_pack, need, cudaMemcpyHostToDevice);
    free(host_pack);
    char *db = (char *)g_clip_meta_buf;
    const float **d_gs = (const float **)(db + 0 * chunk);
    float **d_gos = (float **)(db + 1 * chunk);
    int64_t *d_num = (int64_t *)(db + 2 * chunk);
    cudaMemsetAsync(g_reduce_buf, 0, sizeof(float), 0);
    int bx = grid1d((max_n >> 2) + 1);
    if (bx > 1024) bx = 1024;
    int32_t off = 0;
    while (off < count) {
        int32_t batch = count - off;
        if (batch > 65535) batch = 65535;
        dim3 grid((unsigned int)bx, (unsigned int)batch);
        k_sum_sq_multi<<<grid, 256>>>(batch, d_gs + off, d_num + off, g_reduce_buf);
        off += batch;
    }
    off = 0;
    while (off < count) {
        int32_t batch = count - off;
        if (batch > 65535) batch = 65535;
        dim3 grid((unsigned int)bx, (unsigned int)batch);
        k_clip_scale_multi<<<grid, 256>>>(batch, d_gs + off, d_gos + off, d_num + off, g_reduce_buf, max_norm);
        off += batch;
    }
}

} // extern "C"

#endif // GS_TENSOR_CUDA
