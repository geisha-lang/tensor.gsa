#include "gs_tensor_backend.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__) && !defined(GS_TENSOR_NO_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define GS_HAS_CBLAS 1
#define GS_HAS_VDSP  1
#elif defined(GS_TENSOR_OPENBLAS)
#include <cblas.h>
#define GS_HAS_CBLAS 1
#endif

static int32_t cpu_device_count(void) { return 1; }

static void *cpu_alloc(int32_t dev_id, int64_t nbytes) {
    (void)dev_id;
    size_t sz = nbytes > 0 ? (size_t)nbytes : 64;
    size_t aligned = (sz + 63) & ~(size_t)63;
    void *p = NULL;
    if (posix_memalign(&p, 64, aligned) != 0) return NULL;
    memset(p, 0, aligned);
    return p;
}

static void cpu_free(void *ptr) { free(ptr); }

static void cpu_copy_bytes(int32_t dev_id, void *dst, const void *src, int64_t nbytes) {
    (void)dev_id;
    if (nbytes > 0 && dst != src) memcpy(dst, src, (size_t)nbytes);
}

static void cpu_sgemm(
    int32_t dev_id, int32_t transA, int32_t transB,
    int32_t M, int32_t N, int32_t K,
    float alpha, const float *A, const float *B,
    float beta, float *C
) {
    (void)dev_id;
    if (M <= 0 || N <= 0 || K <= 0) return;
    int32_t lda = transA ? M : K;
    int32_t ldb = transB ? K : N;
    int32_t ldc = N;
#ifdef GS_HAS_CBLAS
    cblas_sgemm(
        CblasRowMajor,
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        M, N, K,
        alpha, A, lda, B, ldb,
        beta, C, ldc
    );
#else
    if (beta == 0.0f) {
        memset(C, 0, (size_t)((int64_t)M * N) * sizeof(float));
    } else if (beta != 1.0f) {
        for (int64_t i = 0; i < (int64_t)M * N; i++) C[i] *= beta;
    }
    const int32_t BS = 32;
    for (int32_t ii = 0; ii < M; ii += BS) {
        int32_t imax = ii + BS < M ? ii + BS : M;
        for (int32_t kk = 0; kk < K; kk += BS) {
            int32_t kmax = kk + BS < K ? kk + BS : K;
            for (int32_t jj = 0; jj < N; jj += BS) {
                int32_t jmax = jj + BS < N ? jj + BS : N;
                for (int32_t i = ii; i < imax; i++) {
                    for (int32_t k = kk; k < kmax; k++) {
                        float av = alpha * (transA ? A[(int64_t)k * lda + i] : A[(int64_t)i * lda + k]);
                        float *crow = C + (int64_t)i * ldc;
                        if (!transB) {
                            const float *brow = B + (int64_t)k * ldb;
                            for (int32_t j = jj; j < jmax; j++) crow[j] += av * brow[j];
                        } else {
                            for (int32_t j = jj; j < jmax; j++) crow[j] += av * B[(int64_t)j * ldb + k];
                        }
                    }
                }
            }
        }
    }
#endif
}

static void cpu_bgemm(
    int32_t dev_id, int32_t batch, int32_t transA, int32_t transB,
    int32_t M, int32_t N, int32_t K,
    float alpha, const float *A, const float *B,
    float beta, float *C
) {
    int64_t strideA = (int64_t)M * K;
    int64_t strideB = (int64_t)K * N;
    int64_t strideC = (int64_t)M * N;
    for (int32_t b = 0; b < batch; b++) {
        cpu_sgemm(
            dev_id, transA, transB, M, N, K, alpha,
            A + (int64_t)b * strideA,
            B + (int64_t)b * strideB,
            beta,
            C + (int64_t)b * strideC
        );
    }
}

static void cpu_fill(int32_t dev_id, float *dst, int64_t n, float val) {
    (void)dev_id;
    for (int64_t i = 0; i < n; i++) dst[i] = val;
}

static void cpu_copy(int32_t dev_id, float *dst, const float *src, int64_t n) {
    (void)dev_id;
    if (n > 0 && dst != src) memcpy(dst, src, (size_t)n * sizeof(float));
}

static void cpu_add(int32_t dev_id, const float *a, const float *b, float *out, int64_t n) {
    (void)dev_id;
#ifdef GS_HAS_VDSP
    vDSP_vadd(a, 1, b, 1, out, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; i++) out[i] = a[i] + b[i];
#endif
}

static void cpu_sub(int32_t dev_id, const float *a, const float *b, float *out, int64_t n) {
    (void)dev_id;
#ifdef GS_HAS_VDSP
    vDSP_vsub(b, 1, a, 1, out, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; i++) out[i] = a[i] - b[i];
#endif
}

static void cpu_mul(int32_t dev_id, const float *a, const float *b, float *out, int64_t n) {
    (void)dev_id;
#ifdef GS_HAS_VDSP
    vDSP_vmul(a, 1, b, 1, out, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; i++) out[i] = a[i] * b[i];
#endif
}

static void cpu_div_ew(int32_t dev_id, const float *a, const float *b, float *out, int64_t n) {
    (void)dev_id;
#ifdef GS_HAS_VDSP
    // vDSP_vdiv computes C = B / A
    vDSP_vdiv(b, 1, a, 1, out, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; i++) out[i] = a[i] / b[i];
#endif
}

static void cpu_scale(int32_t dev_id, const float *a, float s, float *out, int64_t n) {
    (void)dev_id;
#ifdef GS_HAS_VDSP
    vDSP_vsmul(a, 1, &s, out, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; i++) out[i] = a[i] * s;
#endif
}

static void cpu_add_scalar(int32_t dev_id, const float *a, float s, float *out, int64_t n) {
    (void)dev_id;
#ifdef GS_HAS_VDSP
    vDSP_vsadd(a, 1, &s, out, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; i++) out[i] = a[i] + s;
#endif
}

static void cpu_pow_scalar_fwd(int32_t dev_id, const float *a, float p, float *out, int64_t n) {
    (void)dev_id;
    for (int64_t i = 0; i < n; i++) out[i] = powf(a[i], p);
}

static void cpu_pow_scalar_bwd(int32_t dev_id, const float *a, const float *gy, float p, float *gx, int64_t n) {
    (void)dev_id;
    for (int64_t i = 0; i < n; i++) gx[i] = gy[i] * p * powf(a[i], p - 1.0f);
}

static void cpu_clamp_fwd(int32_t dev_id, const float *a, float lo, float hi, float *out, int64_t n) {
    (void)dev_id;
    for (int64_t i = 0; i < n; i++) {
        float v = a[i];
        out[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}

static void cpu_clamp_bwd(int32_t dev_id, const float *a, const float *gy, float lo, float hi, float *gx, int64_t n) {
    (void)dev_id;
    for (int64_t i = 0; i < n; i++) {
        float v = a[i];
        gx[i] = (v >= lo && v <= hi) ? gy[i] : 0.0f;
    }
}

static void cpu_leaky_relu_fwd(int32_t dev_id, const float *a, float slope, float *out, int64_t n) {
    (void)dev_id;
    for (int64_t i = 0; i < n; i++) out[i] = a[i] > 0.0f ? a[i] : slope * a[i];
}

static void cpu_leaky_relu_bwd(int32_t dev_id, const float *a, const float *gy, float slope, float *gx, int64_t n) {
    (void)dev_id;
    for (int64_t i = 0; i < n; i++) gx[i] = a[i] > 0.0f ? gy[i] : slope * gy[i];
}

static void cpu_axpy(int32_t dev_id, float alpha, const float *x, float *y, int64_t n) {
    (void)dev_id;
#ifdef GS_HAS_CBLAS
    cblas_saxpy((int)n, alpha, x, 1, y, 1);
#else
    for (int64_t i = 0; i < n; i++) y[i] += alpha * x[i];
#endif
}

static void cpu_bcast_binop(
    int32_t dev_id, int32_t op, int32_t ndim, int64_t total,
    const int32_t *out_shape, const int32_t *a_strides, const int32_t *b_strides,
    const float *a, const float *b, float *out
) {
    (void)dev_id;
    for (int64_t idx = 0; idx < total; idx++) {
        int64_t rem = idx;
        int64_t oa = 0, ob = 0;
        for (int32_t d = ndim - 1; d >= 0; d--) {
            int32_t coord = (int32_t)(rem % out_shape[d]);
            rem /= out_shape[d];
            oa += (int64_t)coord * a_strides[d];
            ob += (int64_t)coord * b_strides[d];
        }
        float va = a[oa], vb = b[ob];
        if (op == GS_BOP_ADD)      out[idx] = va + vb;
        else if (op == GS_BOP_SUB) out[idx] = va - vb;
        else if (op == GS_BOP_MUL) out[idx] = va * vb;
        else                       out[idx] = va / vb;
    }
}

static void cpu_bcast_reduce_sum(
    int32_t dev_id, int32_t ndim, int64_t src_total, int64_t dst_total,
    const int32_t *src_shape, const int32_t *dst_strides,
    const float *src, float *dst
) {
    (void)dev_id;
    memset(dst, 0, (size_t)dst_total * sizeof(float));
    for (int64_t idx = 0; idx < src_total; idx++) {
        int64_t rem = idx;
        int64_t od = 0;
        for (int32_t d = ndim - 1; d >= 0; d--) {
            int32_t coord = (int32_t)(rem % src_shape[d]);
            rem /= src_shape[d];
            od += (int64_t)coord * dst_strides[d];
        }
        dst[od] += src[idx];
    }
}

static void cpu_unary_fwd(int32_t dev_id, int32_t op, const float *x, float *out, int64_t n) {
    (void)dev_id;
    const float gc = 0.7978845608028654f;
    switch (op) {
        case GS_UOP_EXP:     for (int64_t i = 0; i < n; i++) out[i] = expf(x[i]); break;
        case GS_UOP_LOG:     for (int64_t i = 0; i < n; i++) out[i] = logf(x[i]); break;
        case GS_UOP_SQRT:    for (int64_t i = 0; i < n; i++) out[i] = sqrtf(x[i]); break;
        case GS_UOP_RSQRT:   for (int64_t i = 0; i < n; i++) out[i] = 1.0f / sqrtf(x[i]); break;
        case GS_UOP_ABS:     for (int64_t i = 0; i < n; i++) out[i] = fabsf(x[i]); break;
        case GS_UOP_SQUARE:  for (int64_t i = 0; i < n; i++) out[i] = x[i] * x[i]; break;
        case GS_UOP_TANH:    for (int64_t i = 0; i < n; i++) out[i] = tanhf(x[i]); break;
        case GS_UOP_SIGMOID: for (int64_t i = 0; i < n; i++) out[i] = 1.0f / (1.0f + expf(-x[i])); break;
        case GS_UOP_ROUND:   for (int64_t i = 0; i < n; i++) out[i] = roundf(x[i]); break;
        case GS_UOP_SILU:    for (int64_t i = 0; i < n; i++) { float v = x[i]; out[i] = v / (1.0f + expf(-v)); } break;
        case GS_UOP_GELU:    for (int64_t i = 0; i < n; i++) { float v = x[i]; out[i] = 0.5f * v * (1.0f + tanhf(gc * (v + 0.044715f * v * v * v))); } break;
        case GS_UOP_RELU:    for (int64_t i = 0; i < n; i++) out[i] = x[i] > 0.0f ? x[i] : 0.0f; break;
        default: break;
    }
}

static void cpu_unary_bwd(int32_t dev_id, int32_t op, const float *x, const float *y, const float *gy, float *gx, int64_t n) {
    (void)dev_id;
    const float gc = 0.7978845608028654f;
    switch (op) {
        case GS_UOP_EXP:     for (int64_t i = 0; i < n; i++) gx[i] = gy[i] * y[i]; break;
        case GS_UOP_LOG:     for (int64_t i = 0; i < n; i++) gx[i] = gy[i] / x[i]; break;
        case GS_UOP_SQRT:    for (int64_t i = 0; i < n; i++) gx[i] = gy[i] / (2.0f * y[i] + 1e-12f); break;
        case GS_UOP_RSQRT:   for (int64_t i = 0; i < n; i++) { float r = y[i]; gx[i] = -0.5f * gy[i] * r * r * r; } break;
        case GS_UOP_ABS:     for (int64_t i = 0; i < n; i++) gx[i] = x[i] > 0.0f ? gy[i] : (x[i] < 0.0f ? -gy[i] : 0.0f); break;
        case GS_UOP_SQUARE:  for (int64_t i = 0; i < n; i++) gx[i] = 2.0f * x[i] * gy[i]; break;
        case GS_UOP_TANH:    for (int64_t i = 0; i < n; i++) gx[i] = gy[i] * (1.0f - y[i] * y[i]); break;
        case GS_UOP_SIGMOID: for (int64_t i = 0; i < n; i++) gx[i] = gy[i] * y[i] * (1.0f - y[i]); break;
        case GS_UOP_ROUND:   for (int64_t i = 0; i < n; i++) gx[i] = 0.0f; break;
        case GS_UOP_SILU:    for (int64_t i = 0; i < n; i++) { float v = x[i]; float s = 1.0f / (1.0f + expf(-v)); gx[i] = gy[i] * s * (1.0f + v * (1.0f - s)); } break;
        case GS_UOP_GELU:    for (int64_t i = 0; i < n; i++) { float v = x[i], v2 = v * v; float th = tanhf(gc * (v + 0.044715f * v * v2)); gx[i] = gy[i] * (0.5f * (1.0f + th) + 0.5f * v * (1.0f - th * th) * gc * (1.0f + 0.134145f * v2)); } break;
        case GS_UOP_RELU:    for (int64_t i = 0; i < n; i++) gx[i] = x[i] > 0.0f ? gy[i] : 0.0f; break;
        default: break;
    }
}

static void cpu_slice_axis_fwd(int32_t dev_id, const float *x, float *out, int32_t outer, int32_t in_dim, int32_t inner, int32_t start, int32_t len) {
    (void)dev_id;
    int64_t chunk_bytes = (int64_t)len * inner * (int64_t)sizeof(float);
    for (int32_t o = 0; o < outer; o++) {
        const float *src = x + ((int64_t)o * in_dim + start) * inner;
        float *dst = out + (int64_t)o * len * inner;
        memcpy(dst, src, (size_t)chunk_bytes);
    }
}

static void cpu_slice_axis_bwd(int32_t dev_id, const float *gy, float *gx, int32_t outer, int32_t in_dim, int32_t inner, int32_t start, int32_t len) {
    (void)dev_id;
    memset(gx, 0, (size_t)((int64_t)outer * in_dim * inner) * sizeof(float));
    int64_t chunk_bytes = (int64_t)len * inner * (int64_t)sizeof(float);
    for (int32_t o = 0; o < outer; o++) {
        const float *src = gy + (int64_t)o * len * inner;
        float *dst = gx + ((int64_t)o * in_dim + start) * inner;
        memcpy(dst, src, (size_t)chunk_bytes);
    }
}

static void cpu_sum_axis(int32_t dev_id, const float *x, float *out, int32_t outer, int32_t dim, int32_t inner) {
    (void)dev_id;
    memset(out, 0, (size_t)((int64_t)outer * inner) * sizeof(float));
    for (int32_t o = 0; o < outer; o++) {
        float *orow = out + (int64_t)o * inner;
        for (int32_t d = 0; d < dim; d++) {
            const float *xrow = x + ((int64_t)o * dim + d) * inner;
            for (int32_t k = 0; k < inner; k++) orow[k] += xrow[k];
        }
    }
}

static void cpu_max_axis(int32_t dev_id, const float *x, float *out, int32_t outer, int32_t dim, int32_t inner) {
    (void)dev_id;
    for (int32_t o = 0; o < outer; o++) {
        float *orow = out + (int64_t)o * inner;
        const float *x0 = x + ((int64_t)o * dim) * inner;
        memcpy(orow, x0, (size_t)inner * sizeof(float));
        for (int32_t d = 1; d < dim; d++) {
            const float *xrow = x + ((int64_t)o * dim + d) * inner;
            for (int32_t k = 0; k < inner; k++) if (xrow[k] > orow[k]) orow[k] = xrow[k];
        }
    }
}

static float cpu_sum_all(int32_t dev_id, const float *x, int64_t n) {
    (void)dev_id;
    double acc = 0.0;
    for (int64_t i = 0; i < n; i++) acc += (double)x[i];
    return (float)acc;
}

static float cpu_sum_sq(int32_t dev_id, const float *x, int64_t n) {
    (void)dev_id;
    double acc = 0.0;
    for (int64_t i = 0; i < n; i++) { double v = (double)x[i]; acc += v * v; }
    return (float)acc;
}

static void cpu_rms_norm_fwd(int32_t dev_id, const float *x, const float *w, float *out, int32_t rows, int32_t dim, float eps) {
    (void)dev_id;
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (int64_t)r * dim;
        float *orow = out + (int64_t)r * dim;
        double ss = 0.0;
        for (int32_t c = 0; c < dim; c++) ss += (double)xr[c] * (double)xr[c];
        float inv_rms = 1.0f / sqrtf((float)(ss / (double)dim) + eps);
        for (int32_t c = 0; c < dim; c++) orow[c] = xr[c] * inv_rms * w[c];
    }
}

static void cpu_rms_norm_bwd(int32_t dev_id, const float *x, const float *w, const float *gy, float *gx, float *gw, int32_t rows, int32_t dim, float eps) {
    (void)dev_id;
    memset(gw, 0, (size_t)dim * sizeof(float));
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (int64_t)r * dim;
        const float *gyr = gy + (int64_t)r * dim;
        float *gxr = gx + (int64_t)r * dim;
        double ss = 0.0;
        for (int32_t c = 0; c < dim; c++) ss += (double)xr[c] * (double)xr[c];
        float inv_rms = 1.0f / sqrtf((float)(ss / (double)dim) + eps);
        float inv_rms3_over_d = (inv_rms * inv_rms * inv_rms) / (float)dim;
        double dot = 0.0;
        for (int32_t c = 0; c < dim; c++) {
            gw[c] += gyr[c] * xr[c] * inv_rms;
            dot += (double)(gyr[c] * w[c] * xr[c]);
        }
        float coeff = (float)dot * inv_rms3_over_d;
        for (int32_t c = 0; c < dim; c++) gxr[c] = gyr[c] * w[c] * inv_rms - xr[c] * coeff;
    }
}

static void cpu_layer_norm_fwd(int32_t dev_id, const float *x, const float *w, const float *b, float *out, int32_t rows, int32_t dim, float eps) {
    (void)dev_id;
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (int64_t)r * dim;
        float *orow = out + (int64_t)r * dim;
        double mean = 0.0;
        for (int32_t c = 0; c < dim; c++) mean += (double)xr[c];
        mean /= (double)dim;
        double var = 0.0;
        for (int32_t c = 0; c < dim; c++) { double d = (double)xr[c] - mean; var += d * d; }
        float inv_std = 1.0f / sqrtf((float)(var / (double)dim) + eps);
        for (int32_t c = 0; c < dim; c++) {
            float xhat = ((float)((double)xr[c] - mean)) * inv_std;
            orow[c] = xhat * w[c] + (b ? b[c] : 0.0f);
        }
    }
}

static void cpu_layer_norm_bwd(int32_t dev_id, const float *x, const float *w, const float *gy, float *gx, float *gw, float *gb, int32_t rows, int32_t dim, float eps) {
    (void)dev_id;
    memset(gw, 0, (size_t)dim * sizeof(float));
    if (gb) memset(gb, 0, (size_t)dim * sizeof(float));
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (int64_t)r * dim;
        const float *gyr = gy + (int64_t)r * dim;
        float *gxr = gx + (int64_t)r * dim;
        double mean = 0.0;
        for (int32_t c = 0; c < dim; c++) mean += (double)xr[c];
        mean /= (double)dim;
        double var = 0.0;
        for (int32_t c = 0; c < dim; c++) { double d = (double)xr[c] - mean; var += d * d; }
        float inv_std = 1.0f / sqrtf((float)(var / (double)dim) + eps);
        double sum1 = 0.0, sum2 = 0.0;
        for (int32_t c = 0; c < dim; c++) {
            float xhat = ((float)((double)xr[c] - mean)) * inv_std;
            float g = gyr[c];
            gw[c] += g * xhat;
            if (gb) gb[c] += g;
            float dxhat = g * w[c];
            sum1 += (double)dxhat;
            sum2 += (double)(dxhat * xhat);
        }
        float inv_d = 1.0f / (float)dim;
        for (int32_t c = 0; c < dim; c++) {
            float xhat = ((float)((double)xr[c] - mean)) * inv_std;
            float dxhat = gyr[c] * w[c];
            gxr[c] = inv_std * (dxhat - inv_d * ((float)sum1 + xhat * (float)sum2));
        }
    }
}

static void cpu_l2_norm_last_fwd(int32_t dev_id, const float *x, float *out, int32_t rows, int32_t dim, float eps) {
    (void)dev_id;
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (int64_t)r * dim;
        float *orow = out + (int64_t)r * dim;
        double ss = 0.0;
        for (int32_t c = 0; c < dim; c++) ss += (double)xr[c] * (double)xr[c];
        float norm = sqrtf((float)ss);
        float inv = 1.0f / (norm > eps ? norm : eps);
        for (int32_t c = 0; c < dim; c++) orow[c] = xr[c] * inv;
    }
}

static void cpu_l2_norm_last_bwd(int32_t dev_id, const float *x, const float *gy, float *gx, int32_t rows, int32_t dim, float eps) {
    (void)dev_id;
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (int64_t)r * dim;
        const float *gyr = gy + (int64_t)r * dim;
        float *gxr = gx + (int64_t)r * dim;
        double ss = 0.0;
        for (int32_t c = 0; c < dim; c++) ss += (double)xr[c] * (double)xr[c];
        float norm = sqrtf((float)ss);
        if (norm <= eps) {
            float inv = 1.0f / eps;
            for (int32_t c = 0; c < dim; c++) gxr[c] = gyr[c] * inv;
        } else {
            float inv = 1.0f / norm;
            float inv3 = inv * inv * inv;
            double dot = 0.0;
            for (int32_t c = 0; c < dim; c++) dot += (double)xr[c] * (double)gyr[c];
            float coeff = (float)dot * inv3;
            for (int32_t c = 0; c < dim; c++) gxr[c] = gyr[c] * inv - xr[c] * coeff;
        }
    }
}

static void cpu_softmax_fwd(int32_t dev_id, const float *x, float *out, int32_t rows, int32_t cols) {
    (void)dev_id;
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (int64_t)r * cols;
        float *orow = out + (int64_t)r * cols;
        float mx = xr[0];
        for (int32_t c = 1; c < cols; c++) if (xr[c] > mx) mx = xr[c];
        double sum = 0.0;
        for (int32_t c = 0; c < cols; c++) { float e = expf(xr[c] - mx); orow[c] = e; sum += (double)e; }
        float inv = (float)(1.0 / sum);
        for (int32_t c = 0; c < cols; c++) orow[c] *= inv;
    }
}

static void cpu_softmax_bwd(int32_t dev_id, const float *y, const float *gy, float *gx, int32_t rows, int32_t cols) {
    (void)dev_id;
    for (int32_t r = 0; r < rows; r++) {
        const float *yr = y + (int64_t)r * cols;
        const float *gyr = gy + (int64_t)r * cols;
        float *gxr = gx + (int64_t)r * cols;
        double dot = 0.0;
        for (int32_t c = 0; c < cols; c++) dot += (double)yr[c] * (double)gyr[c];
        float d = (float)dot;
        for (int32_t c = 0; c < cols; c++) gxr[c] = yr[c] * (gyr[c] - d);
    }
}

static void cpu_log_sum_exp_fwd(int32_t dev_id, const float *x, float *out, int32_t rows, int32_t cols) {
    (void)dev_id;
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (int64_t)r * cols;
        float mx = xr[0];
        for (int32_t c = 1; c < cols; c++) if (xr[c] > mx) mx = xr[c];
        double sum = 0.0;
        for (int32_t c = 0; c < cols; c++) sum += exp((double)(xr[c] - mx));
        out[r] = (float)((double)mx + log(sum));
    }
}

static void cpu_log_sum_exp_bwd(int32_t dev_id, const float *x, const float *lse, const float *gy, float *gx, int32_t rows, int32_t cols) {
    (void)dev_id;
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (int64_t)r * cols;
        float *gxr = gx + (int64_t)r * cols;
        float lr = lse[r], g = gy[r];
        for (int32_t c = 0; c < cols; c++) gxr[c] = g * expf(xr[c] - lr);
    }
}

static float cpu_cross_entropy_fwd(int32_t dev_id, const float *logits, const int32_t *targets, float *probs_out, int32_t rows, int32_t vocab, int32_t ignore_idx) {
    (void)dev_id;
    double total_loss = 0.0;
    int32_t valid = 0;
    for (int32_t r = 0; r < rows; r++) {
        const float *lr = logits + (int64_t)r * vocab;
        float *pr = probs_out + (int64_t)r * vocab;
        float mx = lr[0];
        for (int32_t c = 1; c < vocab; c++) if (lr[c] > mx) mx = lr[c];
        double sum = 0.0;
        for (int32_t c = 0; c < vocab; c++) { float e = expf(lr[c] - mx); pr[c] = e; sum += (double)e; }
        float inv = (float)(1.0 / sum);
        for (int32_t c = 0; c < vocab; c++) pr[c] *= inv;
        int32_t tgt = targets[r];
        if (tgt != ignore_idx && tgt >= 0 && tgt < vocab) {
            total_loss -= ((double)(lr[tgt] - mx) - log(sum));
            valid++;
        }
    }
    return valid > 0 ? (float)(total_loss / (double)valid) : 0.0f;
}

static void cpu_cross_entropy_bwd(int32_t dev_id, const float *probs, const int32_t *targets, float g_loss, float *g_logits, int32_t rows, int32_t vocab, int32_t ignore_idx) {
    (void)dev_id;
    int32_t valid = 0;
    for (int32_t r = 0; r < rows; r++) {
        int32_t tgt = targets[r];
        if (tgt != ignore_idx && tgt >= 0 && tgt < vocab) valid++;
    }
    if (valid <= 0) {
        memset(g_logits, 0, (size_t)((int64_t)rows * vocab) * sizeof(float));
        return;
    }
    float scale = g_loss / (float)valid;
    for (int32_t r = 0; r < rows; r++) {
        const float *pr = probs + (int64_t)r * vocab;
        float *gr = g_logits + (int64_t)r * vocab;
        int32_t tgt = targets[r];
        if (tgt == ignore_idx || tgt < 0 || tgt >= vocab) {
            memset(gr, 0, (size_t)vocab * sizeof(float));
        } else {
            for (int32_t c = 0; c < vocab; c++) gr[c] = pr[c] * scale;
            gr[tgt] -= scale;
        }
    }
}

static void cpu_embedding_fwd(int32_t dev_id, const float *table, const int32_t *ids, float *out, int32_t n_ids, int32_t dim) {
    (void)dev_id;
    for (int32_t i = 0; i < n_ids; i++) {
        memcpy(out + (int64_t)i * dim, table + (int64_t)ids[i] * dim, (size_t)dim * sizeof(float));
    }
}

static void cpu_embedding_bwd(int32_t dev_id, const int32_t *ids, const float *gy, float *g_table, int32_t n_ids, int32_t vocab, int32_t dim) {
    (void)dev_id;
    memset(g_table, 0, (size_t)((int64_t)vocab * dim) * sizeof(float));
    for (int32_t i = 0; i < n_ids; i++) {
        int32_t id = ids[i];
        if (id >= 0 && id < vocab) {
            const float *gyr = gy + (int64_t)i * dim;
            float *gtr = g_table + (int64_t)id * dim;
            for (int32_t c = 0; c < dim; c++) gtr[c] += gyr[c];
        }
    }
}

static void cpu_rope_fwd(int32_t dev_id, const float *x, float *out, int32_t batch, int32_t seq, int32_t heads, int32_t head_dim, float theta) {
    (void)dev_id;
    int32_t half = head_dim / 2;
    for (int32_t b = 0; b < batch; b++) {
        for (int32_t t = 0; t < seq; t++) {
            for (int32_t h = 0; h < heads; h++) {
                int64_t base = (((int64_t)b * seq + t) * heads + h) * head_dim;
                for (int32_t i = 0; i < half; i++) {
                    float ang = (float)t / powf(theta, (float)(2 * i) / (float)head_dim);
                    float cs = cosf(ang), sn = sinf(ang);
                    float x0 = x[base + i], x1 = x[base + half + i];
                    out[base + i]        = x0 * cs - x1 * sn;
                    out[base + half + i] = x0 * sn + x1 * cs;
                }
            }
        }
    }
}

static void cpu_rope_bwd(int32_t dev_id, const float *gy, float *gx, int32_t batch, int32_t seq, int32_t heads, int32_t head_dim, float theta) {
    (void)dev_id;
    int32_t half = head_dim / 2;
    for (int32_t b = 0; b < batch; b++) {
        for (int32_t t = 0; t < seq; t++) {
            for (int32_t h = 0; h < heads; h++) {
                int64_t base = (((int64_t)b * seq + t) * heads + h) * head_dim;
                for (int32_t i = 0; i < half; i++) {
                    float ang = (float)t / powf(theta, (float)(2 * i) / (float)head_dim);
                    float cs = cosf(ang), sn = sinf(ang);
                    float g0 = gy[base + i], g1 = gy[base + half + i];
                    gx[base + i]        =  g0 * cs + g1 * sn;
                    gx[base + half + i] = -g0 * sn + g1 * cs;
                }
            }
        }
    }
}

static void cpu_transpose_2d(int32_t dev_id, const float *x, float *out, int32_t rows, int32_t cols) {
    (void)dev_id;
    for (int32_t r = 0; r < rows; r++)
        for (int32_t c = 0; c < cols; c++)
            out[(int64_t)c * rows + r] = x[(int64_t)r * cols + c];
}

static void cpu_transpose_0213(int32_t dev_id, const float *x, float *out, int32_t d0, int32_t d1, int32_t d2, int32_t d3) {
    (void)dev_id;
    for (int32_t i0 = 0; i0 < d0; i0++)
        for (int32_t i1 = 0; i1 < d1; i1++)
            for (int32_t i2 = 0; i2 < d2; i2++)
                memcpy(
                    out + (((int64_t)i0 * d2 + i2) * d1 + i1) * d3,
                    x   + (((int64_t)i0 * d1 + i1) * d2 + i2) * d3,
                    (size_t)d3 * sizeof(float)
                );
}

static void cpu_causal_mask_add(int32_t dev_id, float *scores, int32_t bh, int32_t seq) {
    (void)dev_id;
    for (int32_t b = 0; b < bh; b++) {
        float *s = scores + (int64_t)b * seq * seq;
        for (int32_t i = 0; i < seq; i++)
            for (int32_t j = i + 1; j < seq; j++)
                s[(int64_t)i * seq + j] += -1.0e9f;
    }
}

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void cpu_randn(int32_t dev_id, float *dst, int64_t n, int64_t seed, float std) {
    (void)dev_id;
    uint64_t st = (uint64_t)seed ^ 0x243f6a8885a308d3ULL;
    int64_t i = 0;
    while (i < n) {
        double u1 = ((double)(splitmix64(&st) >> 11) + 1.0) / 9007199254740993.0;
        double u2 = ((double)(splitmix64(&st) >> 11)) / 9007199254740992.0;
        double r = sqrt(-2.0 * log(u1));
        double phi = 6.28318530717958647692 * u2;
        dst[i++] = (float)(r * cos(phi) * (double)std);
        if (i < n) dst[i++] = (float)(r * sin(phi) * (double)std);
    }
}

static void cpu_adamw_step(
    int32_t dev_id,
    const float *p, const float *g, const float *m, const float *v,
    float *p_out, float *m_out, float *v_out,
    int64_t n, float lr, float beta1, float beta2, float eps, float wd, float bc1, float bc2
) {
    (void)dev_id;
    float one_m_b1 = 1.0f - beta1;
    float one_m_b2 = 1.0f - beta2;
    float decay = 1.0f - lr * wd;
    for (int64_t i = 0; i < n; i++) {
        float gi = g[i];
        float mi = beta1 * m[i] + one_m_b1 * gi;
        float vi = beta2 * v[i] + one_m_b2 * gi * gi;
        m_out[i] = mi;
        v_out[i] = vi;
        p_out[i] = p[i] * decay - lr * ((mi / bc1) / (sqrtf(vi / bc2) + eps));
    }
}

static void cpu_adamw_multi_step(
    int32_t dev_id, int32_t count,
    const float *const *ps, const float *const *gs,
    const float *const *ms, const float *const *vs,
    float *const *p_outs, float *const *m_outs, float *const *v_outs,
    const int64_t *numels,
    float lr, float beta1, float beta2, float eps, float wd, float bc1, float bc2
) {
    for (int32_t t = 0; t < count; t++) {
        cpu_adamw_step(
            dev_id, ps[t], gs[t], ms[t], vs[t],
            p_outs[t], m_outs[t], v_outs[t],
            numels[t], lr, beta1, beta2, eps, wd, bc1, bc2
        );
    }
}

const GsTensorBackend gs_backend_cpu = {
#if defined(__APPLE__) && !defined(GS_TENSOR_NO_ACCELERATE)
    .name = "cpu+accelerate",
#elif defined(GS_TENSOR_OPENBLAS)
    .name = "cpu+openblas",
#else
    .name = "cpu+c11",
#endif
    .dev_kind = GS_DEV_CPU,
    .device_count = cpu_device_count,
    .alloc = cpu_alloc,
    .free_ptr = cpu_free,
    .copy_h2d = cpu_copy_bytes,
    .copy_d2h = cpu_copy_bytes,
    .copy_d2d = cpu_copy_bytes,
    .sgemm = cpu_sgemm,
    .bgemm = cpu_bgemm,
    .fill = cpu_fill,
    .copy = cpu_copy,
    .add = cpu_add,
    .sub = cpu_sub,
    .mul = cpu_mul,
    .div_ew = cpu_div_ew,
    .scale = cpu_scale,
    .add_scalar = cpu_add_scalar,
    .pow_scalar_fwd = cpu_pow_scalar_fwd,
    .pow_scalar_bwd = cpu_pow_scalar_bwd,
    .clamp_fwd = cpu_clamp_fwd,
    .clamp_bwd = cpu_clamp_bwd,
    .leaky_relu_fwd = cpu_leaky_relu_fwd,
    .leaky_relu_bwd = cpu_leaky_relu_bwd,
    .axpy = cpu_axpy,
    .bcast_binop = cpu_bcast_binop,
    .bcast_reduce_sum = cpu_bcast_reduce_sum,
    .unary_fwd = cpu_unary_fwd,
    .unary_bwd = cpu_unary_bwd,
    .slice_axis_fwd = cpu_slice_axis_fwd,
    .slice_axis_bwd = cpu_slice_axis_bwd,
    .sum_axis = cpu_sum_axis,
    .max_axis = cpu_max_axis,
    .sum_all = cpu_sum_all,
    .sum_sq = cpu_sum_sq,
    .rms_norm_fwd = cpu_rms_norm_fwd,
    .rms_norm_bwd = cpu_rms_norm_bwd,
    .layer_norm_fwd = cpu_layer_norm_fwd,
    .layer_norm_bwd = cpu_layer_norm_bwd,
    .l2_norm_last_fwd = cpu_l2_norm_last_fwd,
    .l2_norm_last_bwd = cpu_l2_norm_last_bwd,
    .softmax_fwd = cpu_softmax_fwd,
    .softmax_bwd = cpu_softmax_bwd,
    .log_sum_exp_fwd = cpu_log_sum_exp_fwd,
    .log_sum_exp_bwd = cpu_log_sum_exp_bwd,
    .cross_entropy_fwd = cpu_cross_entropy_fwd,
    .cross_entropy_bwd = cpu_cross_entropy_bwd,
    .embedding_fwd = cpu_embedding_fwd,
    .embedding_bwd = cpu_embedding_bwd,
    .rope_fwd = cpu_rope_fwd,
    .rope_bwd = cpu_rope_bwd,
    .transpose_2d = cpu_transpose_2d,
    .transpose_0213 = cpu_transpose_0213,
    .causal_mask_add = cpu_causal_mask_add,
    .randn = cpu_randn,
    .adamw_step = cpu_adamw_step,
    .adamw_multi_step = cpu_adamw_multi_step,
};
