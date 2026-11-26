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
    // Portable cache-tiled SGEMM (works on any CPU without BLAS)
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

static void cpu_scale(int32_t dev_id, const float *a, float s, float *out, int64_t n) {
    (void)dev_id;
#ifdef GS_HAS_VDSP
    vDSP_vsmul(a, 1, &s, out, 1, (vDSP_Length)n);
#else
    for (int64_t i = 0; i < n; i++) out[i] = a[i] * s;
#endif
}

static void cpu_axpy(int32_t dev_id, float alpha, const float *x, float *y, int64_t n) {
    (void)dev_id;
#ifdef GS_HAS_CBLAS
    cblas_saxpy((int)n, alpha, x, 1, y, 1);
#else
    for (int64_t i = 0; i < n; i++) y[i] += alpha * x[i];
#endif
}

static void cpu_add_row_bias(int32_t dev_id, const float *x, const float *bias, float *out, int32_t rows, int32_t cols) {
    (void)dev_id;
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (int64_t)r * cols;
        float *orow = out + (int64_t)r * cols;
        for (int32_t c = 0; c < cols; c++) orow[c] = xr[c] + bias[c];
    }
}

static void cpu_sum_rows(int32_t dev_id, const float *x, float *out, int32_t rows, int32_t cols) {
    (void)dev_id;
    memset(out, 0, (size_t)cols * sizeof(float));
    for (int32_t r = 0; r < rows; r++) {
        const float *xr = x + (int64_t)r * cols;
        for (int32_t c = 0; c < cols; c++) out[c] += xr[c];
    }
}

static void cpu_silu_fwd(int32_t dev_id, const float *x, float *out, int64_t n) {
    (void)dev_id;
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        float sig = 1.0f / (1.0f + expf(-v));
        out[i] = v * sig;
    }
}

static void cpu_silu_bwd(int32_t dev_id, const float *x, const float *gy, float *gx, int64_t n) {
    (void)dev_id;
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        float sig = 1.0f / (1.0f + expf(-v));
        gx[i] = gy[i] * sig * (1.0f + v * (1.0f - sig));
    }
}

static void cpu_gelu_fwd(int32_t dev_id, const float *x, float *out, int64_t n) {
    (void)dev_id;
    const float c = 0.7978845608028654f;
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
}

static void cpu_gelu_bwd(int32_t dev_id, const float *x, const float *gy, float *gx, int64_t n) {
    (void)dev_id;
    const float c = 0.7978845608028654f;
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        float v2 = v * v;
        float th = tanhf(c * (v + 0.044715f * v * v2));
        float sech2 = 1.0f - th * th;
        float dinner = c * (1.0f + 3.0f * 0.044715f * v2);
        gx[i] = gy[i] * (0.5f * (1.0f + th) + 0.5f * v * sech2 * dinner);
    }
}

static void cpu_relu_fwd(int32_t dev_id, const float *x, float *out, int64_t n) {
    (void)dev_id;
    for (int64_t i = 0; i < n; i++) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

static void cpu_relu_bwd(int32_t dev_id, const float *x, const float *gy, float *gx, int64_t n) {
    (void)dev_id;
    for (int64_t i = 0; i < n; i++) gx[i] = x[i] > 0.0f ? gy[i] : 0.0f;
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

static float cpu_cross_entropy_fwd(int32_t dev_id, const float *logits, const int32_t *targets, float *probs_out, int32_t rows, int32_t vocab) {
    (void)dev_id;
    double total_loss = 0.0;
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
        if (tgt >= 0 && tgt < vocab) {
            total_loss -= ((double)(lr[tgt] - mx) - log(sum));
        }
    }
    return rows > 0 ? (float)(total_loss / (double)rows) : 0.0f;
}

static void cpu_cross_entropy_bwd(int32_t dev_id, const float *probs, const int32_t *targets, float g_loss, float *g_logits, int32_t rows, int32_t vocab) {
    (void)dev_id;
    if (rows <= 0) return;
    float scale = g_loss / (float)rows;
    for (int32_t r = 0; r < rows; r++) {
        const float *pr = probs + (int64_t)r * vocab;
        float *gr = g_logits + (int64_t)r * vocab;
        int32_t tgt = targets[r];
        for (int32_t c = 0; c < vocab; c++) gr[c] = pr[c] * scale;
        if (tgt >= 0 && tgt < vocab) gr[tgt] -= scale;
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
    .scale = cpu_scale,
    .axpy = cpu_axpy,
    .add_row_bias = cpu_add_row_bias,
    .sum_rows = cpu_sum_rows,
    .silu_fwd = cpu_silu_fwd,
    .silu_bwd = cpu_silu_bwd,
    .gelu_fwd = cpu_gelu_fwd,
    .gelu_bwd = cpu_gelu_bwd,
    .relu_fwd = cpu_relu_fwd,
    .relu_bwd = cpu_relu_bwd,
    .sum_all = cpu_sum_all,
    .sum_sq = cpu_sum_sq,
    .rms_norm_fwd = cpu_rms_norm_fwd,
    .rms_norm_bwd = cpu_rms_norm_bwd,
    .softmax_fwd = cpu_softmax_fwd,
    .softmax_bwd = cpu_softmax_bwd,
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
};
