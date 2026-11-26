// Minimal native shim for tensor.gsa:
// 1. Vendor BLAS SGEMM / Batched SGEMM (Apple Accelerate AMX / OpenBLAS / C fallback)
// 2. Thin binary file I/O helpers for pure-Geisha Safetensors
// 3. Optional CUDA backend bridge (when compiled with `nvcc` / `-DGS_TENSOR_CUDA`)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__) && !defined(GS_TENSOR_NO_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define GS_HAS_CBLAS 1
#elif defined(GS_TENSOR_OPENBLAS)
#include <cblas.h>
#define GS_HAS_CBLAS 1
#endif

#ifdef GS_TENSOR_CUDA
#include "backend.h"
#endif

// ============================================================================
// CPU BLAS SGEMM & Batched SGEMM
// ============================================================================

void gs_t_sgemm_cpu(
    int32_t transA, int32_t transB,
    int32_t M, int32_t N, int32_t K,
    float alpha, const float *A, const float *B,
    float beta, float *C
) {
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

void gs_t_bgemm_cpu(
    int32_t batch, int32_t transA, int32_t transB,
    int32_t M, int32_t N, int32_t K,
    float alpha, const float *A, const float *B,
    float beta, float *C
) {
    int64_t strideA = (int64_t)M * K;
    int64_t strideB = (int64_t)K * N;
    int64_t strideC = (int64_t)M * N;
    for (int32_t b = 0; b < batch; b++) {
        gs_t_sgemm_cpu(
            transA, transB, M, N, K, alpha,
            A + (int64_t)b * strideA,
            B + (int64_t)b * strideB,
            beta,
            C + (int64_t)b * strideC
        );
    }
}

int32_t gs_t_cuda_device_count(void) {
#ifdef GS_TENSOR_CUDA
    return gs_backend_cuda.device_count();
#else
    return 0;
#endif
}

// ============================================================================
// Binary File I/O Helpers for Pure-Geisha Safetensors
// ============================================================================

void *gs_t_fopen_wb(const char *path) { return (void *)fopen(path, "wb"); }
void *gs_t_fopen_rb(const char *path) { return (void *)fopen(path, "rb"); }
int32_t gs_t_fclose(void *fp) { return fp ? fclose((FILE *)fp) : -1; }

int64_t gs_t_fwrite_u64_le(void *fp, int64_t v) {
    uint8_t buf[8];
    uint64_t u = (uint64_t)v;
    for (int i = 0; i < 8; i++) buf[i] = (uint8_t)((u >> (8 * i)) & 0xffU);
    return (int64_t)fwrite(buf, 1, 8, (FILE *)fp);
}

int64_t gs_t_fread_u64_le(void *fp) {
    uint8_t buf[8];
    if (fread(buf, 1, 8, (FILE *)fp) != 8) return -1;
    uint64_t u = 0;
    for (int i = 0; i < 8; i++) u |= ((uint64_t)buf[i]) << (8 * i);
    return (int64_t)u;
}

int64_t gs_t_fwrite_bytes(void *fp, const void *ptr, int64_t n) { return n > 0 ? (int64_t)fwrite(ptr, 1, (size_t)n, (FILE *)fp) : 0; }
int64_t gs_t_fread_bytes(void *fp, void *ptr, int64_t n) { return n > 0 ? (int64_t)fread(ptr, 1, (size_t)n, (FILE *)fp) : 0; }
int64_t gs_t_fwrite_cstr(void *fp, const char *s, int64_t n) { return n > 0 ? (int64_t)fwrite(s, 1, (size_t)n, (FILE *)fp) : 0; }
