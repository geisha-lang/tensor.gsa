#ifndef GS_TENSOR_BACKEND_H
#define GS_TENSOR_BACKEND_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device kinds matching Geisha `Device` ADT:
// 0 = Cpu, 1 = Cuda(id), 2 = Metal(id)
#define GS_DEV_CPU   0
#define GS_DEV_CUDA  1
#define GS_DEV_METAL 2

typedef struct GsTensorBackend {
    const char *name;
    int32_t dev_kind;

    int32_t (*device_count)(void);
    void   *(*alloc)(int32_t dev_id, int64_t nbytes);
    void    (*free_ptr)(void *ptr);
    void    (*copy_h2d)(int32_t dev_id, void *dst_dev, const void *src_host, int64_t nbytes);
    void    (*copy_d2h)(int32_t dev_id, void *dst_host, const void *src_dev, int64_t nbytes);
    void    (*copy_d2d)(int32_t dev_id, void *dst_dev, const void *src_dev, int64_t nbytes);

    void (*sgemm)(
        int32_t dev_id, int32_t transA, int32_t transB,
        int32_t M, int32_t N, int32_t K,
        float alpha, const float *A, const float *B,
        float beta, float *C
    );
    void (*bgemm)(
        int32_t dev_id, int32_t batch, int32_t transA, int32_t transB,
        int32_t M, int32_t N, int32_t K,
        float alpha, const float *A, const float *B,
        float beta, float *C
    );

    void  (*fill)(int32_t dev_id, float *dst, int64_t n, float val);
    void  (*copy)(int32_t dev_id, float *dst, const float *src, int64_t n);
    void  (*add)(int32_t dev_id, const float *a, const float *b, float *out, int64_t n);
    void  (*sub)(int32_t dev_id, const float *a, const float *b, float *out, int64_t n);
    void  (*mul)(int32_t dev_id, const float *a, const float *b, float *out, int64_t n);
    void  (*scale)(int32_t dev_id, const float *a, float s, float *out, int64_t n);
    void  (*axpy)(int32_t dev_id, float alpha, const float *x, float *y, int64_t n);
    void  (*add_row_bias)(int32_t dev_id, const float *x, const float *bias, float *out, int32_t rows, int32_t cols);
    void  (*sum_rows)(int32_t dev_id, const float *x, float *out, int32_t rows, int32_t cols);

    void  (*silu_fwd)(int32_t dev_id, const float *x, float *out, int64_t n);
    void  (*silu_bwd)(int32_t dev_id, const float *x, const float *gy, float *gx, int64_t n);
    void  (*gelu_fwd)(int32_t dev_id, const float *x, float *out, int64_t n);
    void  (*gelu_bwd)(int32_t dev_id, const float *x, const float *gy, float *gx, int64_t n);
    void  (*relu_fwd)(int32_t dev_id, const float *x, float *out, int64_t n);
    void  (*relu_bwd)(int32_t dev_id, const float *x, const float *gy, float *gx, int64_t n);
    float (*sum_all)(int32_t dev_id, const float *x, int64_t n);
    float (*sum_sq)(int32_t dev_id, const float *x, int64_t n);

    void  (*rms_norm_fwd)(int32_t dev_id, const float *x, const float *w, float *out, int32_t rows, int32_t dim, float eps);
    void  (*rms_norm_bwd)(int32_t dev_id, const float *x, const float *w, const float *gy, float *gx, float *gw, int32_t rows, int32_t dim, float eps);
    void  (*softmax_fwd)(int32_t dev_id, const float *x, float *out, int32_t rows, int32_t cols);
    void  (*softmax_bwd)(int32_t dev_id, const float *y, const float *gy, float *gx, int32_t rows, int32_t cols);
    float (*cross_entropy_fwd)(int32_t dev_id, const float *logits, const int32_t *targets, float *probs_out, int32_t rows, int32_t vocab);
    void  (*cross_entropy_bwd)(int32_t dev_id, const float *probs, const int32_t *targets, float g_loss, float *g_logits, int32_t rows, int32_t vocab);
    void  (*embedding_fwd)(int32_t dev_id, const float *table, const int32_t *ids, float *out, int32_t n_ids, int32_t dim);
    void  (*embedding_bwd)(int32_t dev_id, const int32_t *ids, const float *gy, float *g_table, int32_t n_ids, int32_t vocab, int32_t dim);
    void  (*rope_fwd)(int32_t dev_id, const float *x, float *out, int32_t batch, int32_t seq, int32_t heads, int32_t head_dim, float theta);
    void  (*rope_bwd)(int32_t dev_id, const float *gy, float *gx, int32_t batch, int32_t seq, int32_t heads, int32_t head_dim, float theta);
    void  (*transpose_2d)(int32_t dev_id, const float *x, float *out, int32_t rows, int32_t cols);
    void  (*transpose_0213)(int32_t dev_id, const float *x, float *out, int32_t d0, int32_t d1, int32_t d2, int32_t d3);
    void  (*causal_mask_add)(int32_t dev_id, float *scores, int32_t bh, int32_t seq);

    void  (*randn)(int32_t dev_id, float *dst, int64_t n, int64_t seed, float std);
    void  (*adamw_step)(
        int32_t dev_id,
        const float *p, const float *g, const float *m, const float *v,
        float *p_out, float *m_out, float *v_out,
        int64_t n,
        float lr, float beta1, float beta2, float eps, float wd,
        float bc1, float bc2
    );
} GsTensorBackend;

extern const GsTensorBackend gs_backend_cpu;
#ifdef GS_TENSOR_CUDA
extern const GsTensorBackend gs_backend_cuda;
#endif

#ifdef __cplusplus
}
#endif

#endif // GS_TENSOR_BACKEND_H
