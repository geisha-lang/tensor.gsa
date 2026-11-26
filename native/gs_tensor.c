#include "gs_tensor_backend.h"

#include <stdio.h>
#include <stdlib.h>

static const GsTensorBackend *backend_for(int32_t dev_kind) {
#ifdef GS_TENSOR_CUDA
    if (dev_kind == GS_DEV_CUDA) return &gs_backend_cuda;
#endif
    (void)dev_kind;
    return &gs_backend_cpu;
}

int32_t gs_t_cuda_device_count(void) {
#ifdef GS_TENSOR_CUDA
    return gs_backend_cuda.device_count();
#else
    return 0;
#endif
}

const char *gs_t_backend_name(int32_t dev_kind) {
    return backend_for(dev_kind)->name;
}

void *gs_t_alloc(int32_t dev_kind, int32_t dev_id, int64_t nbytes) {
    return backend_for(dev_kind)->alloc(dev_id, nbytes);
}

void *gs_t_free_fn(int32_t dev_kind) {
    return (void *)backend_for(dev_kind)->free_ptr;
}

void gs_t_memcpy_dev(
    void *dst, int32_t dst_kind, int32_t dst_id,
    const void *src, int32_t src_kind, int32_t src_id,
    int64_t nbytes
) {
    if (nbytes <= 0) return;
    if (dst_kind == GS_DEV_CPU && src_kind == GS_DEV_CPU) {
        gs_backend_cpu.copy_d2d(0, dst, src, nbytes);
    } else if (src_kind == GS_DEV_CPU) {
        backend_for(dst_kind)->copy_h2d(dst_id, dst, src, nbytes);
    } else if (dst_kind == GS_DEV_CPU) {
        backend_for(src_kind)->copy_d2h(src_id, dst, src, nbytes);
    } else if (dst_kind == src_kind && dst_id == src_id) {
        backend_for(dst_kind)->copy_d2d(dst_id, dst, src, nbytes);
    } else {
        void *tmp = malloc((size_t)nbytes);
        if (!tmp) return;
        backend_for(src_kind)->copy_d2h(src_id, tmp, src, nbytes);
        backend_for(dst_kind)->copy_h2d(dst_id, dst, tmp, nbytes);
        free(tmp);
    }
}

void gs_t_sgemm(int32_t dk, int32_t did, int32_t ta, int32_t tb, int32_t M, int32_t N, int32_t K, float alpha, const float *A, const float *B, float beta, float *C) {
    backend_for(dk)->sgemm(did, ta, tb, M, N, K, alpha, A, B, beta, C);
}
void gs_t_bgemm(int32_t dk, int32_t did, int32_t batch, int32_t ta, int32_t tb, int32_t M, int32_t N, int32_t K, float alpha, const float *A, const float *B, float beta, float *C) {
    backend_for(dk)->bgemm(did, batch, ta, tb, M, N, K, alpha, A, B, beta, C);
}
void gs_t_fill(int32_t dk, int32_t did, float *dst, int64_t n, float val) { backend_for(dk)->fill(did, dst, n, val); }
void gs_t_copy(int32_t dk, int32_t did, float *dst, const float *src, int64_t n) { backend_for(dk)->copy(did, dst, src, n); }
void gs_t_add(int32_t dk, int32_t did, const float *a, const float *b, float *out, int64_t n) { backend_for(dk)->add(did, a, b, out, n); }
void gs_t_sub(int32_t dk, int32_t did, const float *a, const float *b, float *out, int64_t n) { backend_for(dk)->sub(did, a, b, out, n); }
void gs_t_mul(int32_t dk, int32_t did, const float *a, const float *b, float *out, int64_t n) { backend_for(dk)->mul(did, a, b, out, n); }
void gs_t_scale(int32_t dk, int32_t did, const float *a, float s, float *out, int64_t n) { backend_for(dk)->scale(did, a, s, out, n); }
void gs_t_axpy(int32_t dk, int32_t did, float alpha, const float *x, float *y, int64_t n) { backend_for(dk)->axpy(did, alpha, x, y, n); }
void gs_t_add_row_bias(int32_t dk, int32_t did, const float *x, const float *bias, float *out, int32_t rows, int32_t cols) { backend_for(dk)->add_row_bias(did, x, bias, out, rows, cols); }
void gs_t_sum_rows(int32_t dk, int32_t did, const float *x, float *out, int32_t rows, int32_t cols) { backend_for(dk)->sum_rows(did, x, out, rows, cols); }
void gs_t_silu_fwd(int32_t dk, int32_t did, const float *x, float *out, int64_t n) { backend_for(dk)->silu_fwd(did, x, out, n); }
void gs_t_silu_bwd(int32_t dk, int32_t did, const float *x, const float *gy, float *gx, int64_t n) { backend_for(dk)->silu_bwd(did, x, gy, gx, n); }
void gs_t_gelu_fwd(int32_t dk, int32_t did, const float *x, float *out, int64_t n) { backend_for(dk)->gelu_fwd(did, x, out, n); }
void gs_t_gelu_bwd(int32_t dk, int32_t did, const float *x, const float *gy, float *gx, int64_t n) { backend_for(dk)->gelu_bwd(did, x, gy, gx, n); }
void gs_t_relu_fwd(int32_t dk, int32_t did, const float *x, float *out, int64_t n) { backend_for(dk)->relu_fwd(did, x, out, n); }
void gs_t_relu_bwd(int32_t dk, int32_t did, const float *x, const float *gy, float *gx, int64_t n) { backend_for(dk)->relu_bwd(did, x, gy, gx, n); }
float gs_t_sum_all(int32_t dk, int32_t did, const float *x, int64_t n) { return backend_for(dk)->sum_all(did, x, n); }
float gs_t_sum_sq(int32_t dk, int32_t did, const float *x, int64_t n) { return backend_for(dk)->sum_sq(did, x, n); }
void gs_t_rms_norm_fwd(int32_t dk, int32_t did, const float *x, const float *w, float *out, int32_t rows, int32_t dim, float eps) { backend_for(dk)->rms_norm_fwd(did, x, w, out, rows, dim, eps); }
void gs_t_rms_norm_bwd(int32_t dk, int32_t did, const float *x, const float *w, const float *gy, float *gx, float *gw, int32_t rows, int32_t dim, float eps) { backend_for(dk)->rms_norm_bwd(did, x, w, gy, gx, gw, rows, dim, eps); }
void gs_t_softmax_fwd(int32_t dk, int32_t did, const float *x, float *out, int32_t rows, int32_t cols) { backend_for(dk)->softmax_fwd(did, x, out, rows, cols); }
void gs_t_softmax_bwd(int32_t dk, int32_t did, const float *y, const float *gy, float *gx, int32_t rows, int32_t cols) { backend_for(dk)->softmax_bwd(did, y, gy, gx, rows, cols); }
float gs_t_cross_entropy_fwd(int32_t dk, int32_t did, const float *logits, const int32_t *targets, float *probs_out, int32_t rows, int32_t vocab) { return backend_for(dk)->cross_entropy_fwd(did, logits, targets, probs_out, rows, vocab); }
void gs_t_cross_entropy_bwd(int32_t dk, int32_t did, const float *probs, const int32_t *targets, float g_loss, float *g_logits, int32_t rows, int32_t vocab) { backend_for(dk)->cross_entropy_bwd(did, probs, targets, g_loss, g_logits, rows, vocab); }
void gs_t_embedding_fwd(int32_t dk, int32_t did, const float *table, const int32_t *ids, float *out, int32_t n_ids, int32_t dim) { backend_for(dk)->embedding_fwd(did, table, ids, out, n_ids, dim); }
void gs_t_embedding_bwd(int32_t dk, int32_t did, const int32_t *ids, const float *gy, float *g_table, int32_t n_ids, int32_t vocab, int32_t dim) { backend_for(dk)->embedding_bwd(did, ids, gy, g_table, n_ids, vocab, dim); }
void gs_t_rope_fwd(int32_t dk, int32_t did, const float *x, float *out, int32_t batch, int32_t seq, int32_t heads, int32_t head_dim, float theta) { backend_for(dk)->rope_fwd(did, x, out, batch, seq, heads, head_dim, theta); }
void gs_t_rope_bwd(int32_t dk, int32_t did, const float *gy, float *gx, int32_t batch, int32_t seq, int32_t heads, int32_t head_dim, float theta) { backend_for(dk)->rope_bwd(did, gy, gx, batch, seq, heads, head_dim, theta); }
void gs_t_transpose_2d(int32_t dk, int32_t did, const float *x, float *out, int32_t rows, int32_t cols) { backend_for(dk)->transpose_2d(did, x, out, rows, cols); }
void gs_t_transpose_0213(int32_t dk, int32_t did, const float *x, float *out, int32_t d0, int32_t d1, int32_t d2, int32_t d3) { backend_for(dk)->transpose_0213(did, x, out, d0, d1, d2, d3); }
void gs_t_causal_mask_add(int32_t dk, int32_t did, float *scores, int32_t bh, int32_t seq) { backend_for(dk)->causal_mask_add(did, scores, bh, seq); }
void gs_t_randn(int32_t dk, int32_t did, float *dst, int64_t n, int64_t seed, float std) { backend_for(dk)->randn(did, dst, n, seed, std); }
void gs_t_adamw_step(int32_t dk, int32_t did, const float *p, const float *g, const float *m, const float *v, float *p_out, float *m_out, float *v_out, int64_t n, float lr, float beta1, float beta2, float eps, float wd, float bc1, float bc2) {
    backend_for(dk)->adamw_step(did, p, g, m, v, p_out, m_out, v_out, n, lr, beta1, beta2, eps, wd, bc1, bc2);
}

// ============================================================================
// Binary File Helpers for Pure-Geisha Safetensors
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

int64_t gs_t_fwrite_bytes(void *fp, const void *ptr, int64_t n) {
    if (n <= 0) return 0;
    return (int64_t)fwrite(ptr, 1, (size_t)n, (FILE *)fp);
}

int64_t gs_t_fread_bytes(void *fp, void *ptr, int64_t n) {
    if (n <= 0) return 0;
    return (int64_t)fread(ptr, 1, (size_t)n, (FILE *)fp);
}

int32_t gs_t_fseek_set(void *fp, int64_t offset) {
    return fseek((FILE *)fp, (long)offset, SEEK_SET);
}

int64_t gs_t_fwrite_cstr(void *fp, const char *s, int64_t n) {
    if (n <= 0) return 0;
    return (int64_t)fwrite(s, 1, (size_t)n, (FILE *)fp);
}
