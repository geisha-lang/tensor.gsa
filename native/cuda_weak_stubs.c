// Weak fallback stubs for `gs_cu_*` symbols when compiling without `nvcc`.
// When `native/cuda.cu` is compiled via `[[target.native.when]] tool = "nvcc"`,
// its strong definitions override these weak symbols at link time.

#include <stdint.h>
#include <stddef.h>

#define WEAK __attribute__((weak))

WEAK int32_t gs_cu_device_count(void) { return 0; }
WEAK void *gs_cu_free_fn(void) { return NULL; }
WEAK void *gs_cu_alloc(int32_t d, int64_t n) { (void)d; (void)n; return NULL; }
WEAK void gs_cu_free(void *p) { (void)p; }
WEAK void gs_cu_sync(int32_t d) { (void)d; }
WEAK void gs_cu_h2d(int32_t d, void *dst, const void *src, int64_t n) { (void)d;(void)dst;(void)src;(void)n; }
WEAK void gs_cu_d2h(int32_t d, void *dst, const void *src, int64_t n) { (void)d;(void)dst;(void)src;(void)n; }
WEAK void gs_cu_d2d(int32_t d, void *dst, const void *src, int64_t n) { (void)d;(void)dst;(void)src;(void)n; }
WEAK void gs_cu_sgemm(int32_t d, int32_t ta, int32_t tb, int32_t M, int32_t N, int32_t K, float a, const float *A, const float *B, float b, float *C) { (void)d;(void)ta;(void)tb;(void)M;(void)N;(void)K;(void)a;(void)A;(void)B;(void)b;(void)C; }
WEAK void gs_cu_bgemm(int32_t d, int32_t batch, int32_t ta, int32_t tb, int32_t M, int32_t N, int32_t K, float a, const float *A, const float *B, float b, float *C) { (void)d;(void)batch;(void)ta;(void)tb;(void)M;(void)N;(void)K;(void)a;(void)A;(void)B;(void)b;(void)C; }
WEAK void gs_cu_fill(int32_t d, float *dst, int64_t n, float v) { (void)d;(void)dst;(void)n;(void)v; }
WEAK void gs_cu_copy(int32_t d, float *dst, const float *src, int64_t n) { (void)d;(void)dst;(void)src;(void)n; }
WEAK void gs_cu_binop_same(int32_t d, int32_t op, const float *a, const float *b, float *o, int64_t n) { (void)d;(void)op;(void)a;(void)b;(void)o;(void)n; }
WEAK void gs_cu_scale(int32_t d, const float *a, float s, float *o, int64_t n) { (void)d;(void)a;(void)s;(void)o;(void)n; }
WEAK void gs_cu_add_s(int32_t d, const float *a, float s, float *o, int64_t n) { (void)d;(void)a;(void)s;(void)o;(void)n; }
WEAK void gs_cu_pow_f(int32_t d, const float *a, float p, float *o, int64_t n) { (void)d;(void)a;(void)p;(void)o;(void)n; }
WEAK void gs_cu_pow_b(int32_t d, const float *a, const float *gy, float p, float *gx, int64_t n) { (void)d;(void)a;(void)gy;(void)p;(void)gx;(void)n; }
WEAK void gs_cu_clamp_f(int32_t d, const float *a, float lo, float hi, float *o, int64_t n) { (void)d;(void)a;(void)lo;(void)hi;(void)o;(void)n; }
WEAK void gs_cu_clamp_b(int32_t d, const float *a, const float *gy, float lo, float hi, float *gx, int64_t n) { (void)d;(void)a;(void)gy;(void)lo;(void)hi;(void)gx;(void)n; }
WEAK void gs_cu_lrelu_f(int32_t d, const float *a, float s, float *o, int64_t n) { (void)d;(void)a;(void)s;(void)o;(void)n; }
WEAK void gs_cu_lrelu_b(int32_t d, const float *a, const float *gy, float s, float *gx, int64_t n) { (void)d;(void)a;(void)gy;(void)s;(void)gx;(void)n; }
WEAK void gs_cu_axpy(int32_t d, float a, const float *x, float *y, int64_t n) { (void)d;(void)a;(void)x;(void)y;(void)n; }
WEAK void gs_cu_bcast_binop(int32_t d, int32_t op, int32_t ndim, int64_t total, const int32_t *sh, const int32_t *sa, const int32_t *sb, const float *a, const float *b, float *o) { (void)d;(void)op;(void)ndim;(void)total;(void)sh;(void)sa;(void)sb;(void)a;(void)b;(void)o; }
WEAK void gs_cu_bcast_reduce_sum(int32_t d, int32_t ndim, int64_t src_total, int64_t dst_total, const int32_t *sh, const int32_t *sd, const float *src, float *dst) { (void)d;(void)ndim;(void)src_total;(void)dst_total;(void)sh;(void)sd;(void)src;(void)dst; }
WEAK void gs_cu_unary_f(int32_t d, int32_t op, const float *x, float *o, int64_t n) { (void)d;(void)op;(void)x;(void)o;(void)n; }
WEAK void gs_cu_unary_b(int32_t d, int32_t op, const float *x, const float *y, const float *gy, float *gx, int64_t n) { (void)d;(void)op;(void)x;(void)y;(void)gy;(void)gx;(void)n; }
WEAK void gs_cu_slice_f(int32_t d, const float *x, float *o, int32_t out0, int32_t id, int32_t in0, int32_t st, int32_t len) { (void)d;(void)x;(void)o;(void)out0;(void)id;(void)in0;(void)st;(void)len; }
WEAK void gs_cu_slice_b(int32_t d, const float *gy, float *gx, int32_t out0, int32_t id, int32_t in0, int32_t st, int32_t len) { (void)d;(void)gy;(void)gx;(void)out0;(void)id;(void)in0;(void)st;(void)len; }
WEAK void gs_cu_sum_ax(int32_t d, const float *x, float *o, int32_t out0, int32_t dim, int32_t in0) { (void)d;(void)x;(void)o;(void)out0;(void)dim;(void)in0; }
WEAK void gs_cu_max_ax(int32_t d, const float *x, float *o, int32_t out0, int32_t dim, int32_t in0) { (void)d;(void)x;(void)o;(void)out0;(void)dim;(void)in0; }
WEAK double gs_cu_sum_all(int32_t d, const float *x, int64_t n) { (void)d;(void)x;(void)n; return 0.0; }
WEAK double gs_cu_sum_sq(int32_t d, const float *x, int64_t n) { (void)d;(void)x;(void)n; return 0.0; }
WEAK void gs_cu_rms_f(int32_t d, const float *x, const float *w, float *o, int32_t r, int32_t dim, float eps) { (void)d;(void)x;(void)w;(void)o;(void)r;(void)dim;(void)eps; }
WEAK void gs_cu_rms_b(int32_t d, const float *x, const float *w, const float *gy, float *gx, float *gw, int32_t r, int32_t dim, float eps) { (void)d;(void)x;(void)w;(void)gy;(void)gx;(void)gw;(void)r;(void)dim;(void)eps; }
WEAK void gs_cu_ln_f(int32_t d, const float *x, const float *w, const float *b, float *o, int32_t r, int32_t dim, float eps) { (void)d;(void)x;(void)w;(void)b;(void)o;(void)r;(void)dim;(void)eps; }
WEAK void gs_cu_ln_b(int32_t d, const float *x, const float *w, const float *gy, float *gx, float *gw, float *gb, int32_t r, int32_t dim, float eps) { (void)d;(void)x;(void)w;(void)gy;(void)gx;(void)gw;(void)gb;(void)r;(void)dim;(void)eps; }
WEAK void gs_cu_l2_f(int32_t d, const float *x, float *o, int32_t r, int32_t dim, float eps) { (void)d;(void)x;(void)o;(void)r;(void)dim;(void)eps; }
WEAK void gs_cu_l2_b(int32_t d, const float *x, const float *gy, float *gx, int32_t r, int32_t dim, float eps) { (void)d;(void)x;(void)gy;(void)gx;(void)r;(void)dim;(void)eps; }
WEAK void gs_cu_sm_f(int32_t d, const float *x, float *o, int32_t r, int32_t c) { (void)d;(void)x;(void)o;(void)r;(void)c; }
WEAK void gs_cu_sm_b(int32_t d, const float *y, const float *gy, float *gx, int32_t r, int32_t c) { (void)d;(void)y;(void)gy;(void)gx;(void)r;(void)c; }
WEAK void gs_cu_lse_f(int32_t d, const float *x, float *o, int32_t r, int32_t c) { (void)d;(void)x;(void)o;(void)r;(void)c; }
WEAK void gs_cu_lse_b(int32_t d, const float *x, const float *lse, const float *gy, float *gx, int32_t r, int32_t c) { (void)d;(void)x;(void)lse;(void)gy;(void)gx;(void)r;(void)c; }
WEAK double gs_cu_ce_f(int32_t d, const float *logits, const int32_t *targets, float *probs_out, int32_t rows, int32_t vocab, int32_t ignore_idx) { (void)d;(void)logits;(void)targets;(void)probs_out;(void)rows;(void)vocab;(void)ignore_idx; return 0.0; }
WEAK void gs_cu_ce_b(int32_t d, const float *probs, const int32_t *targets, float g_loss, float *gl, int32_t rows, int32_t vocab, int32_t ignore_idx) { (void)d;(void)probs;(void)targets;(void)g_loss;(void)gl;(void)rows;(void)vocab;(void)ignore_idx; }
WEAK void gs_cu_emb_f(int32_t d, const float *tab, const int32_t *ids, float *o, int32_t n, int32_t dim) { (void)d;(void)tab;(void)ids;(void)o;(void)n;(void)dim; }
WEAK void gs_cu_emb_b(int32_t d, const int32_t *ids, const float *gy, float *gtab, int32_t n, int32_t v, int32_t dim) { (void)d;(void)ids;(void)gy;(void)gtab;(void)n;(void)v;(void)dim; }
WEAK void gs_cu_rope_f(int32_t d, const float *x, float *o, int32_t b, int32_t s, int32_t h, int32_t hd, float th) { (void)d;(void)x;(void)o;(void)b;(void)s;(void)h;(void)hd;(void)th; }
WEAK void gs_cu_rope_b(int32_t d, const float *gy, float *gx, int32_t b, int32_t s, int32_t h, int32_t hd, float th) { (void)d;(void)gy;(void)gx;(void)b;(void)s;(void)h;(void)hd;(void)th; }
WEAK void gs_cu_t2d(int32_t d, const float *x, float *o, int32_t r, int32_t c) { (void)d;(void)x;(void)o;(void)r;(void)c; }
WEAK void gs_cu_t0213(int32_t d, const float *x, float *o, int32_t d0, int32_t d1, int32_t d2, int32_t d3) { (void)d;(void)x;(void)o;(void)d0;(void)d1;(void)d2;(void)d3; }
WEAK void gs_cu_causal(int32_t d, float *s, int32_t bh, int32_t seq) { (void)d;(void)s;(void)bh;(void)seq; }
WEAK void gs_cu_randn(int32_t d, float *dst, int64_t n, int64_t seed, float std) { (void)d;(void)dst;(void)n;(void)seed;(void)std; }
WEAK void gs_cu_adamw(int32_t d, const float *p, const float *g, const float *m, const float *v, float *po, float *mo, float *vo, int64_t n, float lr, float b1, float b2, float eps, float wd, float bc1, float bc2) { (void)d;(void)p;(void)g;(void)m;(void)v;(void)po;(void)mo;(void)vo;(void)n;(void)lr;(void)b1;(void)b2;(void)eps;(void)wd;(void)bc1;(void)bc2; }
