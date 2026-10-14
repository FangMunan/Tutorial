#ifndef SNN_TRANSFORM_H2_KERNELS_H
#define SNN_TRANSFORM_H2_KERNELS_H

#include <stdint.h>
#include <stddef.h>

/* Frozen H1.1 functional-v9 numerical contract. */
#define H2_F_ACT 20
#define H2_F_PQ 24
#define H2_F_PK 16
#define H2_F_H 20
#define H2_F_MEM 22
#define H2_F_DEC 20
#define H2_LN_EPS_Q40 10995116LL

#ifdef __cplusplus
extern "C" {
#endif

/* Symmetric round-to-nearest arithmetic right shift. */
int64_t h2_rshift_round_i64(int64_t x, unsigned shift);

/* Integer square root: floor(sqrt(x)). */
uint64_t h2_isqrt_u64(uint64_t x);

/*
 * Compute round((num * 2^shift) / den) without forming num<<shift.
 * The remainder is expanded one bit at a time, avoiding the overflow that
 * would otherwise occur in the normalized H/g read on a 32-bit ARM core.
 * den must be positive.  Saturates to signed int32 on overflow.
 */
int32_t h2_div_scaled_round_i64(int64_t num, int64_t den, unsigned shift);

/* Dense Q*.20 GEMV rows.  x and weights are Q*.20, output is Q*.20. */
void h2_dense_linear_q20(
    const int32_t *x,
    uint32_t n_rows,
    uint32_t in_features,
    const int32_t *weights,
    const int32_t *bias,
    uint32_t out_features,
    int32_t *out);

/* Compact sparse-row Q*.20 linear operator. */
void h2_sparse_linear_q20(
    const int32_t *x,
    uint32_t n_rows,
    uint32_t in_features,
    const uint32_t *row_ptr,
    const uint16_t *col_idx,
    const int32_t *values,
    const int32_t *bias,
    uint32_t out_features,
    int32_t *out);

/* Exact H1.1 integer LayerNorm, Q20 input/gamma/beta -> Q20 output. */
void h2_layernorm_q20(
    const int32_t *x,
    uint32_t n_rows,
    uint32_t width,
    const int32_t *gamma,
    const int32_t *beta,
    int32_t *out);

/*
 * Frozen stable_a projection.
 * x layout:     [tokens, heads, head_dim] Q20
 * omega layout: [heads, features, head_dim] Q20
 * out layout:   [tokens, heads, features] Q20
 * scale_q20 is the offline-compiled head_dim^(-1/4) factor in Q20.
 */
void h2_stable_a_q20(
    const int32_t *x,
    uint32_t tokens,
    uint32_t heads,
    uint32_t head_dim,
    const int32_t *omega,
    uint32_t features,
    int32_t scale_q20,
    int is_query,
    int32_t *out);

/*
 * Fused Izhikevich RS substeps + online trajectory decoder.
 * a is Q20; bias/gain are Q22; decoder is Q20.  out_frac must be <= 30 and
 * is H2_F_PQ (query) or H2_F_PK (key) for the frozen contract.
 */
void h2_izh_decode_q20(
    const int32_t *a,
    uint32_t count,
    int32_t bias_q22,
    int32_t gain_q22,
    uint32_t window_ms,
    const int32_t *decoder_q20,
    uint32_t out_frac,
    int32_t *out);

/*
 * In-place subtract-old/add-new update for one resident sample.
 * Layouts:
 *   H       [heads, features, head_dim] Q(F_H)
 *   g       [heads, features]           Q(F_PK)
 *   pk/v    [heads, features] / [heads, head_dim]
 *            Q(F_PK) / Q(F_ACT)
 * This is exactly one changed-token update; there is no batch dimension and
 * therefore no possible cross-batch broadcast.
 */
void h2_attention_token_delta(
    int32_t *H,
    int32_t *g,
    const int32_t *old_pk,
    const int32_t *old_v,
    const int32_t *new_pk,
    const int32_t *new_v,
    uint32_t heads,
    uint32_t features,
    uint32_t head_dim);

/*
 * Normalized H/g read for one resident sample.
 * pq layout [tokens, heads, features], output [tokens, heads, head_dim].
 */
void h2_attention_read(
    const int32_t *pq,
    const int32_t *H,
    const int32_t *g,
    uint32_t tokens,
    uint32_t heads,
    uint32_t features,
    uint32_t head_dim,
    int32_t *attended);

#ifdef __cplusplus
}
#endif

#endif
