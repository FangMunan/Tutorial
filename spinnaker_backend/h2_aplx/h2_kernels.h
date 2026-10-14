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

#ifdef __cplusplus
extern "C" {
#endif

/* Symmetric round-to-nearest arithmetic right shift. */
int64_t h2_rshift_round_i64(int64_t x, unsigned shift);

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
