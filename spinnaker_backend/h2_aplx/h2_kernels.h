#ifndef SNN_TRANSFORM_H2_KERNELS_H
#define SNN_TRANSFORM_H2_KERNELS_H

#include <stdint.h>
#include <stddef.h>

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

int64_t h2_rshift_round_i64(int64_t x, unsigned shift);
uint64_t h2_isqrt_u64(uint64_t x);
int32_t h2_div_scaled_round_i64(int64_t num, int64_t den, unsigned shift);

void h2_dense_linear_q20(
    const int32_t *x, uint32_t n_rows, uint32_t in_features,
    const int32_t *weights, const int32_t *bias,
    uint32_t out_features, int32_t *out);

void h2_sparse_linear_q20(
    const int32_t *x, uint32_t n_rows, uint32_t in_features,
    const uint32_t *row_ptr, const uint16_t *col_idx,
    const int32_t *values, const int32_t *bias,
    uint32_t out_features, int32_t *out);

void h2_layernorm_q20(
    const int32_t *x, uint32_t n_rows, uint32_t width,
    const int32_t *gamma, const int32_t *beta, int32_t *out);

void h2_stable_a_q20(
    const int32_t *x, uint32_t tokens, uint32_t heads, uint32_t head_dim,
    const int32_t *omega, uint32_t features, int32_t scale_q20,
    int is_query, int32_t *out);

void h2_izh_decode_q20(
    const int32_t *a, uint32_t count,
    int32_t bias_q22, int32_t gain_q22, uint32_t window_ms,
    const int32_t *decoder_q20, uint32_t out_frac, int32_t *out);

/*
 * Initial resident H/g from all token K features and V values.
 * pk layout [tokens, heads, features] Q16
 * v  layout [tokens, heads, head_dim] Q20
 * H  layout [heads, features, head_dim] Q20
 * g  layout [heads, features] Q16
 *
 * H1.1 rounds once after summing all token outer products, so this is distinct
 * from repeated h2_attention_token_delta calls.
 */
void h2_attention_state_init(
    const int32_t *pk,
    const int32_t *v,
    uint32_t tokens,
    uint32_t heads,
    uint32_t features,
    uint32_t head_dim,
    int32_t *H,
    int32_t *g);

void h2_attention_token_delta(
    int32_t *H, int32_t *g,
    const int32_t *old_pk, const int32_t *old_v,
    const int32_t *new_pk, const int32_t *new_v,
    uint32_t heads, uint32_t features, uint32_t head_dim);

void h2_attention_read(
    const int32_t *pq, const int32_t *H, const int32_t *g,
    uint32_t tokens, uint32_t heads, uint32_t features,
    uint32_t head_dim, int32_t *attended);

#ifdef __cplusplus
}
#endif

#endif
