#include "h2_kernels.h"

#include <limits.h>

static int32_t sat_i32(int64_t x) {
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

int64_t h2_rshift_round_i64(int64_t x, unsigned shift) {
    if (shift == 0) return x;
    if (shift >= 63) return 0;
    const int64_t add = (int64_t)1 << (shift - 1);
    if (x >= 0) return (x + add) >> shift;
    return -(((-x) + add) >> shift);
}

int32_t h2_div_scaled_round_i64(int64_t num, int64_t den, unsigned shift) {
    if (den <= 0) return 0;

    int neg = (num < 0);
    uint64_t n = (uint64_t)(neg ? -num : num);
    uint64_t d = (uint64_t)den;
    uint64_t q = n / d;
    uint64_t r = n % d;

    /* Build fractional bits without ever evaluating r<<shift in one step. */
    uint64_t frac = 0;
    for (unsigned i = 0; i < shift; ++i) {
        r <<= 1;
        frac <<= 1;
        if (r >= d) {
            r -= d;
            frac |= 1u;
        }
    }

    /* Round the final binary fraction to nearest, ties away from zero. */
    if (r >= ((d + 1u) >> 1)) {
        ++frac;
        if (shift < 63 && frac == ((uint64_t)1 << shift)) {
            frac = 0;
            ++q;
        }
    }

    uint64_t mag;
    if (shift >= 63 || q > ((uint64_t)INT32_MAX >> shift)) {
        mag = (uint64_t)INT32_MAX + (neg ? 1u : 0u);
    } else {
        mag = (q << shift) + frac;
    }

    if (!neg) {
        return mag > (uint64_t)INT32_MAX ? INT32_MAX : (int32_t)mag;
    }
    if (mag >= (uint64_t)INT32_MAX + 1u) return INT32_MIN;
    return -(int32_t)mag;
}

void h2_dense_linear_q20(
    const int32_t *x,
    uint32_t n_rows,
    uint32_t in_features,
    const int32_t *weights,
    const int32_t *bias,
    uint32_t out_features,
    int32_t *out) {

    for (uint32_t r = 0; r < n_rows; ++r) {
        const int32_t *xr = x + (size_t)r * in_features;
        int32_t *yr = out + (size_t)r * out_features;
        for (uint32_t o = 0; o < out_features; ++o) {
            const int32_t *wr = weights + (size_t)o * in_features;
            int64_t acc = 0;
            for (uint32_t i = 0; i < in_features; ++i) {
                acc += (int64_t)xr[i] * (int64_t)wr[i];
            }
            yr[o] = sat_i32(h2_rshift_round_i64(acc, H2_F_ACT) + bias[o]);
        }
    }
}

void h2_sparse_linear_q20(
    const int32_t *x,
    uint32_t n_rows,
    uint32_t in_features,
    const uint32_t *row_ptr,
    const uint16_t *col_idx,
    const int32_t *values,
    const int32_t *bias,
    uint32_t out_features,
    int32_t *out) {

    for (uint32_t r = 0; r < n_rows; ++r) {
        const int32_t *xr = x + (size_t)r * in_features;
        int32_t *yr = out + (size_t)r * out_features;
        for (uint32_t o = 0; o < out_features; ++o) {
            int64_t acc = 0;
            for (uint32_t p = row_ptr[o]; p < row_ptr[o + 1]; ++p) {
                acc += (int64_t)xr[col_idx[p]] * (int64_t)values[p];
            }
            yr[o] = sat_i32(h2_rshift_round_i64(acc, H2_F_ACT) + bias[o]);
        }
    }
}

void h2_attention_token_delta(
    int32_t *H,
    int32_t *g,
    const int32_t *old_pk,
    const int32_t *old_v,
    const int32_t *new_pk,
    const int32_t *new_v,
    uint32_t heads,
    uint32_t features,
    uint32_t head_dim) {

    const unsigned outer_shift = H2_F_PK + H2_F_ACT - H2_F_H;
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t m = 0; m < features; ++m) {
            const size_t pm = (size_t)h * features + m;
            g[pm] = sat_i32((int64_t)g[pm] - old_pk[pm] + new_pk[pm]);
            for (uint32_t d = 0; d < head_dim; ++d) {
                const size_t hd = (size_t)h * head_dim + d;
                const size_t hm = ((size_t)h * features + m) * head_dim + d;
                const int64_t old_outer = h2_rshift_round_i64(
                    (int64_t)old_pk[pm] * old_v[hd], outer_shift);
                const int64_t new_outer = h2_rshift_round_i64(
                    (int64_t)new_pk[pm] * new_v[hd], outer_shift);
                H[hm] = sat_i32((int64_t)H[hm] - old_outer + new_outer);
            }
        }
    }
}

void h2_attention_read(
    const int32_t *pq,
    const int32_t *H,
    const int32_t *g,
    uint32_t tokens,
    uint32_t heads,
    uint32_t features,
    uint32_t head_dim,
    int32_t *attended) {

    const unsigned quotient_shift = H2_F_ACT + H2_F_PK - H2_F_H;
    for (uint32_t n = 0; n < tokens; ++n) {
        for (uint32_t h = 0; h < heads; ++h) {
            int64_t den = 0;
            for (uint32_t m = 0; m < features; ++m) {
                const size_t qidx = ((size_t)n * heads + h) * features + m;
                const size_t gidx = (size_t)h * features + m;
                den += (int64_t)pq[qidx] * g[gidx];
            }
            for (uint32_t d = 0; d < head_dim; ++d) {
                int64_t num = 0;
                for (uint32_t m = 0; m < features; ++m) {
                    const size_t qidx = ((size_t)n * heads + h) * features + m;
                    const size_t hidx = ((size_t)h * features + m) * head_dim + d;
                    num += (int64_t)pq[qidx] * H[hidx];
                }
                const size_t out_idx = ((size_t)n * heads + h) * head_dim + d;
                attended[out_idx] = h2_div_scaled_round_i64(num, den, quotient_shift);
            }
        }
    }
}
