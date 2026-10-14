#include "h2_kernels.h"

#include <limits.h>

static int32_t sat_i32(int64_t x) {
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

static int64_t round_div_ties_even_i64(int64_t n, uint32_t d) {
    if (d == 0) return 0;
    const int neg = n < 0;
    uint64_t a = (uint64_t)(neg ? -n : n);
    uint64_t q = a / d;
    const uint64_t r = a % d;
    const uint64_t twice = r << 1;
    if (twice > d || (twice == d && (q & 1u))) ++q;
    return neg ? -(int64_t)q : (int64_t)q;
}

int64_t h2_rshift_round_i64(int64_t x, unsigned shift) {
    if (shift == 0) return x;
    if (shift >= 63) return 0;
    const int64_t add = (int64_t)1 << (shift - 1);
    if (x >= 0) return (x + add) >> shift;
    return -(((-x) + add) >> shift);
}

uint64_t h2_isqrt_u64(uint64_t x) {
    uint64_t result = 0;
    uint64_t bit = (uint64_t)1 << 62;
    while (bit > x) bit >>= 2;
    while (bit != 0) {
        if (x >= result + bit) {
            x -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    return result;
}

int32_t h2_div_scaled_round_i64(int64_t num, int64_t den, unsigned shift) {
    if (den <= 0) return 0;

    const int neg = (num < 0);
    uint64_t n = (uint64_t)(neg ? -num : num);
    const uint64_t d = (uint64_t)den;
    uint64_t q = n / d;
    uint64_t r = n % d;

    uint64_t frac = 0;
    for (unsigned i = 0; i < shift; ++i) {
        r <<= 1;
        frac <<= 1;
        if (r >= d) {
            r -= d;
            frac |= 1u;
        }
    }

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

void h2_layernorm_q20(
    const int32_t *x,
    uint32_t n_rows,
    uint32_t width,
    const int32_t *gamma,
    const int32_t *beta,
    int32_t *out) {

    if (width == 0) return;
    for (uint32_t r = 0; r < n_rows; ++r) {
        const int32_t *xr = x + (size_t)r * width;
        int32_t *yr = out + (size_t)r * width;
        int64_t sum = 0;
        for (uint32_t i = 0; i < width; ++i) sum += xr[i];
        const int64_t mean = round_div_ties_even_i64(sum, width);

        uint64_t sq = 0;
        for (uint32_t i = 0; i < width; ++i) {
            const int64_t c = (int64_t)xr[i] - mean;
            sq += (uint64_t)(c * c);
        }
        const uint64_t var_q40 = sq / width + (uint64_t)H2_LN_EPS_Q40;
        const int64_t denom_q20 = (int64_t)h2_isqrt_u64(var_q40 > 0 ? var_q40 : 1);

        for (uint32_t i = 0; i < width; ++i) {
            const int64_t c = (int64_t)xr[i] - mean;
            const int32_t norm_q20 = h2_div_scaled_round_i64(c, denom_q20, H2_F_ACT);
            const int64_t scaled = h2_rshift_round_i64((int64_t)norm_q20 * gamma[i], H2_F_ACT);
            yr[i] = sat_i32(scaled + beta[i]);
        }
    }
}

void h2_stable_a_q20(
    const int32_t *x,
    uint32_t tokens,
    uint32_t heads,
    uint32_t head_dim,
    const int32_t *omega,
    uint32_t features,
    int32_t scale_q20,
    int is_query,
    int32_t *out) {

    const int64_t lo = -((int64_t)30 << H2_F_ACT);
    const int64_t hi = (int64_t)30 << H2_F_ACT;

    for (uint32_t n = 0; n < tokens; ++n) {
        for (uint32_t h = 0; h < heads; ++h) {
            int64_t diag_acc = 0;
            for (uint32_t d = 0; d < head_dim; ++d) {
                const size_t xidx = ((size_t)n * heads + h) * head_dim + d;
                const int64_t xx = h2_rshift_round_i64((int64_t)x[xidx] * scale_q20, H2_F_ACT);
                diag_acc += xx * xx;
            }
            const int64_t diag = h2_rshift_round_i64(diag_acc, H2_F_ACT + 1);

            int64_t maximum = INT64_MIN;
            for (uint32_t m = 0; m < features; ++m) {
                int64_t dash_acc = 0;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    const size_t xidx = ((size_t)n * heads + h) * head_dim + d;
                    const size_t widx = ((size_t)h * features + m) * head_dim + d;
                    const int64_t xx = h2_rshift_round_i64((int64_t)x[xidx] * scale_q20, H2_F_ACT);
                    dash_acc += xx * omega[widx];
                }
                const int64_t dash = h2_rshift_round_i64(dash_acc, H2_F_ACT);
                const int64_t a = dash - diag;
                const size_t oidx = ((size_t)n * heads + h) * features + m;
                out[oidx] = sat_i32(a);
                if (a > maximum) maximum = a;
            }

            for (uint32_t m = 0; m < features; ++m) {
                const size_t oidx = ((size_t)n * heads + h) * features + m;
                int64_t a = out[oidx];
                if (is_query) {
                    a -= maximum;
                    if (a < lo) a = lo;
                    if (a > 0) a = 0;
                } else {
                    if (a < lo) a = lo;
                    if (a > hi) a = hi;
                }
                out[oidx] = (int32_t)a;
            }
        }
    }
}

void h2_izh_decode_q20(
    const int32_t *a,
    uint32_t count,
    int32_t bias_q22,
    int32_t gain_q22,
    uint32_t window_ms,
    const int32_t *decoder_q20,
    uint32_t out_frac,
    int32_t *out) {

    if (out_frac > H2_F_MEM + H2_F_DEC) return;
    const int64_t s_mem = (int64_t)1 << H2_F_MEM;
    const int64_t c02v = 838861;
    const int64_t c004 = 167772;
    const int64_t c02 = 83886;
    const int64_t c140 = 140 * s_mem;
    const unsigned final_shift = H2_F_MEM + H2_F_DEC - out_frac;

    for (uint32_t i = 0; i < count; ++i) {
        const int64_t a_mem = (int64_t)a[i] << (H2_F_MEM - H2_F_ACT);
        int64_t v = -65 * s_mem;
        int64_t u = h2_rshift_round_i64(v * c02v, H2_F_MEM);
        const int64_t current = (int64_t)bias_q22
            + h2_rshift_round_i64(a_mem * gain_q22, H2_F_MEM);
        int64_t acc = (int64_t)decoder_q20[0] << H2_F_MEM;

        for (uint32_t t = 0; t < window_ms; ++t) {
            int64_t v2 = h2_rshift_round_i64(v * v, H2_F_MEM);
            int64_t term004 = h2_rshift_round_i64(v2 * c004, H2_F_MEM);
            int64_t dv = term004 + 5 * v + c140 - u + current;
            v += h2_rshift_round_i64(dv, 1);

            v2 = h2_rshift_round_i64(v * v, H2_F_MEM);
            term004 = h2_rshift_round_i64(v2 * c004, H2_F_MEM);
            dv = term004 + 5 * v + c140 - u + current;
            v += h2_rshift_round_i64(dv, 1);

            const int64_t uv = h2_rshift_round_i64(v * c02v, H2_F_MEM) - u;
            const int64_t du = h2_rshift_round_i64(uv * c02, H2_F_MEM);
            u += du;
            if (v >= 30 * s_mem) {
                v = -65 * s_mem;
                u += 8 * s_mem;
            }
            acc += v * decoder_q20[t + 1];
        }

        int64_t raw = h2_rshift_round_i64(acc, final_shift);
        if (raw < 1) raw = 1;
        out[i] = sat_i32(raw);
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
