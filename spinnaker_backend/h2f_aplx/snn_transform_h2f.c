/*
 * H2F: complete frozen H1.1-functional-v9 resident board execution.
 *
 * Execution contract:
 *   - static HIR + all six input snapshots/masks are preloaded once in SDRAM;
 *   - H,g,pq,pk,v stay resident in DTCM for the complete run;
 *   - no host compute or hidden-state roundtrip occurs between steps;
 *   - only final y,H,g,pq,pk,v are written to the output SDRAM region;
 *   - host compares those final integer arrays bit-for-bit with the H2D oracle.
 *
 * The executed data-spec layout is intentionally decoded locally instead of
 * linking SpiNNFrontEndCommon on target.  The host still uses the official
 * GraphFrontEnd data-spec generator; this loader merely reads the documented
 * metadata pointer placed in VCPU user0.  This keeps H2F dependent only on the
 * same low-level SARK/spin1 toolchain already verified by H2E-board.
 */

#include <sark.h>
#include <stdint.h>
#include <spin1_api.h>

#include "../h2_aplx/h2_kernels.h"
#include "h2f_contract.h"

#define H2F_BOARD_MAGIC 0x48324642u  /* "H2FB" */
#define DSE_MAGIC 0xAD130AD6u
#define DSE_VERSION 0x00010000u
#define H2F_EXPECTED_TRAJECTORY_WORDS 3168u
#define H2F_EXPECTED_OUTPUT_WORDS 4168u

typedef struct h2f_region_desc_s {
    void *pointer;
    uint32_t checksum;
    uint32_t n_words;
} h2f_region_desc_t;

typedef struct h2f_dse_metadata_s {
    uint32_t magic_number;
    uint32_t version;
    h2f_region_desc_t regions[];
} h2f_dse_metadata_t;

static const uint32_t *cfg;
static const uint8_t *hir_blob;
static const uint32_t *trajectory;
static uint32_t *output_region;

static uint32_t tokens, d_model, heads, features, ff_dim, steps, head_dim;
static uint32_t scale_q20;
static uint32_t q_evals, k_evals, v_evals;

/* Resident state + work arena. */
static int32_t *H_state;
static int32_t *g_state;
static int32_t *pq_state;
static int32_t *pk_state;
static int32_t *v_state;
static int32_t *ln_buf;
static int32_t *attended;
static int32_t *z_buf;
static int32_t *hidden;
static int32_t *y_buf;
static int32_t *proj;
static int32_t *a_buf;
static int32_t *new_pq;
static int32_t *new_pk;
static int32_t *new_v;

static inline const void *hir_ptr(h2f_array_id_t id) {
    return (const void *) (hir_blob + h2f_desc_offset(cfg, id));
}

static inline const int32_t *hir_i32(h2f_array_id_t id) {
    return (const int32_t *) hir_ptr(id);
}

static inline const uint32_t *hir_u32(h2f_array_id_t id) {
    return (const uint32_t *) hir_ptr(id);
}

static inline const uint16_t *hir_u16(h2f_array_id_t id) {
    return (const uint16_t *) hir_ptr(id);
}

static void write_final_output(uint32_t rc, uint32_t completed_steps) {
    if (output_region == NULL) return;
    uint32_t *p = output_region;
    p[H2F_OUT_MAGIC] = H2F_OUTPUT_MAGIC;
    p[H2F_OUT_RC] = rc;
    p[H2F_OUT_STEPS] = completed_steps;
    p[H2F_OUT_Q_EVALS] = q_evals;
    p[H2F_OUT_K_EVALS] = k_evals;
    p[H2F_OUT_V_EVALS] = v_evals;
    p[H2F_OUT_CHIP] = sark_chip_id();
    p[H2F_OUT_CORE_APP] = (sark_core_id() << 16) | (sark_app_id() & 0xffffu);

    p += H2F_OUTPUT_HEADER_WORDS;
    const uint32_t y_words = tokens * d_model;
    const uint32_t H_words = heads * features * head_dim;
    const uint32_t g_words = heads * features;
    const uint32_t pq_words = tokens * heads * features;
    const uint32_t pk_words = pq_words;
    const uint32_t v_words = tokens * d_model;
    spin1_memcpy(p, y_buf, y_words * sizeof(uint32_t)); p += y_words;
    spin1_memcpy(p, H_state, H_words * sizeof(uint32_t)); p += H_words;
    spin1_memcpy(p, g_state, g_words * sizeof(uint32_t)); p += g_words;
    spin1_memcpy(p, pq_state, pq_words * sizeof(uint32_t)); p += pq_words;
    spin1_memcpy(p, pk_state, pk_words * sizeof(uint32_t)); p += pk_words;
    spin1_memcpy(p, v_state, v_words * sizeof(uint32_t));
}

static void finish(uint32_t rc, uint32_t completed_steps) {
    write_final_output(rc, completed_steps);
    sark.vcpu->user0 = H2F_BOARD_MAGIC;
    sark.vcpu->user1 = rc;
    sark.vcpu->user2 = completed_steps;
    sark.vcpu->user3 = (sark_core_id() << 16) | (sark_app_id() & 0xffffu);
    if (rc == 0) {
        io_printf(IO_BUF,
            "SNN_TRANSFORM_H2F_FULL_BLOCK PASS steps=%u qkv=%u/%u/%u\n",
            completed_steps, q_evals, k_evals, v_evals);
    } else {
        io_printf(IO_BUF,
            "SNN_TRANSFORM_H2F_FULL_BLOCK FAIL rc=%u steps=%u\n",
            rc, completed_steps);
    }
    spin1_exit(rc);
}

static int validate_contract(void) {
    if (cfg[0] != H2F_MAGIC || cfg[1] != H2F_VERSION) return 10;
    tokens = cfg[2]; d_model = cfg[3]; heads = cfg[4]; features = cfg[5];
    ff_dim = cfg[6]; steps = cfg[7];
    if (tokens != 16 || d_model != 32 || heads != 2 || features != 32 ||
            ff_dim != 64 || steps != 6) return 11;
    if (cfg[8] != H2F_F_ACT || cfg[9] != H2F_F_PQ ||
            cfg[10] != H2F_F_PK || cfg[11] != H2F_F_H ||
            cfg[12] != H2F_F_MEM || cfg[13] != H2F_F_DEC) return 12;
    if (cfg[15] != H2F_N_ARRAYS) return 13;
    head_dim = d_model / heads;
    scale_q20 = cfg[14];
    for (uint32_t i = 0; i < H2F_N_ARRAYS; ++i) {
        if (h2f_desc_nbytes(cfg, (h2f_array_id_t) i) == 0) return 14;
    }
    return 0;
}

static int allocate_arena(void) {
    const uint32_t H_words = heads * features * head_dim;
    const uint32_t g_words = heads * features;
    const uint32_t pq_words = tokens * heads * features;
    const uint32_t pk_words = pq_words;
    const uint32_t v_words = tokens * d_model;
    const uint32_t token_words = d_model;
    const uint32_t feature_words = heads * features;
    const uint32_t arena_words =
        H_words + g_words + pq_words + pk_words + v_words +
        tokens * d_model +  /* ln_buf */
        tokens * d_model +  /* attended */
        tokens * d_model +  /* z_buf */
        tokens * ff_dim +   /* hidden */
        tokens * d_model +  /* y_buf */
        token_words + feature_words + feature_words + feature_words + token_words;

    int32_t *arena = (int32_t *) spin1_malloc(arena_words * sizeof(int32_t));
    if (arena == NULL) return 20;
    int32_t *p = arena;
    H_state = p; p += H_words;
    g_state = p; p += g_words;
    pq_state = p; p += pq_words;
    pk_state = p; p += pk_words;
    v_state = p; p += v_words;
    ln_buf = p; p += tokens * d_model;
    attended = p; p += tokens * d_model;
    z_buf = p; p += tokens * d_model;
    hidden = p; p += tokens * ff_dim;
    y_buf = p; p += tokens * d_model;
    proj = p; p += token_words;
    a_buf = p; p += feature_words;
    new_pq = p; p += feature_words;
    new_pk = p; p += feature_words;
    new_v = p;
    return 0;
}

static void decode_phi(const int32_t *a, int is_query, int32_t *dest) {
    const uint32_t spec_base = is_query ? 16u : 22u;
    const uint32_t out_frac = is_query ? H2F_F_PQ : H2F_F_PK;
    for (uint32_t h = 0; h < heads; ++h) {
        const uint32_t sb = spec_base + 3u * h;
        const int32_t bias_q22 = (int32_t) cfg[sb];
        const int32_t gain_q22 = (int32_t) cfg[sb + 1];
        const uint32_t window = cfg[sb + 2];
        const h2f_array_id_t did = is_query
            ? (h == 0 ? H2F_Q_DECODER_0 : H2F_Q_DECODER_1)
            : (h == 0 ? H2F_K_DECODER_0 : H2F_K_DECODER_1);
        h2_izh_decode_q20(
            a + h * features, features,
            bias_q22, gain_q22, window,
            hir_i32(did), out_frac,
            dest + h * features);
    }
}

static void project_token_features(const int32_t *xn, int32_t *pq_out,
                                   int32_t *pk_out, int32_t *v_out) {
    const int32_t *omega = hir_i32(H2F_OMEGA);

    h2_dense_linear_q20(xn, 1, d_model,
        hir_i32(H2F_WQ), hir_i32(H2F_BQ), d_model, proj);
    h2_stable_a_q20(proj, 1, heads, head_dim, omega, features,
        (int32_t) scale_q20, 1, a_buf);
    decode_phi(a_buf, 1, pq_out);
    ++q_evals;

    h2_dense_linear_q20(xn, 1, d_model,
        hir_i32(H2F_WK), hir_i32(H2F_BK), d_model, proj);
    h2_stable_a_q20(proj, 1, heads, head_dim, omega, features,
        (int32_t) scale_q20, 0, a_buf);
    decode_phi(a_buf, 0, pk_out);
    ++k_evals;

    h2_dense_linear_q20(xn, 1, d_model,
        hir_i32(H2F_WV), hir_i32(H2F_BV), d_model, v_out);
    ++v_evals;
}

static void sparse_layer(const int32_t *x, uint32_t n_rows,
                         uint32_t in_features,
                         h2f_array_id_t row_id, h2f_array_id_t col_id,
                         h2f_array_id_t val_id, h2f_array_id_t bias_id,
                         uint32_t out_features, int32_t *out) {
    h2_sparse_linear_q20(
        x, n_rows, in_features,
        hir_u32(row_id), hir_u16(col_id), hir_i32(val_id), hir_i32(bias_id),
        out_features, out);
}

static int execute_step(uint32_t step) {
    const int32_t *x = (const int32_t *) trajectory + step * tokens * d_model;
    const uint32_t x_words_total = steps * tokens * d_model;
    const uint32_t *masks = trajectory + x_words_total;
    const uint32_t *mask = masks + step * tokens;

    /* H2D computes LN1 for the full snapshot, then event-selects Q/K/V. */
    h2_layernorm_q20(
        x, tokens, d_model, hir_i32(H2F_LN1_G), hir_i32(H2F_LN1_B), ln_buf);

    if (step == 0) {
        for (uint32_t n = 0; n < tokens; ++n) {
            int32_t *pq_t = pq_state + n * heads * features;
            int32_t *pk_t = pk_state + n * heads * features;
            int32_t *v_t = v_state + n * d_model;
            project_token_features(ln_buf + n * d_model, pq_t, pk_t, v_t);
        }
        h2_attention_state_init(
            pk_state, v_state, tokens, heads, features, head_dim,
            H_state, g_state);
    } else {
        for (uint32_t n = 0; n < tokens; ++n) {
            if (mask[n] == 0) continue;
            project_token_features(
                ln_buf + n * d_model, new_pq, new_pk, new_v);
            int32_t *pq_t = pq_state + n * heads * features;
            int32_t *pk_t = pk_state + n * heads * features;
            int32_t *v_t = v_state + n * d_model;
            h2_attention_token_delta(
                H_state, g_state, pk_t, v_t, new_pk, new_v,
                heads, features, head_dim);
            spin1_memcpy(pq_t, new_pq, heads * features * sizeof(int32_t));
            spin1_memcpy(pk_t, new_pk, heads * features * sizeof(int32_t));
            spin1_memcpy(v_t, new_v, d_model * sizeof(int32_t));
        }
    }

    h2_attention_read(
        pq_state, H_state, g_state, tokens, heads, features, head_dim, attended);
    sparse_layer(attended, tokens, d_model,
        H2F_WO_ROW_PTR, H2F_WO_COL_IDX, H2F_WO_VALUES, H2F_WO_BIAS,
        d_model, z_buf);
    for (uint32_t i = 0; i < tokens * d_model; ++i) {
        z_buf[i] = (int32_t) ((int64_t) z_buf[i] + x[i]);
    }

    /* Reuse ln_buf as LN2 output. */
    h2_layernorm_q20(
        z_buf, tokens, d_model, hir_i32(H2F_LN2_G), hir_i32(H2F_LN2_B), ln_buf);
    sparse_layer(ln_buf, tokens, d_model,
        H2F_W1_ROW_PTR, H2F_W1_COL_IDX, H2F_W1_VALUES, H2F_W1_BIAS,
        ff_dim, hidden);
    for (uint32_t i = 0; i < tokens * ff_dim; ++i) {
        if (hidden[i] < 0) hidden[i] = 0;
    }
    sparse_layer(hidden, tokens, ff_dim,
        H2F_W2_ROW_PTR, H2F_W2_COL_IDX, H2F_W2_VALUES, H2F_W2_BIAS,
        d_model, y_buf);
    for (uint32_t i = 0; i < tokens * d_model; ++i) {
        y_buf[i] = (int32_t) ((int64_t) y_buf[i] + z_buf[i]);
    }
    return 0;
}

static void run_full_block(uint unused0, uint unused1) {
    (void) unused0;
    (void) unused1;
    q_evals = k_evals = v_evals = 0;
    for (uint32_t step = 0; step < steps; ++step) {
        int rc = execute_step(step);
        if (rc != 0) {
            finish((uint32_t) rc, step);
            return;
        }
    }
    finish(0, steps);
}

static int initialise(void) {
    /* GraphFrontEnd's data-spec executor places this metadata pointer in VCPU
     * user0 before starting the application.  Read and validate it locally. */
    h2f_dse_metadata_t *data = (h2f_dse_metadata_t *) (uintptr_t) sark.vcpu->user0;
    if (data == NULL || data->magic_number != DSE_MAGIC || data->version != DSE_VERSION) {
        return 30;
    }
    for (uint32_t r = 0; r < 4; ++r) {
        if (data->regions[r].pointer == NULL || data->regions[r].n_words == 0) return 32;
    }
    if (data->regions[H2F_REGION_CONFIG].n_words < H2F_HEADER_WORDS) return 33;
    if (data->regions[H2F_REGION_TRAJECTORY].n_words < H2F_EXPECTED_TRAJECTORY_WORDS) return 34;
    if (data->regions[H2F_REGION_OUTPUT].n_words < H2F_EXPECTED_OUTPUT_WORDS) return 35;

    cfg = (const uint32_t *) data->regions[H2F_REGION_CONFIG].pointer;
    hir_blob = (const uint8_t *) data->regions[H2F_REGION_HIR].pointer;
    trajectory = (const uint32_t *) data->regions[H2F_REGION_TRAJECTORY].pointer;
    output_region = (uint32_t *) data->regions[H2F_REGION_OUTPUT].pointer;
    int rc = validate_contract();
    if (rc != 0) return rc;
    return allocate_arena();
}

void c_main(void) {
    int rc = initialise();
    if (rc != 0) {
        sark.vcpu->user0 = H2F_BOARD_MAGIC;
        sark.vcpu->user1 = (uint32_t) rc;
        io_printf(IO_BUF, "SNN_TRANSFORM_H2F_FULL_BLOCK FAIL init_rc=%d\n", rc);
        return;
    }
    if (!spin1_schedule_callback(run_full_block, 0, 0, 1)) {
        finish(31, 0);
        return;
    }
    spin1_start(SYNC_WAIT);
}
