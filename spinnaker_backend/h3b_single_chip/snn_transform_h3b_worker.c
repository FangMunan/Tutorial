/*
 * H3B0: one frozen SNNTransformBlock distributed over several ARM cores on
 * one SpiNNaker chip.
 *
 * Every core holds a contiguous local token shard (pq/pk/v) and a replicated
 * global H/g.  Rank 0 coordinates an on-chip multicast gather/broadcast.  No
 * host arithmetic or hidden-state roundtrip occurs while the six-step block is
 * running.
 */

#include <sark.h>
#include <stdint.h>
#include <spin1_api.h>

#include "../h2_aplx/h2_kernels.h"
#include "h3b_contract.h"

#define DSE_MAGIC 0xAD130AD6u
#define DSE_VERSION 0x00010000u
#define H3B_RC_ALLOC 20u
#define H3B_RC_PROTOCOL 40u
#define H3B_RC_PACKET_COUNT 41u
#define H3B_RC_BAD_SENDER 42u

#define H3B_STAGE_PREP 1u
#define H3B_STAGE_GATHER 2u
#define H3B_STAGE_WAIT_BCAST 3u
#define H3B_STAGE_BCAST 4u
#define H3B_STAGE_TAIL 5u

/* DSE metadata is the same compact executed-data-spec table used by H2F/H3A. */
typedef struct h3b_region_desc_s {
    void *pointer;
    uint32_t checksum;
    uint32_t n_words;
} h3b_region_desc_t;

typedef struct h3b_dse_metadata_s {
    uint32_t magic_number;
    uint32_t version;
    h3b_region_desc_t regions[];
} h3b_dse_metadata_t;

static const uint32_t *cfg;
static const uint8_t *hir_blob;
static const uint32_t *trajectory;
static uint32_t *output_region;

static uint32_t local_tokens, global_tokens, d_model, heads, features, ff_dim;
static uint32_t steps, head_dim, scale_q20;
static uint32_t rank_id, n_workers, send_key, pacing_us;
static uint32_t peer_keys[H3B_MAX_WORKERS];

static uint32_t q_evals, k_evals, v_evals;
static uint32_t tx_packets, rx_packets, tx_retries, protocol_errors;
static uint32_t ready_events, gather_phases, bcast_phases;
static uint32_t elapsed_ms, arena_bytes_requested, arena_allocated;
static uint32_t current_step, current_stage, active_sender;
static uint32_t ready_mask, ready_count, recv_words, bcast_words_received;
static uint32_t recv_low_word;
static uint32_t start_ms;
static uint32_t final_rc;

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

/* Reduction workspaces reuse buffers that are dead before attention read. */
static int64_t *reduce_H64;
static int64_t *reduce_g64;

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

static uint32_t H_words(void) { return heads * features * head_dim; }
static uint32_t g_words(void) { return heads * features; }
static uint32_t gather_payload_words(void) {
    return 2u * (H_words() + g_words());
}
static uint32_t hg_payload_words(void) { return H_words() + g_words(); }
static uint32_t expected_local_trajectory_words(void) {
    return steps * local_tokens * d_model + steps * local_tokens;
}
static uint32_t expected_output_words(void) {
    return H3B_OUTPUT_HEADER_WORDS
        + local_tokens * d_model
        + H_words() + g_words()
        + local_tokens * heads * features
        + local_tokens * heads * features
        + local_tokens * d_model;
}

static void write_output(uint32_t rc, uint32_t completed_steps) {
    if (output_region == NULL) return;
    uint32_t *p = output_region;
    p[H3B_OUT_MAGIC] = H3B_OUTPUT_MAGIC;
    p[H3B_OUT_RC] = rc;
    p[H3B_OUT_STEPS] = completed_steps;
    p[H3B_OUT_RANK] = rank_id;
    p[H3B_OUT_N_WORKERS] = n_workers;
    p[H3B_OUT_GLOBAL_TOKENS] = global_tokens;
    p[H3B_OUT_LOCAL_TOKENS] = local_tokens;
    p[H3B_OUT_Q_EVALS] = q_evals;
    p[H3B_OUT_K_EVALS] = k_evals;
    p[H3B_OUT_V_EVALS] = v_evals;
    p[H3B_OUT_TX_PACKETS] = tx_packets;
    p[H3B_OUT_RX_PACKETS] = rx_packets;
    p[H3B_OUT_TX_RETRIES] = tx_retries;
    p[H3B_OUT_PROTOCOL_ERRORS] = protocol_errors;
    p[H3B_OUT_ELAPSED_MS] = elapsed_ms;
    p[H3B_OUT_CHIP] = sark_chip_id();
    p[H3B_OUT_CORE_APP] = (sark_core_id() << 16) | (sark_app_id() & 0xffffu);
    p[H3B_OUT_READY_EVENTS] = ready_events;
    p[H3B_OUT_GATHER_PHASES] = gather_phases;
    p[H3B_OUT_BCAST_PHASES] = bcast_phases;
    p[H3B_OUT_ARENA_BYTES] = arena_bytes_requested;
    p[H3B_OUT_RESERVED0] = 0;
    p[H3B_OUT_RESERVED1] = 0;
    p[H3B_OUT_RESERVED2] = 0;

    if (rc != 0 || !arena_allocated) return;
    p += H3B_OUTPUT_HEADER_WORDS;
    spin1_memcpy(p, y_buf, local_tokens * d_model * sizeof(uint32_t));
    p += local_tokens * d_model;
    spin1_memcpy(p, H_state, H_words() * sizeof(uint32_t)); p += H_words();
    spin1_memcpy(p, g_state, g_words() * sizeof(uint32_t)); p += g_words();
    spin1_memcpy(p, pq_state, local_tokens * heads * features * sizeof(uint32_t));
    p += local_tokens * heads * features;
    spin1_memcpy(p, pk_state, local_tokens * heads * features * sizeof(uint32_t));
    p += local_tokens * heads * features;
    spin1_memcpy(p, v_state, local_tokens * d_model * sizeof(uint32_t));
}

static void finish_worker(uint32_t rc, uint32_t completed_steps) {
    if (final_rc != 0 && rc == 0) rc = final_rc;
    if (start_ms != 0) elapsed_ms = (uint32_t) sv->clock_ms - start_ms;
    write_output(rc, completed_steps);
    sark.vcpu->user1 = rc;
    sark.vcpu->user2 = completed_steps;
    sark.vcpu->user3 = (sark_core_id() << 16) | (sark_app_id() & 0xffffu);
    if (rc == 0) {
        io_printf(IO_BUF,
            "SNN_TRANSFORM_H3B_WORKER PASS rank=%u/%u Nlocal=%u Nglobal=%u tx=%u rx=%u retries=%u ms=%u\n",
            rank_id, n_workers, local_tokens, global_tokens,
            tx_packets, rx_packets, tx_retries, elapsed_ms);
    } else {
        io_printf(IO_BUF,
            "SNN_TRANSFORM_H3B_WORKER FAIL rank=%u rc=%u step=%u stage=%u tx=%u rx=%u proto=%u\n",
            rank_id, rc, current_step, current_stage,
            tx_packets, rx_packets, protocol_errors);
    }
    spin1_exit(rc);
}

static void fail_task(uint arg0, uint arg1) {
    (void) arg1;
    finish_worker((uint32_t) arg0, current_step);
}

static void protocol_fail(uint32_t rc) {
    ++protocol_errors;
    if (final_rc == 0) {
        final_rc = rc;
        (void) spin1_schedule_callback(fail_task, rc, 0, 0);
    }
}

static int send_payload(uint32_t key, uint32_t payload) {
    uint32_t spins = 0;
    while (!spin1_send_mc_packet(key, payload, WITH_PAYLOAD)) {
        ++tx_retries;
        if (++spins > 1000000u) return 0;
    }
    ++tx_packets;
    if (pacing_us != 0 && ((tx_packets & 7u) == 0u)) spin1_delay_us(pacing_us);
    return 1;
}

static int sender_from_key(uint32_t key, uint32_t *offset) {
    for (uint32_t r = 0; r < n_workers; ++r) {
        uint32_t base = peer_keys[r];
        if (key >= base && key < base + H3B_N_KEYS) {
            *offset = key - base;
            return (int) r;
        }
    }
    return -1;
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

static int allocate_arena(void) {
    const uint32_t Hw = H_words();
    const uint32_t gw = g_words();
    const uint32_t pqw = local_tokens * heads * features;
    const uint32_t vwords = local_tokens * d_model;
    const uint32_t token_words = d_model;
    const uint32_t feature_words = heads * features;
    const uint32_t base_words =
        Hw + gw + pqw + pqw + vwords
        + local_tokens * d_model
        + local_tokens * d_model
        + local_tokens * d_model
        + local_tokens * ff_dim
        + local_tokens * d_model
        + token_words + feature_words + feature_words + feature_words + token_words;

    arena_bytes_requested = (base_words + 1u) * sizeof(int32_t);
    uint8_t *raw = (uint8_t *) spin1_malloc(arena_bytes_requested);
    if (raw == NULL) return H3B_RC_ALLOC;
    arena_allocated = 1;
    uintptr_t aligned = ((uintptr_t) raw + 7u) & ~(uintptr_t) 7u;
    int32_t *p = (int32_t *) aligned;

    H_state = p; p += Hw;
    g_state = p; p += gw;
    pq_state = p; p += pqw;
    pk_state = p; p += pqw;
    v_state = p; p += vwords;
    ln_buf = p; p += local_tokens * d_model;
    attended = p; p += local_tokens * d_model;
    z_buf = p; p += local_tokens * d_model;
    hidden = p; p += local_tokens * ff_dim;
    y_buf = p; p += local_tokens * d_model;
    proj = p; p += token_words;
    a_buf = p; p += feature_words;
    new_pq = p; p += feature_words;
    new_pk = p; p += feature_words;
    new_v = p;

    /* N_local=32 provides exactly 2048 int32 words across attended+z, enough
     * for the 1024 int64 H reduction values.  hidden has ample room for g64. */
    if ((2u * local_tokens * d_model) < (2u * Hw) ||
            (local_tokens * ff_dim) < (2u * gw)) return 21;
    reduce_H64 = (int64_t *) attended;
    reduce_g64 = (int64_t *) hidden;
    return 0;
}

static void zero_reduction(void) {
    for (uint32_t i = 0; i < H_words(); ++i) reduce_H64[i] = 0;
    for (uint32_t i = 0; i < g_words(); ++i) reduce_g64[i] = 0;
}

static const int32_t *step_x(uint32_t step) {
    return (const int32_t *) trajectory + step * local_tokens * d_model;
}
static const uint32_t *step_mask(uint32_t step) {
    const uint32_t x_words = steps * local_tokens * d_model;
    return trajectory + x_words + step * local_tokens;
}

static void prepare_step0(void) {
    const int32_t *x = step_x(0);
    h2_layernorm_q20(
        x, local_tokens, d_model,
        hir_i32(H2F_LN1_G), hir_i32(H2F_LN1_B), ln_buf);
    for (uint32_t n = 0; n < local_tokens; ++n) {
        project_token_features(
            ln_buf + n * d_model,
            pq_state + n * heads * features,
            pk_state + n * heads * features,
            v_state + n * d_model);
    }

    zero_reduction();
    for (uint32_t n = 0; n < local_tokens; ++n) {
        const int32_t *pk = pk_state + n * heads * features;
        const int32_t *vv = v_state + n * d_model;
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t m = 0; m < features; ++m) {
                uint32_t hi = (h * features + m) * head_dim;
                int32_t kval = pk[h * features + m];
                for (uint32_t d = 0; d < head_dim; ++d) {
                    reduce_H64[hi + d] += (int64_t) kval * vv[h * head_dim + d];
                }
                reduce_g64[h * features + m] += kval;
            }
        }
    }
}

static void prepare_delta_step(uint32_t step) {
    const int32_t *x = step_x(step);
    const uint32_t *mask = step_mask(step);
    h2_layernorm_q20(
        x, local_tokens, d_model,
        hir_i32(H2F_LN1_G), hir_i32(H2F_LN1_B), ln_buf);
    zero_reduction();
    const uint32_t shift = H2F_F_PK + H2F_F_ACT - H2F_F_H;

    for (uint32_t n = 0; n < local_tokens; ++n) {
        if (mask[n] == 0) continue;
        int32_t *pq_old = pq_state + n * heads * features;
        int32_t *pk_old = pk_state + n * heads * features;
        int32_t *v_old = v_state + n * d_model;
        project_token_features(
            ln_buf + n * d_model, new_pq, new_pk, new_v);
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t m = 0; m < features; ++m) {
                uint32_t hi = (h * features + m) * head_dim;
                int32_t ko = pk_old[h * features + m];
                int32_t kn = new_pk[h * features + m];
                for (uint32_t d = 0; d < head_dim; ++d) {
                    int32_t vo = v_old[h * head_dim + d];
                    int32_t vn = new_v[h * head_dim + d];
                    int64_t op = h2_rshift_round_i64((int64_t) ko * vo, shift);
                    int64_t np = h2_rshift_round_i64((int64_t) kn * vn, shift);
                    reduce_H64[hi + d] += np - op;
                }
                reduce_g64[h * features + m] += (int64_t) kn - ko;
            }
        }
        spin1_memcpy(pq_old, new_pq, heads * features * sizeof(int32_t));
        spin1_memcpy(pk_old, new_pk, heads * features * sizeof(int32_t));
        spin1_memcpy(v_old, new_v, d_model * sizeof(int32_t));
    }
}

static void apply_rank0_reduction(void) {
    if (current_step == 0) {
        const uint32_t shift = H2F_F_PK + H2F_F_ACT - H2F_F_H;
        for (uint32_t i = 0; i < H_words(); ++i)
            H_state[i] = (int32_t) h2_rshift_round_i64(reduce_H64[i], shift);
        for (uint32_t i = 0; i < g_words(); ++i)
            g_state[i] = (int32_t) reduce_g64[i];
    } else {
        for (uint32_t i = 0; i < H_words(); ++i)
            H_state[i] = (int32_t) ((int64_t) H_state[i] + reduce_H64[i]);
        for (uint32_t i = 0; i < g_words(); ++i)
            g_state[i] = (int32_t) ((int64_t) g_state[i] + reduce_g64[i]);
    }
}

static void execute_tail(uint32_t step) {
    const int32_t *x = step_x(step);
    h2_attention_read(
        pq_state, H_state, g_state,
        local_tokens, heads, features, head_dim, attended);
    sparse_layer(attended, local_tokens, d_model,
        H2F_WO_ROW_PTR, H2F_WO_COL_IDX, H2F_WO_VALUES, H2F_WO_BIAS,
        d_model, z_buf);
    for (uint32_t i = 0; i < local_tokens * d_model; ++i)
        z_buf[i] = (int32_t) ((int64_t) z_buf[i] + x[i]);

    h2_layernorm_q20(
        z_buf, local_tokens, d_model,
        hir_i32(H2F_LN2_G), hir_i32(H2F_LN2_B), ln_buf);
    sparse_layer(ln_buf, local_tokens, d_model,
        H2F_W1_ROW_PTR, H2F_W1_COL_IDX, H2F_W1_VALUES, H2F_W1_BIAS,
        ff_dim, hidden);
    for (uint32_t i = 0; i < local_tokens * ff_dim; ++i)
        if (hidden[i] < 0) hidden[i] = 0;
    sparse_layer(hidden, local_tokens, ff_dim,
        H2F_W2_ROW_PTR, H2F_W2_COL_IDX, H2F_W2_VALUES, H2F_W2_BIAS,
        d_model, y_buf);
    for (uint32_t i = 0; i < local_tokens * d_model; ++i)
        y_buf[i] = (int32_t) ((int64_t) y_buf[i] + z_buf[i]);
}

static void send_grant(uint32_t target) {
    if (!send_payload(send_key + H3B_KEY_GRANT_BASE + current_step, target))
        protocol_fail(H3B_RC_PROTOCOL);
}

static void send_gather_task(uint arg0, uint arg1) {
    (void) arg0; (void) arg1;
    current_stage = H3B_STAGE_GATHER;
    uint32_t word = 0;
    for (uint32_t i = 0; i < H_words(); ++i) {
        uint64_t u = (uint64_t) reduce_H64[i];
        if (!send_payload(send_key + word++, (uint32_t) u)) return;
        if (!send_payload(send_key + word++, (uint32_t) (u >> 32))) return;
    }
    for (uint32_t i = 0; i < g_words(); ++i) {
        uint64_t u = (uint64_t) reduce_g64[i];
        if (!send_payload(send_key + word++, (uint32_t) u)) return;
        if (!send_payload(send_key + word++, (uint32_t) (u >> 32))) return;
    }
    if (word != gather_payload_words()) {
        protocol_fail(H3B_RC_PACKET_COUNT);
        return;
    }
    if (!send_payload(send_key + H3B_KEY_DONE_BASE + current_step, rank_id)) return;
    current_stage = H3B_STAGE_WAIT_BCAST;
}

static void tail_task(uint arg0, uint arg1);

static void broadcast_hg_task(uint arg0, uint arg1) {
    (void) arg0; (void) arg1;
    current_stage = H3B_STAGE_BCAST;
    uint32_t word = 0;
    for (uint32_t i = 0; i < H_words(); ++i)
        if (!send_payload(send_key + word++, (uint32_t) H_state[i])) return;
    for (uint32_t i = 0; i < g_words(); ++i)
        if (!send_payload(send_key + word++, (uint32_t) g_state[i])) return;
    if (word != hg_payload_words()) {
        protocol_fail(H3B_RC_PACKET_COUNT);
        return;
    }
    if (!send_payload(send_key + H3B_KEY_BCAST_DONE_BASE + current_step, rank_id)) return;
    ++bcast_phases;
    (void) spin1_schedule_callback(tail_task, 0, 0, 1);
}

static void rank0_gather_complete(void) {
    ++gather_phases;
    apply_rank0_reduction();
    (void) spin1_schedule_callback(broadcast_hg_task, 0, 0, 1);
}

static void start_rank0_gather(void) {
    active_sender = 1;
    recv_words = 0;
    current_stage = H3B_STAGE_GATHER;
    if (n_workers <= 1) {
        rank0_gather_complete();
        return;
    }
    send_grant(active_sender);
}

static void maybe_start_rank0_gather(void) {
    if (rank_id == 0 && ready_count == n_workers && current_stage == H3B_STAGE_PREP)
        start_rank0_gather();
}

static void send_ready(void) {
    current_stage = H3B_STAGE_PREP;
    if (rank_id == 0) {
        ready_mask = 1u;
        ready_count = 1u;
        maybe_start_rank0_gather();
    } else {
        if (!send_payload(send_key + H3B_KEY_READY_BASE + current_step, rank_id))
            protocol_fail(H3B_RC_PROTOCOL);
    }
}

static void prepare_task(uint arg0, uint arg1) {
    (void) arg0; (void) arg1;
    if (current_step == 0) prepare_step0();
    else prepare_delta_step(current_step);
    send_ready();
}

static void tail_task(uint arg0, uint arg1) {
    (void) arg0; (void) arg1;
    current_stage = H3B_STAGE_TAIL;
    execute_tail(current_step);
    if (current_step + 1u >= steps) {
        finish_worker(0, steps);
        return;
    }
    ++current_step;
    ready_mask = 0;
    ready_count = 0;
    recv_words = 0;
    bcast_words_received = 0;
    (void) spin1_schedule_callback(prepare_task, 0, 0, 1);
}

static void receive_rank0_worker_data(uint32_t sender, uint32_t off, uint32_t payload) {
    if (sender != active_sender || current_stage != H3B_STAGE_GATHER) {
        protocol_fail(H3B_RC_BAD_SENDER);
        return;
    }
    if (off >= gather_payload_words()) {
        protocol_fail(H3B_RC_PACKET_COUNT);
        return;
    }
    if ((recv_words & 1u) == 0u) {
        recv_low_word = payload;
    } else {
        uint64_t u = ((uint64_t) payload << 32) | recv_low_word;
        int64_t value = (int64_t) u;
        uint32_t idx64 = recv_words >> 1;
        if (idx64 < H_words()) reduce_H64[idx64] += value;
        else reduce_g64[idx64 - H_words()] += value;
    }
    ++recv_words;
}

static void receive_rank0_control(uint32_t sender, uint32_t off, uint32_t payload) {
    if (off == H3B_KEY_READY_BASE + current_step) {
        if (sender == 0 || payload != sender) return;
        uint32_t bit = 1u << sender;
        if ((ready_mask & bit) == 0) {
            ready_mask |= bit;
            ++ready_count;
            ++ready_events;
        }
        maybe_start_rank0_gather();
        return;
    }
    if (off == H3B_KEY_DONE_BASE + current_step) {
        if (sender != active_sender || payload != sender ||
                recv_words != gather_payload_words()) {
            protocol_fail(H3B_RC_PACKET_COUNT);
            return;
        }
        ++active_sender;
        recv_words = 0;
        if (active_sender >= n_workers) rank0_gather_complete();
        else send_grant(active_sender);
    }
}

static void receive_worker_from_rank0(uint32_t off, uint32_t payload) {
    if (off == H3B_KEY_GRANT_BASE + current_step) {
        if (payload == rank_id && current_stage == H3B_STAGE_PREP)
            (void) spin1_schedule_callback(send_gather_task, 0, 0, 1);
        return;
    }
    if (off < hg_payload_words()) {
        if (current_stage != H3B_STAGE_WAIT_BCAST && current_stage != H3B_STAGE_BCAST) {
            protocol_fail(H3B_RC_PROTOCOL);
            return;
        }
        current_stage = H3B_STAGE_BCAST;
        if (off < H_words()) H_state[off] = (int32_t) payload;
        else g_state[off - H_words()] = (int32_t) payload;
        ++bcast_words_received;
        return;
    }
    if (off == H3B_KEY_BCAST_DONE_BASE + current_step) {
        if (bcast_words_received != hg_payload_words()) {
            protocol_fail(H3B_RC_PACKET_COUNT);
            return;
        }
        ++bcast_phases;
        bcast_words_received = 0;
        (void) spin1_schedule_callback(tail_task, 0, 0, 1);
    }
}

static void receive_mcpl(uint key, uint payload) {
    ++rx_packets;
    uint32_t off = 0;
    int sender_i = sender_from_key(key, &off);
    if (sender_i < 0) {
        protocol_fail(H3B_RC_BAD_SENDER);
        return;
    }
    uint32_t sender = (uint32_t) sender_i;
    if (sender == rank_id) return;

    if (rank_id == 0) {
        if (off < gather_payload_words()) receive_rank0_worker_data(sender, off, payload);
        else receive_rank0_control(sender, off, payload);
    } else {
        if (sender != 0) {
            protocol_fail(H3B_RC_BAD_SENDER);
            return;
        }
        receive_worker_from_rank0(off, payload);
    }
}

static int validate_contract(void) {
    if (cfg[0] != H3B_MAGIC || cfg[1] != H3B_VERSION) return 10;
    local_tokens = cfg[2]; d_model = cfg[3]; heads = cfg[4]; features = cfg[5];
    ff_dim = cfg[6]; steps = cfg[7];
    if (local_tokens < 1 || d_model != 32 || heads != 2 || features != 32 ||
            ff_dim != 64 || steps != 6) return 11;
    if (cfg[8] != H2F_F_ACT || cfg[9] != H2F_F_PQ ||
            cfg[10] != H2F_F_PK || cfg[11] != H2F_F_H ||
            cfg[12] != H2F_F_MEM || cfg[13] != H2F_F_DEC) return 12;
    if (cfg[15] != H2F_N_ARRAYS) return 13;
    rank_id = cfg[H3B_CFG_RANK];
    n_workers = cfg[H3B_CFG_N_WORKERS];
    global_tokens = cfg[H3B_CFG_GLOBAL_TOKENS];
    send_key = cfg[H3B_CFG_SEND_KEY];
    pacing_us = cfg[H3B_CFG_PACING_US];
    if (n_workers < 2 || n_workers > H3B_MAX_WORKERS || rank_id >= n_workers ||
            global_tokens != n_workers * local_tokens) return 15;
    for (uint32_t r = 0; r < n_workers; ++r)
        peer_keys[r] = cfg[H3B_CFG_PEER_KEYS + r];
    head_dim = d_model / heads;
    scale_q20 = cfg[14];
    for (uint32_t i = 0; i < H2F_N_ARRAYS; ++i)
        if (h2f_desc_nbytes(cfg, (h2f_array_id_t) i) == 0) return 14;
    return 0;
}

static int initialise(void) {
    h3b_dse_metadata_t *data = (h3b_dse_metadata_t *) (uintptr_t) sark.vcpu->user0;
    if (data == NULL || data->magic_number != DSE_MAGIC || data->version != DSE_VERSION) return 30;
    for (uint32_t r = 0; r < 4; ++r)
        if (data->regions[r].pointer == NULL || data->regions[r].n_words == 0) return 32;
    if (data->regions[H2F_REGION_CONFIG].n_words < H3B_HEADER_WORDS) return 33;
    cfg = (const uint32_t *) data->regions[H2F_REGION_CONFIG].pointer;
    hir_blob = (const uint8_t *) data->regions[H2F_REGION_HIR].pointer;
    trajectory = (const uint32_t *) data->regions[H2F_REGION_TRAJECTORY].pointer;
    output_region = (uint32_t *) data->regions[H2F_REGION_OUTPUT].pointer;
    int rc = validate_contract();
    if (rc != 0) return rc;
    if (data->regions[H2F_REGION_TRAJECTORY].n_words < expected_local_trajectory_words()) return 34;
    if (data->regions[H2F_REGION_OUTPUT].n_words < expected_output_words()) return 35;
    return 0;
}

static void start_worker_task(uint arg0, uint arg1) {
    (void) arg0; (void) arg1;
    int rc = allocate_arena();
    if (rc != 0) {
        finish_worker((uint32_t) rc, 0);
        return;
    }
    q_evals = k_evals = v_evals = 0;
    tx_packets = rx_packets = tx_retries = protocol_errors = 0;
    ready_events = gather_phases = bcast_phases = 0;
    current_step = 0;
    current_stage = H3B_STAGE_PREP;
    ready_mask = ready_count = recv_words = bcast_words_received = 0;
    final_rc = 0;
    start_ms = (uint32_t) sv->clock_ms;
    (void) spin1_schedule_callback(prepare_task, 0, 0, 1);
}

void c_main(void) {
    int rc = initialise();
    spin1_callback_on(MCPL_PACKET_RECEIVED, receive_mcpl, -1);
    if (rc != 0) {
        final_rc = (uint32_t) rc;
        (void) spin1_schedule_callback(fail_task, (uint32_t) rc, 0, 0);
    } else {
        (void) spin1_schedule_callback(start_worker_task, 0, 0, 1);
    }
    spin1_start(SYNC_WAIT);
}
