/*
 * H2 APLX arithmetic bring-up for SNNTransformBlock-v1.1.
 *
 * This is intentionally a one-core arithmetic self-test, not yet a claim that
 * the full frozen-HIR block has run on SpiNNaker.  It validates that the same C
 * kernels used by the H2D host parity gate execute on the ARM968 target.
 *
 * The application uses the standard SpiNNaker SYNC_WAIT start barrier.  This
 * avoids a race where a tiny raw c_main() could finish before the host front-end
 * had observed its RUNNING state.  After SYNC0 the scheduled callback performs
 * the tests, writes the result into VCPU user registers + IOBUF, and exits.
 */

#include <sark.h>
#include <stdint.h>
#include "spin1_api.h"

#include "h2_kernels.h"

#define H2E_BOARD_MAGIC 0x48324531u  /* ASCII-ish "H2E1" */

static int self_test(void) {
    /* Exact scaled-division checks exercise the 64-bit quotient path. */
    if (h2_div_scaled_round_i64(3, 2, 16) != 98304) return 1;
    if (h2_div_scaled_round_i64(-3, 2, 16) != -98304) return 2;
    if (h2_isqrt_u64(144) != 12) return 3;

    /* One-head, one-feature, one-dimension resident-state delta update. */
    int32_t H[1] = {1000};
    int32_t g[1] = {500};
    const int32_t old_pk[1] = {100};
    const int32_t old_v[1] = {1 << H2_F_ACT};
    const int32_t new_pk[1] = {120};
    const int32_t new_v[1] = {1 << H2_F_ACT};
    h2_attention_token_delta(H, g, old_pk, old_v, new_pk, new_v, 1, 1, 1);
    if (g[0] != 520) return 4;

    /*
     * pk is Q16, v is Q20 and H is Q20.  Therefore the outer-product
     * conversion shifts by 16 bits.  With v=1.0, an integer pk code of 100
     * contributes 100 * 2^(20-16) = 1600 H-code units; 120 contributes 1920.
     * Hence H_new = 1000 - 1600 + 1920 = 1320.
     *
     * The previous self-test incorrectly expected 1020.  The kernel itself was
     * correct; the physical-board rc=5 result exposed this test-vector mistake.
     */
    if (H[0] != 1320) return 5;

    /* Normalized read: pq*H/(pq*g), converted to Q20 by the frozen contract. */
    const int32_t pq[1] = {1 << H2_F_PQ};
    int32_t attended[1] = {0};
    h2_attention_read(pq, H, g, 1, 1, 1, 1, attended);
    const int32_t expected = h2_div_scaled_round_i64(
        H[0], g[0], H2_F_ACT + H2_F_PK - H2_F_H);
    if (attended[0] != expected) return 6;

    /* Sparse Q20 linear identity on one scalar. */
    const uint32_t row_ptr[2] = {0, 1};
    const uint16_t col_idx[1] = {0};
    const int32_t values[1] = {1 << H2_F_ACT};
    const int32_t bias[1] = {0};
    const int32_t x[1] = {123456};
    int32_t y[1] = {0};
    h2_sparse_linear_q20(x, 1, 1, row_ptr, col_idx, values, bias, 1, y);
    if (y[0] != x[0]) return 7;

    return 0;
}

static void run_self_test(uint unused0, uint unused1) {
    (void) unused0;
    (void) unused1;

    const uint chip = sark_chip_id();
    const uint core = sark_core_id();
    const uint app = sark_app_id();
    const int rc = self_test();

    /* Leave a machine-visible signature as well as the IOBUF text. */
    sark.vcpu->user0 = H2E_BOARD_MAGIC;
    sark.vcpu->user1 = (uint) rc;
    sark.vcpu->user2 = chip;
    sark.vcpu->user3 = (core << 16) | (app & 0xffffu);

    if (rc == 0) {
        io_printf(IO_BUF,
            "SNN_TRANSFORM_H2_APLX_SELFTEST PASS chip=0x%04x core=%u app=%u\n",
            chip, core, app);
    } else {
        io_printf(IO_BUF,
            "SNN_TRANSFORM_H2_APLX_SELFTEST FAIL rc=%d chip=0x%04x core=%u app=%u\n",
            rc, chip, core, app);
    }

    spin1_exit((uint) rc);
}

void c_main(void) {
    /* Queue the work now; it will execute only after the host sends SYNC0. */
    if (!spin1_schedule_callback(run_self_test, 0, 0, 1)) {
        sark.vcpu->user0 = H2E_BOARD_MAGIC;
        sark.vcpu->user1 = 8;
        io_printf(IO_BUF,
            "SNN_TRANSFORM_H2_APLX_SELFTEST FAIL rc=8 callback_schedule\n");
        return;
    }

    spin1_start(SYNC_WAIT);
}
