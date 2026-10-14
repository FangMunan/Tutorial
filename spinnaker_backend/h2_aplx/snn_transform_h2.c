/*
 * H2 APLX arithmetic bring-up for SNNTransformBlock-v1.1.
 *
 * This is intentionally a resident arithmetic self-test, not yet a claim that
 * the full block has run on SpiNNaker.  It validates that the same C kernels
 * used by the host parity gate can be linked into an APLX image and executed on
 * an ARM968 core before wiring SDRAM HIR loading and multicast token events.
 */

#include <sark.h>
#include <stdint.h>

#include "h2_kernels.h"

static int self_test(void) {
    /* Exact scaled-division check: round((3/2)*2^16) = 98304. */
    if (h2_div_scaled_round_i64(3, 2, 16) != 98304) return 1;
    if (h2_div_scaled_round_i64(-3, 2, 16) != -98304) return 2;

    /* One-head, one-feature, one-dimension subtract-old/add-new state update. */
    int32_t H[1] = {1000};
    int32_t g[1] = {500};
    const int32_t old_pk[1] = {100};
    const int32_t old_v[1] = {1 << H2_F_ACT};
    const int32_t new_pk[1] = {120};
    const int32_t new_v[1] = {1 << H2_F_ACT};
    h2_attention_token_delta(H, g, old_pk, old_v, new_pk, new_v, 1, 1, 1);
    if (g[0] != 520) return 3;
    /* outer shift is 16, so a Q16 pk times Q20 v contributes pk in Q20. */
    if (H[0] != 1020) return 4;

    return 0;
}

void c_main(void) {
    const int rc = self_test();
    if (rc == 0) {
        io_printf(IO_BUF, "SNN_TRANSFORM_H2_APLX_SELFTEST PASS\n");
    } else {
        io_printf(IO_BUF, "SNN_TRANSFORM_H2_APLX_SELFTEST FAIL rc=%d\n", rc);
    }
}
