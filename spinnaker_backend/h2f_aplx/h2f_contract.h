#ifndef SNN_TRANSFORM_H2F_CONTRACT_H
#define SNN_TRANSFORM_H2F_CONTRACT_H

#include <stdint.h>

#define H2F_MAGIC 0x48324631u
#define H2F_OUTPUT_MAGIC 0x4832464fu
#define H2F_VERSION 1u
#define H2F_HEADER_WORDS 96u
#define H2F_OUTPUT_HEADER_WORDS 8u
#define H2F_DESC_BASE 28u

#define H2F_F_ACT 20u
#define H2F_F_PQ 24u
#define H2F_F_PK 16u
#define H2F_F_H 20u
#define H2F_F_MEM 22u
#define H2F_F_DEC 20u

/* HIR blob descriptor order; must match h2f_contract.py. */
typedef enum h2f_array_id_e {
    H2F_LN1_G = 0,
    H2F_LN1_B,
    H2F_LN2_G,
    H2F_LN2_B,
    H2F_WQ,
    H2F_BQ,
    H2F_WK,
    H2F_BK,
    H2F_WV,
    H2F_BV,
    H2F_OMEGA,
    H2F_WO_ROW_PTR,
    H2F_WO_COL_IDX,
    H2F_WO_VALUES,
    H2F_WO_BIAS,
    H2F_W1_ROW_PTR,
    H2F_W1_COL_IDX,
    H2F_W1_VALUES,
    H2F_W1_BIAS,
    H2F_W2_ROW_PTR,
    H2F_W2_COL_IDX,
    H2F_W2_VALUES,
    H2F_W2_BIAS,
    H2F_Q_DECODER_0,
    H2F_Q_DECODER_1,
    H2F_K_DECODER_0,
    H2F_K_DECODER_1,
    H2F_N_ARRAYS
} h2f_array_id_t;

typedef enum h2f_regions_e {
    H2F_REGION_CONFIG = 0,
    H2F_REGION_HIR = 1,
    H2F_REGION_TRAJECTORY = 2,
    H2F_REGION_OUTPUT = 3
} h2f_regions_t;

/* output header words */
#define H2F_OUT_MAGIC 0u
#define H2F_OUT_RC 1u
#define H2F_OUT_STEPS 2u
#define H2F_OUT_Q_EVALS 3u
#define H2F_OUT_K_EVALS 4u
#define H2F_OUT_V_EVALS 5u
#define H2F_OUT_CHIP 6u
#define H2F_OUT_CORE_APP 7u

static inline uint32_t h2f_desc_offset(const uint32_t *header, h2f_array_id_t id) {
    return header[H2F_DESC_BASE + 2u * (uint32_t) id];
}

static inline uint32_t h2f_desc_nbytes(const uint32_t *header, h2f_array_id_t id) {
    return header[H2F_DESC_BASE + 2u * (uint32_t) id + 1u];
}

#endif
