#ifndef H2_RESIDENT_FORMAT_H
#define H2_RESIDENT_FORMAT_H

#include <stdint.h>

/* Data-region IDs must match resident_vertex.py. */
typedef enum {
    H2_REGION_SYSTEM = 0,
    H2_REGION_PARAMS = 1,
    H2_REGION_MODEL = 2,
    H2_REGION_CALIBRATION = 3,
    H2_REGION_LABELS = 4,
    H2_REGION_CANVASES = 5,
    H2_REGION_RECORDING = 6
} h2_region_t;

#define H2_PARAM_MAGIC 0x48325231u  /* 'H2R1' */
#define H2_PARAM_VERSION 1u
#define H2_PARAM_WORDS 16u
#define H2_MAX_DECODER 32u
#define H2_CALIB_RECORDS 8u

/* 16 words written by resident_vertex.py. */
typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t model_bytes;
    uint32_t calibration_bytes;
    uint32_t labels_bytes;
    uint32_t canvases_bytes;
    uint32_t output_bytes;
    uint32_t profile_bytes;
    uint32_t batch;
    uint32_t rounds;
    uint32_t seq;
    uint32_t d_model;
    uint32_t layers;
    uint32_t heads;
    uint32_t features;
    uint32_t flags;
} h2_params_t;

/* Fixed-size record written by export_h2_resident_bundle.py. */
typedef struct __attribute__((packed)) {
    uint32_t layer;
    uint32_t side;       /* 0=Q, 1=K */
    float bias_ref;      /* FP32 reference; fixed-point exporter will replace */
    float gain_ref;
    uint32_t window_ms;
    uint32_t decoder_len;
    float decoder_ref[H2_MAX_DECODER];
} h2_calib_record_f32_t;

/* Compact profiling record returned in profile mode.  Extend only by versioning. */
typedef struct __attribute__((packed)) {
    uint32_t status;
    uint32_t version;
    uint64_t cycles_total;
    uint64_t cycles_embed;
    uint64_t cycles_qkv;
    uint64_t cycles_dynamic_phi;
    uint64_t cycles_hg;
    uint64_t cycles_ffn;
    uint64_t cycles_norm_output;
    uint64_t dma_bytes_read;
    uint64_t dma_bytes_written;
    uint64_t multicast_packets;
} h2_profile_v1_t;

#endif
