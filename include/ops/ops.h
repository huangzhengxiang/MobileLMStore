#ifndef OPS_HPP
#define OPS_HPP

#include <cstddef>
#include <vector>
#include <cstdint>


namespace LMStore {
struct rope_config_params;
struct sq_per_tensor_params {
    float scale;
    int zero_point;
};

#if defined(__ARM_FP16_FORMAT_IEEE)
using fp16_t = __fp16;
#define LMSTORE_HAS_FP16 1
#elif defined(__FLT16_MANT_DIG__)
using fp16_t = _Float16;
#define LMSTORE_HAS_FP16 1
#endif

// dist functions
float l1_dist_fp32(const float* a, const float* b, size_t n);
float l2_dist_fp32(const float* a, const float* b, size_t n);
float dot_product_fp32(const float* a, const float* b, size_t n);
float cosine_sim_fp32(const float* a, const float* b, size_t n);
float l1_dist_u8(const uint8_t* a, const uint8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b);
float l2_dist_u8(const uint8_t* a, const uint8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b);
float dot_product_u8(const uint8_t* a, const uint8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b);
float cosine_sim_u8(const uint8_t* a, const uint8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b);
float l1_dist_s8(const int8_t* a, const int8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b);
float l2_dist_s8(const int8_t* a, const int8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b);
float dot_product_s8(const int8_t* a, const int8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b);
float cosine_sim_s8(const int8_t* a, const int8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b);

// RoPE, support in-place.
void apply_rope_emb(
    float* k,     // [B, L, D]
    float* result,      // [B, L, D]
    const float* cos,   // [L, D/2]
    const float* sin,   // [L, D/2]
    int B,                    // batch = 8
    int L,                    // prompt_len = 125 (可变)
    int D                     // hidden = 128
);
#if defined(LMSTORE_HAS_FP16)
void apply_rope_emb(
    fp16_t* k,
    fp16_t* result,
    const fp16_t* cos,
    const fp16_t* sin,
    int B,
    int L,
    int D);
#endif
std::vector<float> apply_rope_emb_wrapper(const std::vector<float>& k,
                                          const std::vector<float>& cos,
                                          const std::vector<float>& sin,
                                          int B, int L, int D);

// Precompute RoPE freqs_cos/freqs_sin for C++ pipeline.
// Output layout follows apply_rope_emb(): [L, D/2].
void precompute_rope_freqs_cis(
    float* freqs_cos,                 // [L, D/2]
    float* freqs_sin,                 // [L, D/2]
    int L,
    int D,
    float theta,                      // usually 10000.0
    bool use_hf_rope,                 // static_llama: hf_precompute_freqs_cis vs precompute_freqs_cis
    float partial_rotary_factor,      // only used when use_hf_rope=true
    bool use_scaled_rope = false,     // only used when use_hf_rope=false
    int scale_factor = 8,             // only used when use_scaled_rope=true
    int high_freq_factor = 4          // same default as rope.py
);

// Overload for config-driven pipeline usage.
// Auto maps parameters from rope_config_params to the base interface.
void precompute_rope_freqs_cis(
    float* freqs_cos,                 // [L, head_dim/2]
    float* freqs_sin,                 // [L, head_dim/2]
    int L,
    const rope_config_params& rope_cfg
);

// hadamard transform (FWHT), support in-place if transpose=false.                     
void fwht_tensor(float* input, float* output, 
                 int B, int L, int D, 
                 bool transpose=false);
void fwht_tensor(float* input, float* output, 
                 int B, int L, int D, float scale, 
                 bool transpose=false);
#if defined(LMSTORE_HAS_FP16)
void fwht_tensor(fp16_t* input, fp16_t* output,
                 int B, int L, int D,
                 bool transpose=false);
void fwht_tensor(fp16_t* input, fp16_t* output,
                 int B, int L, int D, fp16_t scale,
                 bool transpose=false);
#endif

// don't support inplace.
void transpose_tensor(float* input, float* output, 
                      int dim0, int dim1, int dim2);
void transpose_tensor(uint8_t* input, uint8_t* output,
                      int dim0, int dim1, int dim2);
#if defined(LMSTORE_HAS_FP16)
void transpose_tensor(fp16_t* input, fp16_t* output,
                      int dim0, int dim1, int dim2);
#endif

void compute_rerotation_cos_sin(
    const float* freqs_cos, // [L, half]
    const float* freqs_sin, // [L, half]
    float* rerotation_cos,  // [matched_len, half]
    float* rerotation_sin,  // [matched_len, half]
    int64_t L, int64_t half,
    int64_t ori_pos, int64_t new_pos, int64_t matched_len);
#if defined(LMSTORE_HAS_FP16)
void compute_rerotation_cos_sin(
    const fp16_t* freqs_cos,
    const fp16_t* freqs_sin,
    fp16_t* rerotation_cos,
    fp16_t* rerotation_sin,
    int64_t L, int64_t half,
    int64_t ori_pos, int64_t new_pos, int64_t matched_len);
#endif

// Per-Layer Rerotation, support in-place.                 
void rerotate_k_fp32(float* k, float* output,
                     const float* cos, const float* sin,
                     int B, int L, int D, 
                     bool enable_r3);
void rerotate_k_u8(uint8_t* k, uint8_t* output, 
                   const float* cos, const float* sin,
                   int B, int L, int D, struct sq_per_tensor_params params,
                   bool enable_r3);
#if defined(LMSTORE_HAS_FP16)
void rerotate_k_u8_fp16(uint8_t* k, uint8_t* output,
                        const fp16_t* cos, const fp16_t* sin,
                        int B, int L, int D, struct sq_per_tensor_params params,
                        bool enable_r3);
#endif

// SQ
void scalar_quantize_per_tensor_u8(
    const float* input,
    uint8_t* output,
    size_t size,
    float scale,
    int zero_point);
void scalar_quantize_per_tensor_i8(
    const float* input,
    int8_t* output,
    size_t size,
    float scale,
    int zero_point);
void scalar_dequantize_per_tensor_u8(
    const uint8_t* input,
    float* output,
    size_t size,
    float scale,
    int zero_point);
#if defined(LMSTORE_HAS_FP16)
void scalar_dequantize_per_tensor_u8(
    const uint8_t* input,
    fp16_t* output,
    size_t size,
    float scale,
    int zero_point);
#endif
void scalar_dequantize_per_tensor_i8(
    const int8_t* input,
    float* output,
    size_t size,
    float scale,
    int zero_point);
void scalar_dequantize_per_tensor_u16(
    const uint16_t* input,
    float* output,
    size_t size,
    float scale,
    int zero_point);
void scalar_dequantize_per_tensor_i16(
    const int16_t* input,
    float* output,
    size_t size,
    float scale,
    int zero_point);
}

#endif // OPS_HPP
