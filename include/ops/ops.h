#ifndef OPS_HPP
#define OPS_HPP

#include <cstddef>
#include <vector>
#include <cstdint>


namespace LMStore {
struct sq_per_tensor_params {
    float scale;
    int zero_point;
};

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
std::vector<float> apply_rope_emb_wrapper(const std::vector<float>& k,
                                          const std::vector<float>& cos,
                                          const std::vector<float>& sin,
                                          int B, int L, int D);

// hadamard transform (FWHT), support in-place if transpose=false.                     
void fwht_tensor(float* input, float* output, 
                 int B, int L, int D, 
                 bool transpose=false);
void fwht_tensor(float* input, float* output, 
                 int B, int L, int D, float scale, 
                 bool transpose=false);

// don't support inplace.
void transpose_tensor(float* input, float* output, 
                      int dim0, int dim1, int dim2);

// Per-Layer Rerotation, support in-place.                 
void rerotate_k_fp32(float* k, float* output,
                     const float* cos, const float* sin,
                     int B, int L, int D, 
                     bool enable_r3);
void rerotate_k_u8(uint8_t* k, uint8_t* output, 
                   const float* cos, const float* sin,
                   int B, int L, int D, struct sq_per_tensor_params params,
                   bool enable_r3);

// SQ
void scalar_quantize_per_tensor_u8(
    const float* input,
    uint8_t* output,
    size_t size,
    float scale,
    int zero_point);
void scalar_dequantize_per_tensor_u8(
    const uint8_t* input,
    float* output,
    size_t size,
    float scale,
    int zero_point);
}

#endif // OPS_HPP