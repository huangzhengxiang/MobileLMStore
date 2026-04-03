#include <cstddef>
#include <cstring>
#include <cmath>
#include "ops/ops.h"
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__NEON__)
#include <arm_neon.h>
#endif
#include <iostream>


namespace LMStore {

void rerotate_k_fp32(float* k, float* output, 
                     const float* cos, const float* sin,
                     int B, int L, int D, 
                     bool enable_r3) {
    // k: [1, B, D, L]
    // cos, sin: [L, D/2]
    // output: [1, B, D, L]
    float* buffer = new float[B * L * D];
    if (enable_r3) {
        fwht_tensor(k, buffer, B, L, D, 1/std::sqrt(D), true);
    } else {
        transpose_tensor(k, buffer, B, D, L);
    }
    apply_rope_emb(buffer, buffer, cos, sin, B, L, D);
    if (enable_r3) {
        fwht_tensor(buffer, buffer, B, L, D, 1/std::sqrt(D), false);
    }
    transpose_tensor(buffer, output, B, L, D);
    delete[] buffer;
}

void rerotate_k_u8(uint8_t* k, uint8_t* output, 
                   const float* cos, const float* sin,
                   int B, int L, int D, struct sq_per_tensor_params params,
                   bool enable_r3) {
    // k: [1, B, D, L]
    // cos, sin: [L, D/2]
    // output: [1, B, D, L]
    size_t size = B * L * D;
    float* k_buffer = new float[size];
    scalar_dequantize_per_tensor_u8(k, k_buffer, size, params.scale, params.zero_point);
    rerotate_k_fp32(k_buffer, k_buffer, cos, sin, B, L, D, enable_r3);
    scalar_quantize_per_tensor_u8(k_buffer, output, size, params.scale, params.zero_point);
    delete[] k_buffer;
}

}