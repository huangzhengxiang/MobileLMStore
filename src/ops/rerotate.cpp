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

void compute_rerotation_cos_sin(
    const float* freqs_cos, // [L, half]
    const float* freqs_sin, // [L, half]
    float* rerotation_cos,  // [matched_len, half]
    float* rerotation_sin,  // [matched_len, half]
    int64_t L,
    int64_t half,
    int64_t ori_pos,
    int64_t new_pos,
    int64_t matched_len
) {
    for (int i = 0; i < matched_len; ++i) {
        const float* orig_cos = freqs_cos + (ori_pos + i) * half;
        const float* orig_sin = freqs_sin + (ori_pos + i) * half;
        const float* new_cos  = freqs_cos + (new_pos + i) * half;
        const float* new_sin  = freqs_sin + (new_pos + i) * half;

        float* out_cos = rerotation_cos + i * half;
        float* out_sin = rerotation_sin + i * half;

        int j = 0;
#if defined(__AVX2__)
        for (; j + 8 <= half; j += 8) {
            __m256 orig_c = _mm256_loadu_ps(orig_cos + j);
            __m256 orig_s = _mm256_loadu_ps(orig_sin + j);
            __m256 new_c  = _mm256_loadu_ps(new_cos + j);
            __m256 new_s  = _mm256_loadu_ps(new_sin + j);

            // cos = new_c * orig_c + new_s * orig_s
            __m256 cos_part1 = _mm256_mul_ps(new_c, orig_c);
            __m256 cos_part2 = _mm256_mul_ps(new_s, orig_s);
            __m256 cos_res   = _mm256_add_ps(cos_part1, cos_part2);

            // sin = new_s * orig_c - new_c * orig_s
            __m256 sin_part1 = _mm256_mul_ps(new_s, orig_c);
            __m256 sin_part2 = _mm256_mul_ps(new_c, orig_s);
            __m256 sin_res   = _mm256_sub_ps(sin_part1, sin_part2);

            _mm256_storeu_ps(out_cos + j, cos_res);
            _mm256_storeu_ps(out_sin + j, sin_res);
        }
#elif defined(__NEON__)
        for (; j + 4 <= half; j += 4) {
            float32x4_t orig_c = vld1q_f32(orig_cos + j);
            float32x4_t orig_s = vld1q_f32(orig_sin + j);
            float32x4_t new_c  = vld1q_f32(new_cos + j);
            float32x4_t new_s  = vld1q_f32(new_sin + j);

            float32x4_t cos_res = vmlaq_f32(
                vmulq_f32(new_c, orig_c),
                new_s, orig_s
            );

            float32x4_t sin_res = vsubq_f32(
                vmulq_f32(new_s, orig_c),
                vmulq_f32(new_c, orig_s)
            );

            vst1q_f32(out_cos + j, cos_res);
            vst1q_f32(out_sin + j, sin_res);
        }
#endif
        // tail/fallback
        for (int64_t j = 0; j < half; ++j) {
            out_cos[j] = new_cos[j] * orig_cos[j] + new_sin[j] * orig_sin[j];
            out_sin[j] = new_sin[j] * orig_cos[j] - new_cos[j] * orig_sin[j];
        }
    }
}

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