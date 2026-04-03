#include <cstddef>
#include "ops/ops.h"
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__NEON__)
#include <arm_neon.h>
#endif

namespace LMStore {


void apply_rope_emb(
    float* k,     // [B, L, D]
    float* result,      // [B, L, D]
    const float* cos,   // [L, D/2]
    const float* sin,   // [L, D/2]
    int B,
    int L,
    int D
) {
    // in-place operation is allowed, i.e., k and result can point to the same memory
    const int half = D / 2;

    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            const float* k_ptr = k + b * L * D + l * D;
            float* out_ptr     = result + b * L * D + l * D;

            const float* cos_ptr = cos + l * half;
            const float* sin_ptr = sin + l * half;

            int d = 0;
#ifdef __AVX2__
            for (; d + 8 <= half; d += 8) {
                __m256 xr = _mm256_loadu_ps(k_ptr + d);
                __m256 xi = _mm256_loadu_ps(k_ptr + d + half);
                __m256 c  = _mm256_loadu_ps(cos_ptr + d);
                __m256 s  = _mm256_loadu_ps(sin_ptr + d);

                // real: xr * c - xi * s
                __m256 out_r = _mm256_sub_ps(_mm256_mul_ps(xr, c), _mm256_mul_ps(xi, s));
                // imag: xr * s + xi * c
                __m256 out_i = _mm256_add_ps(_mm256_mul_ps(xr, s), _mm256_mul_ps(xi, c));

                _mm256_storeu_ps(out_ptr + d, out_r);
                _mm256_storeu_ps(out_ptr + d + half, out_i);
            }
#elif defined(__NEON__)
            for (; d + 4 <= half; d += 4) {
                float32x4_t xr = vld1q_f32(k_ptr + d);
                float32x4_t xi = vld1q_f32(k_ptr + d + half);
                float32x4_t c  = vld1q_f32(cos_ptr + d);
                float32x4_t s  = vld1q_f32(sin_ptr + d);

                float32x4_t out_r = vsubq_f32(vmulq_f32(xr, c), vmulq_f32(xi, s));
                float32x4_t out_i = vaddq_f32(vmulq_f32(xr, s), vmulq_f32(xi, c));

                vst1q_f32(out_ptr + d, out_r);
                vst1q_f32(out_ptr + d + half, out_i);
            }
#endif
            for (; d < half; ++d) {
                float xr = k_ptr[d];
                float xi = k_ptr[d + half];

                float c = cos_ptr[d];
                float s = sin_ptr[d];

                out_ptr[d] = xr * c - xi * s;
                out_ptr[d + half] = xr * s + xi * c;
            }
        }
    }
}

std::vector<float> apply_rope_emb_wrapper(std::vector<float>& k,
                                          const std::vector<float>& cos,
                                          const std::vector<float>& sin,
                                          int B, int L, int D) {
    std::vector<float> result(k.size());
    apply_rope_emb(k.data(), result.data(), cos.data(), sin.data(),  B, L, D);
    return result;
}

} // namespace LMStore