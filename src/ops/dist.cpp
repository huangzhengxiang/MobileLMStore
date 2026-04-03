#include <cstddef>
#include <cstring>
#include <cmath>
#include "ops/ops.h"
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__NEON__)
#include <arm_neon.h>
#endif

namespace LMStore {

//////////////////////////////////////////////////////////////
// Vector-Vector Distance Functions
//////////////////////////////////////////////////////////////
float l1_dist_fp32(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
    float tmp[8];

#if defined(__AVX2__)
    __m256 vsum = _mm256_setzero_ps();

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff); // abs
        vsum = _mm256_add_ps(vsum, diff);
    }

    _mm256_storeu_ps(tmp, vsum);
    for (int k = 0; k < 8; ++k) sum += tmp[k];
#elif defined(__NEON__)
    float32x4_t vsum = vdupq_n_f32(0.0f);

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vabsq_f32(vsubq_f32(va, vb));
        vsum = vaddq_f32(vsum, diff);
    }

    vst1q_f32(tmp, vsum);
    for (int k = 0; k < 4; ++k) sum += tmp[k];
#endif
    for (; i < n; ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

float l2_dist_fp32(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
    float tmp[8];

#if defined(__AVX2__)
    __m256 vsum = _mm256_setzero_ps();

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(diff, diff));
    }

    _mm256_storeu_ps(tmp, vsum);
    for (int k = 0; k < 8; ++k) sum += tmp[k];
#elif defined(__NEON__)
    float32x4_t vsum = vdupq_n_f32(0.0f);

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        vsum = vmlaq_f32(vsum, diff, diff);
    }

    vst1q_f32(tmp, vsum);
    for (int k = 0; k < 4; ++k) sum += tmp[k];
#endif
    for (; i < n; ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return std::sqrt(sum);
}

float dot_product_fp32(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
    float tmp[8];

#if defined(__AVX2__)
    __m256 vsum = _mm256_setzero_ps();

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
    }

    _mm256_storeu_ps(tmp, vsum);
    for (int k = 0; k < 8; ++k) sum += tmp[k];
#elif defined(__NEON__)
    float32x4_t vsum = vdupq_n_f32(0.0f);

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, va, vb);
    }

    vst1q_f32(tmp, vsum);
    for (int k = 0; k < 4; ++k) sum += tmp[k];
#endif
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float cosine_sim_fp32(const float* a, const float* b, size_t n) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    size_t i = 0;
    float tmp1[8], tmp2[8], tmp3[8];

#if defined(__AVX2__)
    __m256 vdot = _mm256_setzero_ps();
    __m256 va2 = _mm256_setzero_ps();
    __m256 vb2 = _mm256_setzero_ps();

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

        vdot = _mm256_add_ps(vdot, _mm256_mul_ps(va, vb));
        va2  = _mm256_add_ps(va2, _mm256_mul_ps(va, va));
        vb2  = _mm256_add_ps(vb2, _mm256_mul_ps(vb, vb));
    }

    _mm256_storeu_ps(tmp1, vdot);
    _mm256_storeu_ps(tmp2, va2);
    _mm256_storeu_ps(tmp3, vb2);

    for (int k = 0; k < 8; ++k) {
        dot += tmp1[k];
        norm_a += tmp2[k];
        norm_b += tmp3[k];
    }
#elif defined(__NEON__)
    float32x4_t vdot = vdupq_n_f32(0);
    float32x4_t va2  = vdupq_n_f32(0);
    float32x4_t vb2  = vdupq_n_f32(0);

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);

        vdot = vmlaq_f32(vdot, va, vb);
        va2  = vmlaq_f32(va2, va, va);
        vb2  = vmlaq_f32(vb2, vb, vb);
    }

    vst1q_f32(tmp1, vdot);
    vst1q_f32(tmp2, va2);
    vst1q_f32(tmp3, vb2);

    for (int k = 0; k < 4; ++k) {
        dot += tmp1[k];
        norm_a += tmp2[k];
        norm_b += tmp3[k];
    }
#endif
    for (; i < n; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-8f);
}

//////////////////////////////////////////////////////////////
// Vector-Matrix Distance Functions
//////////////////////////////////////////////////////////////

} // namespace LMStore