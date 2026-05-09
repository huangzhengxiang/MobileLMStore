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

float l1_dist_u8(const uint8_t* a, const uint8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b) {
    float sum = 0.0f;
    size_t i = 0;
#if defined(__AVX2__)
    const __m128i vza16 = _mm_set1_epi16(static_cast<short>(zero_point_a));
    const __m128i vzb16 = _mm_set1_epi16(static_cast<short>(zero_point_b));
    const __m256 vsa = _mm256_set1_ps(scale_a);
    const __m256 vsb = _mm256_set1_ps(scale_b);
    const __m256 vabs_mask = _mm256_set1_ps(-0.0f);
    float tmp[8];
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m128i va8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
        __m128i va16 = _mm_sub_epi16(_mm_cvtepu8_epi16(va8), vza16);
        __m128i vb16 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb8), vzb16);
        __m256 fa = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va16)), vsa);
        __m256 fb = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb16)), vsb);
        __m256 diff = _mm256_sub_ps(fa, fb);
        diff = _mm256_andnot_ps(vabs_mask, diff);
        vsum = _mm256_add_ps(vsum, diff);
    }
    _mm256_storeu_ps(tmp, vsum);
    for (int k = 0; k < 8; ++k) sum += tmp[k];
#elif defined(__NEON__)
    const int16x8_t vza16 = vdupq_n_s16(static_cast<int16_t>(zero_point_a));
    const int16x8_t vzb16 = vdupq_n_s16(static_cast<int16_t>(zero_point_b));
    const float32x4_t vsa = vdupq_n_f32(scale_a);
    const float32x4_t vsb = vdupq_n_f32(scale_b);
    float tmp[4];
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        int16x8_t va16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(a + i))), vza16);
        int16x8_t vb16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(b + i))), vzb16);
        float32x4_t fa_lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(va16))), vsa);
        float32x4_t fa_hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(va16))), vsa);
        float32x4_t fb_lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vb16))), vsb);
        float32x4_t fb_hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vb16))), vsb);
        vsum = vaddq_f32(vsum, vabsq_f32(vsubq_f32(fa_lo, fb_lo)));
        vsum = vaddq_f32(vsum, vabsq_f32(vsubq_f32(fa_hi, fb_hi)));
    }
    vst1q_f32(tmp, vsum);
    for (int k = 0; k < 4; ++k) sum += tmp[k];
#endif
    for (; i < n; ++i) {
        const float da = (static_cast<int>(a[i]) - zero_point_a) * scale_a;
        const float db = (static_cast<int>(b[i]) - zero_point_b) * scale_b;
        sum += std::abs(da - db);
    }
    return sum;
}

float l2_dist_u8(const uint8_t* a, const uint8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b) {
    float sum = 0.0f;
    size_t i = 0;
#if defined(__AVX2__)
    const __m128i vza16 = _mm_set1_epi16(static_cast<short>(zero_point_a));
    const __m128i vzb16 = _mm_set1_epi16(static_cast<short>(zero_point_b));
    const __m256 vsa = _mm256_set1_ps(scale_a);
    const __m256 vsb = _mm256_set1_ps(scale_b);
    float tmp[8];
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m128i va8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
        __m128i va16 = _mm_sub_epi16(_mm_cvtepu8_epi16(va8), vza16);
        __m128i vb16 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb8), vzb16);
        __m256 fa = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va16)), vsa);
        __m256 fb = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb16)), vsb);
        __m256 diff = _mm256_sub_ps(fa, fb);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(diff, diff));
    }
    _mm256_storeu_ps(tmp, vsum);
    for (int k = 0; k < 8; ++k) sum += tmp[k];
#elif defined(__NEON__)
    const int16x8_t vza16 = vdupq_n_s16(static_cast<int16_t>(zero_point_a));
    const int16x8_t vzb16 = vdupq_n_s16(static_cast<int16_t>(zero_point_b));
    const float32x4_t vsa = vdupq_n_f32(scale_a);
    const float32x4_t vsb = vdupq_n_f32(scale_b);
    float tmp[4];
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        int16x8_t va16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(a + i))), vza16);
        int16x8_t vb16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(b + i))), vzb16);
        float32x4_t fa_lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(va16))), vsa);
        float32x4_t fa_hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(va16))), vsa);
        float32x4_t fb_lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vb16))), vsb);
        float32x4_t fb_hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vb16))), vsb);
        float32x4_t diff_lo = vsubq_f32(fa_lo, fb_lo);
        float32x4_t diff_hi = vsubq_f32(fa_hi, fb_hi);
        vsum = vmlaq_f32(vsum, diff_lo, diff_lo);
        vsum = vmlaq_f32(vsum, diff_hi, diff_hi);
    }
    vst1q_f32(tmp, vsum);
    for (int k = 0; k < 4; ++k) sum += tmp[k];
#endif
    for (; i < n; ++i) {
        const float da = (static_cast<int>(a[i]) - zero_point_a) * scale_a;
        const float db = (static_cast<int>(b[i]) - zero_point_b) * scale_b;
        const float diff = da - db;
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float dot_product_u8(const uint8_t* a, const uint8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b) {
    std::int64_t sum = 0;
    size_t i = 0;
#if defined(__AVX2__)
    const __m128i vza16 = _mm_set1_epi16(static_cast<short>(zero_point_a));
    const __m128i vzb16 = _mm_set1_epi16(static_cast<short>(zero_point_b));
    int tmp[8];
    __m256i vsum = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m128i va8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
        __m128i va16 = _mm_sub_epi16(_mm_cvtepu8_epi16(va8), vza16);
        __m128i vb16 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb8), vzb16);
        __m256i va32 = _mm256_cvtepi16_epi32(va16);
        __m256i vb32 = _mm256_cvtepi16_epi32(vb16);
        vsum = _mm256_add_epi32(vsum, _mm256_mullo_epi32(va32, vb32));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), vsum);
    for (int k = 0; k < 8; ++k) sum += tmp[k];
#elif defined(__NEON__)
    int32x4_t vsum_lo = vdupq_n_s32(0);
    int32x4_t vsum_hi = vdupq_n_s32(0);
    const int16x8_t vza16 = vdupq_n_s16(static_cast<int16_t>(zero_point_a));
    const int16x8_t vzb16 = vdupq_n_s16(static_cast<int16_t>(zero_point_b));
    for (; i + 8 <= n; i += 8) {
        int16x8_t va16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(a + i))), vza16);
        int16x8_t vb16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(b + i))), vzb16);
        vsum_lo = vaddq_s32(vsum_lo, vmull_s16(vget_low_s16(va16), vget_low_s16(vb16)));
        vsum_hi = vaddq_s32(vsum_hi, vmull_s16(vget_high_s16(va16), vget_high_s16(vb16)));
    }
    int tmp[4];
    vst1q_s32(tmp, vsum_lo);
    for (int k = 0; k < 4; ++k) sum += tmp[k];
    vst1q_s32(tmp, vsum_hi);
    for (int k = 0; k < 4; ++k) sum += tmp[k];
#endif
    for (; i < n; ++i) {
        sum += static_cast<std::int64_t>(static_cast<int>(a[i]) - zero_point_a) *
               static_cast<std::int64_t>(static_cast<int>(b[i]) - zero_point_b);
    }
    return static_cast<float>(sum) * scale_a * scale_b;
}

float cosine_sim_u8(const uint8_t* a, const uint8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b) {
    std::int64_t dot = 0;
    std::int64_t norm_a = 0;
    std::int64_t norm_b = 0;
    size_t i = 0;
#if defined(__AVX2__)
    const __m128i vza16 = _mm_set1_epi16(static_cast<short>(zero_point_a));
    const __m128i vzb16 = _mm_set1_epi16(static_cast<short>(zero_point_b));
    int tmp[8];
    __m256i vdot = _mm256_setzero_si256();
    __m256i vna = _mm256_setzero_si256();
    __m256i vnb = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m128i va8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
        __m128i va16 = _mm_sub_epi16(_mm_cvtepu8_epi16(va8), vza16);
        __m128i vb16 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb8), vzb16);
        __m256i va32 = _mm256_cvtepi16_epi32(va16);
        __m256i vb32 = _mm256_cvtepi16_epi32(vb16);
        vdot = _mm256_add_epi32(vdot, _mm256_mullo_epi32(va32, vb32));
        vna = _mm256_add_epi32(vna, _mm256_mullo_epi32(va32, va32));
        vnb = _mm256_add_epi32(vnb, _mm256_mullo_epi32(vb32, vb32));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), vdot);
    for (int k = 0; k < 8; ++k) dot += tmp[k];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), vna);
    for (int k = 0; k < 8; ++k) norm_a += tmp[k];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), vnb);
    for (int k = 0; k < 8; ++k) norm_b += tmp[k];
#elif defined(__NEON__)
    int32x4_t vdot_lo = vdupq_n_s32(0);
    int32x4_t vdot_hi = vdupq_n_s32(0);
    int32x4_t vna_lo = vdupq_n_s32(0);
    int32x4_t vna_hi = vdupq_n_s32(0);
    int32x4_t vnb_lo = vdupq_n_s32(0);
    int32x4_t vnb_hi = vdupq_n_s32(0);
    const int16x8_t vza16 = vdupq_n_s16(static_cast<int16_t>(zero_point_a));
    const int16x8_t vzb16 = vdupq_n_s16(static_cast<int16_t>(zero_point_b));
    for (; i + 8 <= n; i += 8) {
        int16x8_t va16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(a + i))), vza16);
        int16x8_t vb16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(b + i))), vzb16);
        vdot_lo = vaddq_s32(vdot_lo, vmull_s16(vget_low_s16(va16), vget_low_s16(vb16)));
        vdot_hi = vaddq_s32(vdot_hi, vmull_s16(vget_high_s16(va16), vget_high_s16(vb16)));
        vna_lo = vaddq_s32(vna_lo, vmull_s16(vget_low_s16(va16), vget_low_s16(va16)));
        vna_hi = vaddq_s32(vna_hi, vmull_s16(vget_high_s16(va16), vget_high_s16(va16)));
        vnb_lo = vaddq_s32(vnb_lo, vmull_s16(vget_low_s16(vb16), vget_low_s16(vb16)));
        vnb_hi = vaddq_s32(vnb_hi, vmull_s16(vget_high_s16(vb16), vget_high_s16(vb16)));
    }
    int tmp[4];
    vst1q_s32(tmp, vdot_lo); for (int k = 0; k < 4; ++k) dot += tmp[k];
    vst1q_s32(tmp, vdot_hi); for (int k = 0; k < 4; ++k) dot += tmp[k];
    vst1q_s32(tmp, vna_lo); for (int k = 0; k < 4; ++k) norm_a += tmp[k];
    vst1q_s32(tmp, vna_hi); for (int k = 0; k < 4; ++k) norm_a += tmp[k];
    vst1q_s32(tmp, vnb_lo); for (int k = 0; k < 4; ++k) norm_b += tmp[k];
    vst1q_s32(tmp, vnb_hi); for (int k = 0; k < 4; ++k) norm_b += tmp[k];
#endif
    for (; i < n; ++i) {
        const std::int64_t da = static_cast<int>(a[i]) - zero_point_a;
        const std::int64_t db = static_cast<int>(b[i]) - zero_point_b;
        dot += da * db;
        norm_a += da * da;
        norm_b += db * db;
    }
    return static_cast<float>(dot) / (std::sqrt(static_cast<float>(norm_a) * static_cast<float>(norm_b)) + 1e-8f);
}

float l1_dist_s8(const int8_t* a, const int8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b) {
    float sum = 0.0f;
    size_t i = 0;
#if defined(__AVX2__)
    const __m128i vza16 = _mm_set1_epi16(static_cast<short>(zero_point_a));
    const __m128i vzb16 = _mm_set1_epi16(static_cast<short>(zero_point_b));
    const __m256 vsa = _mm256_set1_ps(scale_a);
    const __m256 vsb = _mm256_set1_ps(scale_b);
    const __m256 vabs_mask = _mm256_set1_ps(-0.0f);
    float tmp[8];
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m128i va64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
        __m128i va16 = _mm_sub_epi16(_mm_cvtepi8_epi16(va64), vza16);
        __m128i vb16 = _mm_sub_epi16(_mm_cvtepi8_epi16(vb64), vzb16);
        __m256 fa = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va16)), vsa);
        __m256 fb = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb16)), vsb);
        __m256 diff = _mm256_sub_ps(fa, fb);
        diff = _mm256_andnot_ps(vabs_mask, diff);
        vsum = _mm256_add_ps(vsum, diff);
    }
    _mm256_storeu_ps(tmp, vsum);
    for (int k = 0; k < 8; ++k) sum += tmp[k];
#elif defined(__NEON__)
    const int16x8_t vza16 = vdupq_n_s16(static_cast<int16_t>(zero_point_a));
    const int16x8_t vzb16 = vdupq_n_s16(static_cast<int16_t>(zero_point_b));
    const float32x4_t vsa = vdupq_n_f32(scale_a);
    const float32x4_t vsb = vdupq_n_f32(scale_b);
    float tmp[4];
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        int16x8_t va16 = vsubq_s16(vmovl_s8(vld1_s8(a + i)), vza16);
        int16x8_t vb16 = vsubq_s16(vmovl_s8(vld1_s8(b + i)), vzb16);
        float32x4_t fa_lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(va16))), vsa);
        float32x4_t fa_hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(va16))), vsa);
        float32x4_t fb_lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vb16))), vsb);
        float32x4_t fb_hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vb16))), vsb);
        vsum = vaddq_f32(vsum, vabsq_f32(vsubq_f32(fa_lo, fb_lo)));
        vsum = vaddq_f32(vsum, vabsq_f32(vsubq_f32(fa_hi, fb_hi)));
    }
    vst1q_f32(tmp, vsum);
    for (int k = 0; k < 4; ++k) sum += tmp[k];
#endif
    for (; i < n; ++i) {
        const float da = (static_cast<int>(a[i]) - zero_point_a) * scale_a;
        const float db = (static_cast<int>(b[i]) - zero_point_b) * scale_b;
        sum += std::abs(da - db);
    }
    return sum;
}

float l2_dist_s8(const int8_t* a, const int8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b) {
    float sum = 0.0f;
    size_t i = 0;
#if defined(__AVX2__)
    const __m128i vza16 = _mm_set1_epi16(static_cast<short>(zero_point_a));
    const __m128i vzb16 = _mm_set1_epi16(static_cast<short>(zero_point_b));
    const __m256 vsa = _mm256_set1_ps(scale_a);
    const __m256 vsb = _mm256_set1_ps(scale_b);
    float tmp[8];
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m128i va64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
        __m128i va16 = _mm_sub_epi16(_mm_cvtepi8_epi16(va64), vza16);
        __m128i vb16 = _mm_sub_epi16(_mm_cvtepi8_epi16(vb64), vzb16);
        __m256 fa = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(va16)), vsa);
        __m256 fb = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(vb16)), vsb);
        __m256 diff = _mm256_sub_ps(fa, fb);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(diff, diff));
    }
    _mm256_storeu_ps(tmp, vsum);
    for (int k = 0; k < 8; ++k) sum += tmp[k];
#elif defined(__NEON__)
    const int16x8_t vza16 = vdupq_n_s16(static_cast<int16_t>(zero_point_a));
    const int16x8_t vzb16 = vdupq_n_s16(static_cast<int16_t>(zero_point_b));
    const float32x4_t vsa = vdupq_n_f32(scale_a);
    const float32x4_t vsb = vdupq_n_f32(scale_b);
    float tmp[4];
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        int16x8_t va16 = vsubq_s16(vmovl_s8(vld1_s8(a + i)), vza16);
        int16x8_t vb16 = vsubq_s16(vmovl_s8(vld1_s8(b + i)), vzb16);
        float32x4_t fa_lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(va16))), vsa);
        float32x4_t fa_hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(va16))), vsa);
        float32x4_t fb_lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vb16))), vsb);
        float32x4_t fb_hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vb16))), vsb);
        float32x4_t diff_lo = vsubq_f32(fa_lo, fb_lo);
        float32x4_t diff_hi = vsubq_f32(fa_hi, fb_hi);
        vsum = vmlaq_f32(vsum, diff_lo, diff_lo);
        vsum = vmlaq_f32(vsum, diff_hi, diff_hi);
    }
    vst1q_f32(tmp, vsum);
    for (int k = 0; k < 4; ++k) sum += tmp[k];
#endif
    for (; i < n; ++i) {
        const float da = (static_cast<int>(a[i]) - zero_point_a) * scale_a;
        const float db = (static_cast<int>(b[i]) - zero_point_b) * scale_b;
        const float diff = da - db;
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float dot_product_s8(const int8_t* a, const int8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b) {
    std::int64_t sum = 0;
    size_t i = 0;
#if defined(__AVX2__)
    const __m128i vza16 = _mm_set1_epi16(static_cast<short>(zero_point_a));
    const __m128i vzb16 = _mm_set1_epi16(static_cast<short>(zero_point_b));
    int tmp[8];
    __m256i vsum = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m128i va64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
        __m128i va16 = _mm_sub_epi16(_mm_cvtepi8_epi16(va64), vza16);
        __m128i vb16 = _mm_sub_epi16(_mm_cvtepi8_epi16(vb64), vzb16);
        __m256i va32 = _mm256_cvtepi16_epi32(va16);
        __m256i vb32 = _mm256_cvtepi16_epi32(vb16);
        vsum = _mm256_add_epi32(vsum, _mm256_mullo_epi32(va32, vb32));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), vsum);
    for (int k = 0; k < 8; ++k) sum += tmp[k];
#elif defined(__NEON__)
    int32x4_t vsum_lo = vdupq_n_s32(0);
    int32x4_t vsum_hi = vdupq_n_s32(0);
    const int16x8_t vza16 = vdupq_n_s16(static_cast<int16_t>(zero_point_a));
    const int16x8_t vzb16 = vdupq_n_s16(static_cast<int16_t>(zero_point_b));
    for (; i + 8 <= n; i += 8) {
        int16x8_t va16 = vsubq_s16(vmovl_s8(vld1_s8(a + i)), vza16);
        int16x8_t vb16 = vsubq_s16(vmovl_s8(vld1_s8(b + i)), vzb16);
        vsum_lo = vaddq_s32(vsum_lo, vmull_s16(vget_low_s16(va16), vget_low_s16(vb16)));
        vsum_hi = vaddq_s32(vsum_hi, vmull_s16(vget_high_s16(va16), vget_high_s16(vb16)));
    }
    int tmp[4];
    vst1q_s32(tmp, vsum_lo); for (int k = 0; k < 4; ++k) sum += tmp[k];
    vst1q_s32(tmp, vsum_hi); for (int k = 0; k < 4; ++k) sum += tmp[k];
#endif
    for (; i < n; ++i) {
        sum += static_cast<std::int64_t>(static_cast<int>(a[i]) - zero_point_a) *
               static_cast<std::int64_t>(static_cast<int>(b[i]) - zero_point_b);
    }
    return static_cast<float>(sum) * scale_a * scale_b;
}

float cosine_sim_s8(const int8_t* a, const int8_t* b, size_t n, float scale_a, int zero_point_a, float scale_b, int zero_point_b) {
    std::int64_t dot = 0;
    std::int64_t norm_a = 0;
    std::int64_t norm_b = 0;
    size_t i = 0;
#if defined(__AVX2__)
    const __m128i vza16 = _mm_set1_epi16(static_cast<short>(zero_point_a));
    const __m128i vzb16 = _mm_set1_epi16(static_cast<short>(zero_point_b));
    int tmp[8];
    __m256i vdot = _mm256_setzero_si256();
    __m256i vna = _mm256_setzero_si256();
    __m256i vnb = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m128i va64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
        __m128i va16 = _mm_sub_epi16(_mm_cvtepi8_epi16(va64), vza16);
        __m128i vb16 = _mm_sub_epi16(_mm_cvtepi8_epi16(vb64), vzb16);
        __m256i va32 = _mm256_cvtepi16_epi32(va16);
        __m256i vb32 = _mm256_cvtepi16_epi32(vb16);
        vdot = _mm256_add_epi32(vdot, _mm256_mullo_epi32(va32, vb32));
        vna = _mm256_add_epi32(vna, _mm256_mullo_epi32(va32, va32));
        vnb = _mm256_add_epi32(vnb, _mm256_mullo_epi32(vb32, vb32));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), vdot);
    for (int k = 0; k < 8; ++k) dot += tmp[k];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), vna);
    for (int k = 0; k < 8; ++k) norm_a += tmp[k];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), vnb);
    for (int k = 0; k < 8; ++k) norm_b += tmp[k];
#elif defined(__NEON__)
    int32x4_t vdot_lo = vdupq_n_s32(0);
    int32x4_t vdot_hi = vdupq_n_s32(0);
    int32x4_t vna_lo = vdupq_n_s32(0);
    int32x4_t vna_hi = vdupq_n_s32(0);
    int32x4_t vnb_lo = vdupq_n_s32(0);
    int32x4_t vnb_hi = vdupq_n_s32(0);
    const int16x8_t vza16 = vdupq_n_s16(static_cast<int16_t>(zero_point_a));
    const int16x8_t vzb16 = vdupq_n_s16(static_cast<int16_t>(zero_point_b));
    for (; i + 8 <= n; i += 8) {
        int16x8_t va16 = vsubq_s16(vmovl_s8(vld1_s8(a + i)), vza16);
        int16x8_t vb16 = vsubq_s16(vmovl_s8(vld1_s8(b + i)), vzb16);
        vdot_lo = vaddq_s32(vdot_lo, vmull_s16(vget_low_s16(va16), vget_low_s16(vb16)));
        vdot_hi = vaddq_s32(vdot_hi, vmull_s16(vget_high_s16(va16), vget_high_s16(vb16)));
        vna_lo = vaddq_s32(vna_lo, vmull_s16(vget_low_s16(va16), vget_low_s16(va16)));
        vna_hi = vaddq_s32(vna_hi, vmull_s16(vget_high_s16(va16), vget_high_s16(va16)));
        vnb_lo = vaddq_s32(vnb_lo, vmull_s16(vget_low_s16(vb16), vget_low_s16(vb16)));
        vnb_hi = vaddq_s32(vnb_hi, vmull_s16(vget_high_s16(vb16), vget_high_s16(vb16)));
    }
    int tmp[4];
    vst1q_s32(tmp, vdot_lo); for (int k = 0; k < 4; ++k) dot += tmp[k];
    vst1q_s32(tmp, vdot_hi); for (int k = 0; k < 4; ++k) dot += tmp[k];
    vst1q_s32(tmp, vna_lo); for (int k = 0; k < 4; ++k) norm_a += tmp[k];
    vst1q_s32(tmp, vna_hi); for (int k = 0; k < 4; ++k) norm_a += tmp[k];
    vst1q_s32(tmp, vnb_lo); for (int k = 0; k < 4; ++k) norm_b += tmp[k];
    vst1q_s32(tmp, vnb_hi); for (int k = 0; k < 4; ++k) norm_b += tmp[k];
#endif
    for (; i < n; ++i) {
        const std::int64_t da = static_cast<int>(a[i]) - zero_point_a;
        const std::int64_t db = static_cast<int>(b[i]) - zero_point_b;
        dot += da * db;
        norm_a += da * da;
        norm_b += db * db;
    }
    return static_cast<float>(dot) / (std::sqrt(static_cast<float>(norm_a) * static_cast<float>(norm_b)) + 1e-8f);
}

//////////////////////////////////////////////////////////////
// Vector-Matrix Distance Functions
//////////////////////////////////////////////////////////////

} // namespace LMStore
