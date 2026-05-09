#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include "ops/ops.h"
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__NEON__)
#include <arm_neon.h>
#endif
#include <cmath>

namespace LMStore {

void scalar_quantize_per_tensor_u8(
    const float* input,
    uint8_t* output,
    size_t size,
    float scale,
    int zero_point)
{
    const float inv_scale = 1.0f / scale;
    size_t i = 0;

#if defined(__AVX2__)
    __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
    __m256 v_zp = _mm256_set1_ps(static_cast<float>(zero_point));
    __m256 v_min = _mm256_set1_ps(0.0f);
    __m256 v_max = _mm256_set1_ps(255.0f);

    for (; i + 16 <= size; i += 16) {
        __m256 x0 = _mm256_loadu_ps(input + i);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);

        __m256 q0 = _mm256_mul_ps(x0, v_inv_scale);
        __m256 q1 = _mm256_mul_ps(x1, v_inv_scale);
        q0 = _mm256_round_ps(q0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        q1 = _mm256_round_ps(q1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        q0 = _mm256_add_ps(q0, v_zp);
        q1 = _mm256_add_ps(q1, v_zp);
        q0 = _mm256_max_ps(v_min, _mm256_min_ps(v_max, q0));
        q1 = _mm256_max_ps(v_min, _mm256_min_ps(v_max, q1));

        __m256i qi320 = _mm256_cvtps_epi32(q0);
        __m256i qi321 = _mm256_cvtps_epi32(q1);
        __m128i lo0 = _mm256_extracti128_si256(qi320, 0);
        __m128i hi0 = _mm256_extracti128_si256(qi320, 1);
        __m128i lo1 = _mm256_extracti128_si256(qi321, 0);
        __m128i hi1 = _mm256_extracti128_si256(qi321, 1);
        __m128i i16 = _mm_packs_epi32(_mm_packs_epi32(lo0, hi0), _mm_packs_epi32(lo1, hi1));
        __m128i u8 = _mm_packus_epi16(i16, i16);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), u8);
    }
#elif defined(__NEON__)
    float32x4_t v_inv_scale = vdupq_n_f32(inv_scale);
    float32x4_t v_zp = vdupq_n_f32(static_cast<float>(zero_point));
    float32x4_t v_min = vdupq_n_f32(0.0f);
    float32x4_t v_max = vdupq_n_f32(255.0f);

    for (; i + 16 <= size; i += 16) {
        float32x4_t x0 = vld1q_f32(input + i);
        float32x4_t x1 = vld1q_f32(input + i + 4);
        float32x4_t x2 = vld1q_f32(input + i + 8);
        float32x4_t x3 = vld1q_f32(input + i + 12);

        float32x4_t q0 = vmulq_f32(x0, v_inv_scale);
        float32x4_t q1 = vmulq_f32(x1, v_inv_scale);
        float32x4_t q2 = vmulq_f32(x2, v_inv_scale);
        float32x4_t q3 = vmulq_f32(x3, v_inv_scale);

        q0 = vrndnq_f32(q0);
        q1 = vrndnq_f32(q1);
        q2 = vrndnq_f32(q2);
        q3 = vrndnq_f32(q3);

        q0 = vaddq_f32(q0, v_zp);
        q1 = vaddq_f32(q1, v_zp);
        q2 = vaddq_f32(q2, v_zp);
        q3 = vaddq_f32(q3, v_zp);

        q0 = vmaxq_f32(v_min, vminq_f32(v_max, q0));
        q1 = vmaxq_f32(v_min, vminq_f32(v_max, q1));
        q2 = vmaxq_f32(v_min, vminq_f32(v_max, q2));
        q3 = vmaxq_f32(v_min, vminq_f32(v_max, q3));

        int32x4_t i0 = vcvtq_s32_f32(q0);
        int32x4_t i1 = vcvtq_s32_f32(q1);
        int32x4_t i2 = vcvtq_s32_f32(q2);
        int32x4_t i3 = vcvtq_s32_f32(q3);

        int16x8_t i16 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        int16x8_t i17 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        uint8x8_t u80 = vqmovun_s16(i16);
        uint8x8_t u81 = vqmovun_s16(i17);
        vst1_u8(output + i, u80);
        vst1_u8(output + i + 8, u81);
    }
#endif
    // tail/fallback
    for (; i < size; ++i) {
        float q = std::round(input[i] * inv_scale) + zero_point;
        q = std::max(0.0f, std::min(255.0f, q));
        output[i] = static_cast<uint8_t>(q);
    }
}

void scalar_dequantize_per_tensor_u8(
    const uint8_t* input,
    float* output,
    size_t size,
    float scale,
    int zero_point)
{
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256i v_zp = _mm256_set1_epi32(zero_point);

    for (; i + 16 <= size; i += 16) {
        __m128i u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        __m128i lo8 = u8;
        __m128i hi8 = _mm_srli_si128(u8, 8);
        __m256i i320 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(lo8), v_zp);
        __m256i i321 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(hi8), v_zp);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_cvtepi32_ps(i320), v_scale));
        _mm256_storeu_ps(output + i + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(i321), v_scale));
    }
#elif defined(__NEON__)
    float32x4_t v_scale = vdupq_n_f32(scale);
    int32x4_t v_zp = vdupq_n_s32(zero_point);

    for (; i + 16 <= size; i += 16) {
        uint8x16_t u8 = vld1q_u8(input + i);
        uint16x8_t u160 = vmovl_u8(vget_low_u8(u8));
        uint16x8_t u161 = vmovl_u8(vget_high_u8(u8));
        uint32x4_t lo0 = vmovl_u16(vget_low_u16(u160));
        uint32x4_t hi0 = vmovl_u16(vget_high_u16(u160));
        uint32x4_t lo1 = vmovl_u16(vget_low_u16(u161));
        uint32x4_t hi1 = vmovl_u16(vget_high_u16(u161));
        float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(lo0), v_zp)), v_scale);
        float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(hi0), v_zp)), v_scale);
        float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(lo1), v_zp)), v_scale);
        float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(hi1), v_zp)), v_scale);
        vst1q_f32(output + i, f0);
        vst1q_f32(output + i + 4, f1);
        vst1q_f32(output + i + 8, f2);
        vst1q_f32(output + i + 12, f3);
    }
#endif
    // tail/fallback
    for (; i < size; ++i) {
        output[i] = scale * (static_cast<int>(input[i]) - zero_point);
    }
}

#if defined(LMSTORE_HAS_FP16)
void scalar_dequantize_per_tensor_u8(
    const uint8_t* input,
    fp16_t* output,
    size_t size,
    float scale,
    int zero_point)
{
    size_t i = 0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    const float32x4_t v_scale_f32 = vdupq_n_f32(scale);
    const int32x4_t v_zp = vdupq_n_s32(zero_point);

    for (; i + 16 <= size; i += 16) {
        uint8x16_t u8 = vld1q_u8(input + i);
        uint16x8_t u160 = vmovl_u8(vget_low_u8(u8));
        uint16x8_t u161 = vmovl_u8(vget_high_u8(u8));
        uint32x4_t lo0 = vmovl_u16(vget_low_u16(u160));
        uint32x4_t hi0 = vmovl_u16(vget_high_u16(u160));
        uint32x4_t lo1 = vmovl_u16(vget_low_u16(u161));
        uint32x4_t hi1 = vmovl_u16(vget_high_u16(u161));

        float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(lo0), v_zp)), v_scale_f32);
        float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(hi0), v_zp)), v_scale_f32);
        float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(lo1), v_zp)), v_scale_f32);
        float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(hi1), v_zp)), v_scale_f32);

        vst1_f16(output + i, vcvt_f16_f32(f0));
        vst1_f16(output + i + 4, vcvt_f16_f32(f1));
        vst1_f16(output + i + 8, vcvt_f16_f32(f2));
        vst1_f16(output + i + 12, vcvt_f16_f32(f3));
    }
#endif
    for (; i < size; ++i) {
        output[i] = static_cast<fp16_t>(scale * (static_cast<int>(input[i]) - zero_point));
    }
}
#endif

void scalar_dequantize_per_tensor_i8(
    const int8_t* input,
    float* output,
    size_t size,
    float scale,
    int zero_point) {
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256i v_zp = _mm256_set1_epi32(zero_point);

    for (; i + 16 <= size; i += 16) {
        __m128i i8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        __m128i lo8 = i8;
        __m128i hi8 = _mm_srli_si128(i8, 8);
        __m256i i320 = _mm256_sub_epi32(_mm256_cvtepi8_epi32(lo8), v_zp);
        __m256i i321 = _mm256_sub_epi32(_mm256_cvtepi8_epi32(hi8), v_zp);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_cvtepi32_ps(i320), v_scale));
        _mm256_storeu_ps(output + i + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(i321), v_scale));
    }
#elif defined(__NEON__)
    float32x4_t v_scale = vdupq_n_f32(scale);
    int32x4_t v_zp = vdupq_n_s32(zero_point);

    for (; i + 16 <= size; i += 16) {
        int8x16_t i8 = vld1q_s8(input + i);
        int16x8_t i160 = vmovl_s8(vget_low_s8(i8));
        int16x8_t i161 = vmovl_s8(vget_high_s8(i8));
        int32x4_t i0 = vmovl_s16(vget_low_s16(i160));
        int32x4_t i1 = vmovl_s16(vget_high_s16(i160));
        int32x4_t i2 = vmovl_s16(vget_low_s16(i161));
        int32x4_t i3 = vmovl_s16(vget_high_s16(i161));
        vst1q_f32(output + i, vmulq_f32(vcvtq_f32_s32(vsubq_s32(i0, v_zp)), v_scale));
        vst1q_f32(output + i + 4, vmulq_f32(vcvtq_f32_s32(vsubq_s32(i1, v_zp)), v_scale));
        vst1q_f32(output + i + 8, vmulq_f32(vcvtq_f32_s32(vsubq_s32(i2, v_zp)), v_scale));
        vst1q_f32(output + i + 12, vmulq_f32(vcvtq_f32_s32(vsubq_s32(i3, v_zp)), v_scale));
    }
#endif
    for (; i < size; ++i) {
        output[i] = scale * (static_cast<int>(input[i]) - zero_point);
    }
}

void scalar_dequantize_per_tensor_u16(
    const uint16_t* input,
    float* output,
    size_t size,
    float scale,
    int zero_point) {
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256i v_zp = _mm256_set1_epi32(zero_point);

    for (; i + 16 <= size; i += 16) {
        __m128i u160 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        __m128i u161 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i + 8));
        __m256i i320 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(u160), v_zp);
        __m256i i321 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(u161), v_zp);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_cvtepi32_ps(i320), v_scale));
        _mm256_storeu_ps(output + i + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(i321), v_scale));
    }
#elif defined(__NEON__)
    float32x4_t v_scale = vdupq_n_f32(scale);
    int32x4_t v_zp = vdupq_n_s32(zero_point);

    for (; i + 16 <= size; i += 16) {
        uint16x8_t u160 = vld1q_u16(input + i);
        uint16x8_t u161 = vld1q_u16(input + i + 8);
        uint32x4_t lo0 = vmovl_u16(vget_low_u16(u160));
        uint32x4_t hi0 = vmovl_u16(vget_high_u16(u160));
        uint32x4_t lo1 = vmovl_u16(vget_low_u16(u161));
        uint32x4_t hi1 = vmovl_u16(vget_high_u16(u161));
        vst1q_f32(output + i, vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(lo0), v_zp)), v_scale));
        vst1q_f32(output + i + 4, vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(hi0), v_zp)), v_scale));
        vst1q_f32(output + i + 8, vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(lo1), v_zp)), v_scale));
        vst1q_f32(output + i + 12, vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(hi1), v_zp)), v_scale));
    }
#endif
    for (; i < size; ++i) {
        output[i] = scale * (static_cast<int>(input[i]) - zero_point);
    }
}

void scalar_dequantize_per_tensor_i16(
    const int16_t* input,
    float* output,
    size_t size,
    float scale,
    int zero_point) {
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256i v_zp = _mm256_set1_epi32(zero_point);

    for (; i + 16 <= size; i += 16) {
        __m128i i160 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        __m128i i161 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i + 8));
        __m256i i320 = _mm256_sub_epi32(_mm256_cvtepi16_epi32(i160), v_zp);
        __m256i i321 = _mm256_sub_epi32(_mm256_cvtepi16_epi32(i161), v_zp);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_cvtepi32_ps(i320), v_scale));
        _mm256_storeu_ps(output + i + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(i321), v_scale));
    }
#elif defined(__NEON__)
    float32x4_t v_scale = vdupq_n_f32(scale);
    int32x4_t v_zp = vdupq_n_s32(zero_point);

    for (; i + 16 <= size; i += 16) {
        int16x8_t i160 = vld1q_s16(input + i);
        int16x8_t i161 = vld1q_s16(input + i + 8);
        int32x4_t i0 = vmovl_s16(vget_low_s16(i160));
        int32x4_t i1 = vmovl_s16(vget_high_s16(i160));
        int32x4_t i2 = vmovl_s16(vget_low_s16(i161));
        int32x4_t i3 = vmovl_s16(vget_high_s16(i161));
        vst1q_f32(output + i, vmulq_f32(vcvtq_f32_s32(vsubq_s32(i0, v_zp)), v_scale));
        vst1q_f32(output + i + 4, vmulq_f32(vcvtq_f32_s32(vsubq_s32(i1, v_zp)), v_scale));
        vst1q_f32(output + i + 8, vmulq_f32(vcvtq_f32_s32(vsubq_s32(i2, v_zp)), v_scale));
        vst1q_f32(output + i + 12, vmulq_f32(vcvtq_f32_s32(vsubq_s32(i3, v_zp)), v_scale));
    }
#endif
    for (; i < size; ++i) {
        output[i] = scale * (static_cast<int>(input[i]) - zero_point);
    }
}

} // namespace LMStore
