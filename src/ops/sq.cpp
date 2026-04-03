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

    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);

        __m256 q = _mm256_mul_ps(x, v_inv_scale);
        q = _mm256_round_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        q = _mm256_add_ps(q, v_zp);

        // clamp
        q = _mm256_max_ps(v_min, _mm256_min_ps(v_max, q));

        // convert to int32
        __m256i qi32 = _mm256_cvtps_epi32(q);

        // pack int32 -> int16 -> uint8
        __m128i lo = _mm256_extracti128_si256(qi32, 0);
        __m128i hi = _mm256_extracti128_si256(qi32, 1);

        __m128i i16 = _mm_packs_epi32(lo, hi);
        __m128i i8  = _mm_packus_epi16(i16, i16);   // 16x uint8 (duplicate)

        // store lower 8 bytes
        std::memcpy(output + i, &i8, 8);
    }
#elif defined(__NEON__)
    float32x4_t v_inv_scale = vdupq_n_f32(inv_scale);
    float32x4_t v_zp = vdupq_n_f32(static_cast<float>(zero_point));
    float32x4_t v_min = vdupq_n_f32(0.0f);
    float32x4_t v_max = vdupq_n_f32(255.0f);

    for (; i + 8 <= size; i += 8) {
        float32x4_t x0 = vld1q_f32(input + i);
        float32x4_t x1 = vld1q_f32(input + i + 4);

        float32x4_t q0 = vmulq_f32(x0, v_inv_scale);
        float32x4_t q1 = vmulq_f32(x1, v_inv_scale);

        // round
        q0 = vrndnq_f32(q0);
        q1 = vrndnq_f32(q1);

        q0 = vaddq_f32(q0, v_zp);
        q1 = vaddq_f32(q1, v_zp);

        // clamp
        q0 = vmaxq_f32(v_min, vminq_f32(v_max, q0));
        q1 = vmaxq_f32(v_min, vminq_f32(v_max, q1));

        // float -> int32
        int32x4_t i0 = vcvtq_s32_f32(q0);
        int32x4_t i1 = vcvtq_s32_f32(q1);

        // int32 -> int16 -> uint8
        int16x8_t i16 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        uint8x8_t u8 = vqmovun_s16(i16);

        vst1_u8(output + i, u8);
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

    for (; i + 8 <= size; i += 8) {
        // load 8 uint8 -> expand to int32
        __m128i u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input + i));
        __m256i i32 = _mm256_cvtepu8_epi32(u8);

        // subtract zero_point
        i32 = _mm256_sub_epi32(i32, v_zp);

        // int -> float
        __m256 f = _mm256_cvtepi32_ps(i32);

        // scale
        f = _mm256_mul_ps(f, v_scale);

        _mm256_storeu_ps(output + i, f);
    }
#elif defined(__NEON__)
    float32x4_t v_scale = vdupq_n_f32(scale);
    int32x4_t v_zp = vdupq_n_s32(zero_point);

    for (; i + 8 <= size; i += 8) {
        uint8x8_t u8 = vld1_u8(input + i);

        uint16x8_t u16 = vmovl_u8(u8);
        uint32x4_t lo = vmovl_u16(vget_low_u16(u16));
        uint32x4_t hi = vmovl_u16(vget_high_u16(u16));

        int32x4_t i0 = vreinterpretq_s32_u32(lo);
        int32x4_t i1 = vreinterpretq_s32_u32(hi);

        i0 = vsubq_s32(i0, v_zp);
        i1 = vsubq_s32(i1, v_zp);

        float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(i0), v_scale);
        float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(i1), v_scale);

        vst1q_f32(output + i, f0);
        vst1q_f32(output + i + 4, f1);
    }
#endif
    // tail/fallback
    for (; i < size; ++i) {
        output[i] = scale * (static_cast<int>(input[i]) - zero_point);
    }
}

} // namespace LMStore