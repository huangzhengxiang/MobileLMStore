#include <cstddef>
#include <cmath>
#include "RopeConfigParser.h"
#include "ops/ops.h"
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__NEON__)
#include <arm_neon.h>
#endif

namespace LMStore {

namespace {

inline float maybe_apply_scaling(
    float freq,
    int scale_factor,
    int high_freq_factor
) {
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kOldContextLen = 8192.0f;
    constexpr float kLowFreqFactor = 1.0f;

    const float low_freq_wavelen = kOldContextLen / kLowFreqFactor;
    const float high_freq_wavelen = kOldContextLen / static_cast<float>(high_freq_factor);
    const float wavelen = 2.0f * kPi / freq;

    if (wavelen < high_freq_wavelen) {
        return freq;
    }
    if (wavelen > low_freq_wavelen) {
        return freq / static_cast<float>(scale_factor);
    }

    const float smooth = (kOldContextLen / wavelen - kLowFreqFactor) /
                         (static_cast<float>(high_freq_factor) - kLowFreqFactor);
    return (1.0f - smooth) * (freq / static_cast<float>(scale_factor)) + smooth * freq;
}

} // namespace

void precompute_rope_freqs_cis(
    float* freqs_cos,                 // [L, D/2]
    float* freqs_sin,                 // [L, D/2]
    int L,
    int D,
    float theta,
    bool use_hf_rope,
    float partial_rotary_factor,
    bool use_scaled_rope,
    int scale_factor,
    int high_freq_factor
) {
    if (freqs_cos == nullptr || freqs_sin == nullptr || L <= 0 || D <= 0 || theta <= 0.0f) {
        return;
    }

    const int half = D / 2;
    if (half <= 0) {
        return;
    }

    // Default to identity rotation for all dims first.
    for (int l = 0; l < L; ++l) {
        float* cos_row = freqs_cos + l * half;
        float* sin_row = freqs_sin + l * half;
        for (int h = 0; h < half; ++h) {
            cos_row[h] = 1.0f;
            sin_row[h] = 0.0f;
        }
    }

    int rotary_dim = D;
    if (use_hf_rope) {
        rotary_dim = static_cast<int>(static_cast<float>(D) * partial_rotary_factor);
        if (rotary_dim < 0) {
            rotary_dim = 0;
        }
        if (rotary_dim > D) {
            rotary_dim = D;
        }
        // Must be even because RoPE pairs dimensions.
        rotary_dim = (rotary_dim / 2) * 2;
    }

    const int rotary_half = rotary_dim / 2;
    if (rotary_half <= 0) {
        return;
    }

    const float denom_dim = static_cast<float>(use_hf_rope ? rotary_dim : D);
    for (int h = 0; h < rotary_half; ++h) {
        const float exponent = static_cast<float>(2 * h) / denom_dim;
        float freq = 1.0f / std::pow(theta, exponent);
        if (!use_hf_rope && use_scaled_rope && scale_factor > 0 && high_freq_factor > 1) {
            freq = maybe_apply_scaling(freq, scale_factor, high_freq_factor);
        }

        for (int l = 0; l < L; ++l) {
            const float angle = static_cast<float>(l) * freq;
            freqs_cos[l * half + h] = std::cos(angle);
            freqs_sin[l * half + h] = std::sin(angle);
        }
    }
}

void precompute_rope_freqs_cis(
    float* freqs_cos,                 // [L, head_dim/2]
    float* freqs_sin,                 // [L, head_dim/2]
    int L,
    const rope_config_params& rope_cfg
) {
    precompute_rope_freqs_cis(
        freqs_cos,
        freqs_sin,
        L,
        rope_cfg.head_dim,
        rope_cfg.rope_theta,
        rope_cfg.use_hf_rope,
        rope_cfg.partial_rotary_factor,
        rope_cfg.use_scaled_rope,
        rope_cfg.rope_scale_factor,
        rope_cfg.high_freq_factor
    );
}


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
