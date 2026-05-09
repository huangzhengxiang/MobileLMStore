#include <cstddef>
#include <cstring>
#include "ops/ops.h"
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__NEON__)
#include <arm_neon.h>
#endif

namespace LMStore {

namespace {

#if defined(__AVX2__)
inline void transpose4x4_scale_store_sse(
    const float* src,
    int src_stride,
    float* dst,
    int dst_stride,
    float scale) {
    __m128 row0 = _mm_loadu_ps(src + 0 * src_stride);
    __m128 row1 = _mm_loadu_ps(src + 1 * src_stride);
    __m128 row2 = _mm_loadu_ps(src + 2 * src_stride);
    __m128 row3 = _mm_loadu_ps(src + 3 * src_stride);
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
    const __m128 scale_vec = _mm_set1_ps(scale);
    _mm_storeu_ps(dst + 0 * dst_stride, _mm_mul_ps(row0, scale_vec));
    _mm_storeu_ps(dst + 1 * dst_stride, _mm_mul_ps(row1, scale_vec));
    _mm_storeu_ps(dst + 2 * dst_stride, _mm_mul_ps(row2, scale_vec));
    _mm_storeu_ps(dst + 3 * dst_stride, _mm_mul_ps(row3, scale_vec));
}
#elif defined(__NEON__)
inline void transpose4x4_scale_store_neon(
    const float* src,
    int src_stride,
    float* dst,
    int dst_stride,
    float scale) {
    float32x4_t row0 = vld1q_f32(src + 0 * src_stride);
    float32x4_t row1 = vld1q_f32(src + 1 * src_stride);
    float32x4_t row2 = vld1q_f32(src + 2 * src_stride);
    float32x4_t row3 = vld1q_f32(src + 3 * src_stride);
    float32x4x2_t t0 = vtrnq_f32(row0, row1);
    float32x4x2_t t1 = vtrnq_f32(row2, row3);
    float32x2_t a0 = vget_low_f32(t0.val[0]);
    float32x2_t a1 = vget_low_f32(t1.val[0]);
    float32x2_t a2 = vget_low_f32(t0.val[1]);
    float32x2_t a3 = vget_low_f32(t1.val[1]);
    float32x2_t b0 = vget_high_f32(t0.val[0]);
    float32x2_t b1 = vget_high_f32(t1.val[0]);
    float32x2_t b2 = vget_high_f32(t0.val[1]);
    float32x2_t b3 = vget_high_f32(t1.val[1]);
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    vst1q_f32(dst + 0 * dst_stride, vmulq_f32(vcombine_f32(a0, a1), scale_vec));
    vst1q_f32(dst + 1 * dst_stride, vmulq_f32(vcombine_f32(a2, a3), scale_vec));
    vst1q_f32(dst + 2 * dst_stride, vmulq_f32(vcombine_f32(b0, b1), scale_vec));
    vst1q_f32(dst + 3 * dst_stride, vmulq_f32(vcombine_f32(b2, b3), scale_vec));
}
#endif

} // namespace

void fwht(float* tensor, int n) {
    // input can equal to output for in-place FWHT
    if (n >= 2) {
#if defined(__NEON__)
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            float32x4x2_t x = vld2q_f32(tensor + i);
            float32x4_t sum = vaddq_f32(x.val[0], x.val[1]);
            float32x4_t diff = vsubq_f32(x.val[0], x.val[1]);
            float32x4x2_t y;
            y.val[0] = sum;
            y.val[1] = diff;
            vst2q_f32(tensor + i, y);
        }
        for (; i < n; i += 2) {
            float u = tensor[i];
            float v = tensor[i + 1];
            tensor[i] = u + v;
            tensor[i + 1] = u - v;
        }
#elif defined(__AVX2__)
        int i = 0;
        const __m256 even_mask = _mm256_castsi256_ps(
            _mm256_setr_epi32(-1, 0, -1, 0, -1, 0, -1, 0));
        const __m256 odd_mask = _mm256_castsi256_ps(
            _mm256_setr_epi32(0, -1, 0, -1, 0, -1, 0, -1));
        for (; i + 8 <= n; i += 8) {
            __m256 x = _mm256_loadu_ps(tensor + i);
            __m256 even = _mm256_and_ps(x, even_mask);
            __m256 odd = _mm256_and_ps(x, odd_mask);
            __m256 odd_swapped = _mm256_permute_ps(odd, 0xB1);
            __m256 sum = _mm256_add_ps(even, odd_swapped);
            __m256 diff = _mm256_sub_ps(even, odd_swapped);
            __m256 out = _mm256_blend_ps(sum, diff, 0xAA);
            _mm256_storeu_ps(tensor + i, out);
        }
        for (; i < n; i += 2) {
            float u = tensor[i];
            float v = tensor[i + 1];
            tensor[i] = u + v;
            tensor[i + 1] = u - v;
        }
#else
        for (int i = 0; i < n; i += 2) {
            float u = tensor[i];
            float v = tensor[i + 1];
            tensor[i] = u + v;
            tensor[i + 1] = u - v;
        }
#endif
    }

    if (n >= 4) {
#if defined(__NEON__)
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            float32x4_t x0 = vld1q_f32(tensor + i);
            float32x4_t x1 = vld1q_f32(tensor + i + 4);

            float32x2_t a0 = vget_low_f32(x0);
            float32x2_t b0 = vget_high_f32(x0);
            float32x2_t a1 = vget_low_f32(x1);
            float32x2_t b1 = vget_high_f32(x1);

            vst1_f32(tensor + i, vadd_f32(a0, b0));
            vst1_f32(tensor + i + 2, vsub_f32(a0, b0));
            vst1_f32(tensor + i + 4, vadd_f32(a1, b1));
            vst1_f32(tensor + i + 6, vsub_f32(a1, b1));
        }
        for (; i < n; i += 4) {
            float u0 = tensor[i];
            float u1 = tensor[i + 1];
            float v0 = tensor[i + 2];
            float v1 = tensor[i + 3];
            tensor[i] = u0 + v0;
            tensor[i + 1] = u1 + v1;
            tensor[i + 2] = u0 - v0;
            tensor[i + 3] = u1 - v1;
        }
#elif defined(__AVX2__)
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 x = _mm256_loadu_ps(tensor + i);
            __m256 swapped = _mm256_permute_ps(x, 0x4E);
            __m256 sum = _mm256_add_ps(x, swapped);
            __m256 diff = _mm256_sub_ps(x, swapped);
            __m256 out = _mm256_blend_ps(sum, diff, 0xCC);
            _mm256_storeu_ps(tensor + i, out);
        }
        for (; i < n; i += 4) {
            float u0 = tensor[i];
            float u1 = tensor[i + 1];
            float v0 = tensor[i + 2];
            float v1 = tensor[i + 3];
            tensor[i] = u0 + v0;
            tensor[i + 1] = u1 + v1;
            tensor[i + 2] = u0 - v0;
            tensor[i + 3] = u1 - v1;
        }
#else
        for (int i = 0; i < n; i += 4) {
            float u0 = tensor[i];
            float u1 = tensor[i + 1];
            float v0 = tensor[i + 2];
            float v1 = tensor[i + 3];
            tensor[i] = u0 + v0;
            tensor[i + 1] = u1 + v1;
            tensor[i + 2] = u0 - v0;
            tensor[i + 3] = u1 - v1;
        }
#endif
    }

    if (n >= 8) {
#if defined(__NEON__)
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            float32x4_t a = vld1q_f32(tensor + i);
            float32x4_t b = vld1q_f32(tensor + i + 4);
            vst1q_f32(tensor + i, vaddq_f32(a, b));
            vst1q_f32(tensor + i + 4, vsubq_f32(a, b));
        }
        for (; i < n; i += 8) {
            for (int j = 0; j < 4; ++j) {
                float u = tensor[i + j];
                float v = tensor[i + j + 4];
                tensor[i + j] = u + v;
                tensor[i + j + 4] = u - v;
            }
        }
#elif defined(__AVX2__)
        for (int i = 0; i < n; i += 8) {
            __m128 a0 = _mm_loadu_ps(tensor + i);
            __m128 a1 = _mm_loadu_ps(tensor + i + 4);
            _mm_storeu_ps(tensor + i, _mm_add_ps(a0, a1));
            _mm_storeu_ps(tensor + i + 4, _mm_sub_ps(a0, a1));
        }
#else
        for (int i = 0; i < n; i += 8) {
            for (int j = 0; j < 4; ++j) {
                float u = tensor[i + j];
                float v = tensor[i + j + 4];
                tensor[i + j] = u + v;
                tensor[i + j + 4] = u - v;
            }
        }
#endif
    }

    for (int len = 8; len < n; len *= 2) {
        for (int i = 0; i < n; i += 2 * len) {
            int j = 0;
#if defined(__AVX2__)
            for (; j + 8 <= len; j += 8) {
                __m256 u = _mm256_loadu_ps(tensor + i + j);
                __m256 v = _mm256_loadu_ps(tensor + i + j + len);
                _mm256_storeu_ps(tensor + i + j, _mm256_add_ps(u, v));
                _mm256_storeu_ps(tensor + i + j + len, _mm256_sub_ps(u, v));
            }
#elif defined(__NEON__)
            for (; j + 8 <= len; j += 8) {
                float32x4_t u0 = vld1q_f32(tensor + i + j);
                float32x4_t v0 = vld1q_f32(tensor + i + j + len);
                float32x4_t u1 = vld1q_f32(tensor + i + j + 4);
                float32x4_t v1 = vld1q_f32(tensor + i + j + len + 4);
                vst1q_f32(tensor + i + j, vaddq_f32(u0, v0));
                vst1q_f32(tensor + i + j + len, vsubq_f32(u0, v0));
                vst1q_f32(tensor + i + j + 4, vaddq_f32(u1, v1));
                vst1q_f32(tensor + i + j + len + 4, vsubq_f32(u1, v1));
            }
#endif
            for (; j < len; ++j) {
                float u = tensor[i + j];
                float v = tensor[i + j + len];

                tensor[i + j] = u + v;
                tensor[i + j + len] = u - v;
            }
        }
    }
}

// Out-of-place 3D FWHT
// input: [B,L,D] in row-major ([B,D,L] if transpose=true)
// output: [B,L,D] in row-major
// transpose: if true, input shape is [B,D,L] (swap L and D)
void fwht_tensor(float* input, float* output, 
                 int B, int L, int D, 
                 bool transpose) {
    if (transpose) {
        // [B,D,L] -> [B,L,D]
        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                for (int d = 0; d < D; ++d) {
                    output[b * D * L + l * D + d] = input[b * L * D + d * L + l];
                }
                float* slice_out = output + b * D * L + l * D;
                fwht(slice_out, D);
            }
        }
    }
    else {
        bool inplace = (input == output);
        // normal case: [B,L,D] -> [B,L,D]
        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                float* slice_in = input + (b * L + l) * D;
                float* slice_out = output + (b * L + l) * D;
                if (!inplace) { ::memcpy(slice_out, slice_in, D * sizeof(float)); }
                fwht(slice_out, D);
            }
        }
    }
}

// Out-of-place 3D FWHT
// input: [B,L,D] in row-major ([B,D,L] if transpose=true)
// output: [B,L,D] in row-major
// transpose: if true, input shape is [B,D,L] (swap L and D)
void fwht_tensor(float* input, float* output, 
                 int B, int L, int D, float scale, 
                 bool transpose) {
    if (transpose) {
        // [B,D,L] -> [B,L,D]
        for (int b = 0; b < B; ++b) {
            const float* src_base = input + b * D * L;
            float* dst_base = output + b * L * D;
            int l = 0;
            for (; l + 8 <= L; l += 8) {
                int d = 0;
#if defined(__AVX2__)
                for (; d + 8 <= D; d += 8) {
                    transpose4x4_scale_store_sse(
                        src_base + (d + 0) * L + l,
                        L,
                        dst_base + (l + 0) * D + d,
                        D,
                        scale);
                    transpose4x4_scale_store_sse(
                        src_base + (d + 4) * L + l,
                        L,
                        dst_base + (l + 0) * D + d + 4,
                        D,
                        scale);
                    transpose4x4_scale_store_sse(
                        src_base + (d + 0) * L + l + 4,
                        L,
                        dst_base + (l + 4) * D + d,
                        D,
                        scale);
                    transpose4x4_scale_store_sse(
                        src_base + (d + 4) * L + l + 4,
                        L,
                        dst_base + (l + 4) * D + d + 4,
                        D,
                        scale);
                }
#elif defined(__NEON__)
                for (; d + 8 <= D; d += 8) {
                    transpose4x4_scale_store_neon(
                        src_base + (d + 0) * L + l,
                        L,
                        dst_base + (l + 0) * D + d,
                        D,
                        scale);
                    transpose4x4_scale_store_neon(
                        src_base + (d + 4) * L + l,
                        L,
                        dst_base + (l + 0) * D + d + 4,
                        D,
                        scale);
                    transpose4x4_scale_store_neon(
                        src_base + (d + 0) * L + l + 4,
                        L,
                        dst_base + (l + 4) * D + d,
                        D,
                        scale);
                    transpose4x4_scale_store_neon(
                        src_base + (d + 4) * L + l + 4,
                        L,
                        dst_base + (l + 4) * D + d + 4,
                        D,
                        scale);
                }
#endif
                for (; d < D; ++d) {
                    dst_base[(l + 0) * D + d] = src_base[d * L + l + 0] * scale;
                    dst_base[(l + 1) * D + d] = src_base[d * L + l + 1] * scale;
                    dst_base[(l + 2) * D + d] = src_base[d * L + l + 2] * scale;
                    dst_base[(l + 3) * D + d] = src_base[d * L + l + 3] * scale;
                    dst_base[(l + 4) * D + d] = src_base[d * L + l + 4] * scale;
                    dst_base[(l + 5) * D + d] = src_base[d * L + l + 5] * scale;
                    dst_base[(l + 6) * D + d] = src_base[d * L + l + 6] * scale;
                    dst_base[(l + 7) * D + d] = src_base[d * L + l + 7] * scale;
                }
            }
            for (; l < L; ++l) {
                const float* src = src_base + l;
                float* dst = dst_base + l * D;
                int d = 0;
#if defined(__AVX2__)
                __m256 scale_vec = _mm256_set1_ps(scale);
                __m256i step = _mm256_set1_epi32(L);
                __m256i offsets =
                    _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), step);
                for (; d + 8 <= D; d += 8) {
                    __m256 gathered =
                        _mm256_i32gather_ps(src + d * L, offsets, 4);
                    _mm256_storeu_ps(dst + d, _mm256_mul_ps(gathered, scale_vec));
                }
#elif defined(__NEON__)
                const float32x4_t scale_vec = vdupq_n_f32(scale);
                for (; d + 8 <= D; d += 8) {
                    float32x4_t in0 = {src[(d + 0) * L], src[(d + 1) * L],
                                       src[(d + 2) * L], src[(d + 3) * L]};
                    float32x4_t in1 = {src[(d + 4) * L], src[(d + 5) * L],
                                       src[(d + 6) * L], src[(d + 7) * L]};
                    vst1q_f32(dst + d, vmulq_f32(in0, scale_vec));
                    vst1q_f32(dst + d + 4, vmulq_f32(in1, scale_vec));
                }
#endif
                for (; d < D; ++d) {
                    dst[d] = src[d * L] * scale;
                }
            }
            for (int l_row = 0; l_row < L; ++l_row) {
                fwht(dst_base + l_row * D, D);
            }
        }
    }
    else {
        // normal case: [B,L,D] -> [B,L,D]
        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                float* slice_in = input + (b * L + l) * D;
                float* slice_out = output + (b * L + l) * D;
                int d = 0;
#if defined(__AVX2__)
                __m256 scale_vec = _mm256_set1_ps(scale);
                for (; d + 8 <= D; d += 8) {
                    __m256 in = _mm256_loadu_ps(slice_in + d);
                    _mm256_storeu_ps(slice_out + d, _mm256_mul_ps(in, scale_vec));
                }
#elif defined(__NEON__)
                float32x4_t scale_vec0 = vdupq_n_f32(scale);
                float32x4_t scale_vec1 = vdupq_n_f32(scale);
                for (; d + 8 <= D; d += 8) {
                    float32x4_t in0 = vld1q_f32(slice_in + d);
                    float32x4_t in1 = vld1q_f32(slice_in + d + 4);
                    vst1q_f32(slice_out + d, vmulq_f32(in0, scale_vec0));
                    vst1q_f32(slice_out + d + 4, vmulq_f32(in1, scale_vec1));
                }
#endif
                for (; d < D; ++d) {
                    slice_out[d] = slice_in[d] * scale;
                }
                fwht(slice_out, D);
            }
        }
    }
}




}
