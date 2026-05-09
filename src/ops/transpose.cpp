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
inline void transpose4x4_store_sse(
    const float* src,
    int src_stride,
    float* dst,
    int dst_stride) {
    __m128 row0 = _mm_loadu_ps(src + 0 * src_stride);
    __m128 row1 = _mm_loadu_ps(src + 1 * src_stride);
    __m128 row2 = _mm_loadu_ps(src + 2 * src_stride);
    __m128 row3 = _mm_loadu_ps(src + 3 * src_stride);
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
    _mm_storeu_ps(dst + 0 * dst_stride, row0);
    _mm_storeu_ps(dst + 1 * dst_stride, row1);
    _mm_storeu_ps(dst + 2 * dst_stride, row2);
    _mm_storeu_ps(dst + 3 * dst_stride, row3);
}
#elif defined(__NEON__)
inline void transpose4x4_store_neon(
    const float* src,
    int src_stride,
    float* dst,
    int dst_stride) {
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

    vst1q_f32(dst + 0 * dst_stride, vcombine_f32(a0, a1));
    vst1q_f32(dst + 1 * dst_stride, vcombine_f32(a2, a3));
    vst1q_f32(dst + 2 * dst_stride, vcombine_f32(b0, b1));
    vst1q_f32(dst + 3 * dst_stride, vcombine_f32(b2, b3));
}
#if defined(LMSTORE_HAS_FP16) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline void transpose4x4_store_fp16_neon(
    const fp16_t* src,
    int src_stride,
    fp16_t* dst,
    int dst_stride) {
    float16x4_t row0 = vld1_f16(src + 0 * src_stride);
    float16x4_t row1 = vld1_f16(src + 1 * src_stride);
    float16x4_t row2 = vld1_f16(src + 2 * src_stride);
    float16x4_t row3 = vld1_f16(src + 3 * src_stride);
    fp16_t r0[4];
    fp16_t r1[4];
    fp16_t r2[4];
    fp16_t r3[4];
    vst1_f16(r0, row0);
    vst1_f16(r1, row1);
    vst1_f16(r2, row2);
    vst1_f16(r3, row3);
    dst[0 * dst_stride + 0] = r0[0];
    dst[0 * dst_stride + 1] = r1[0];
    dst[0 * dst_stride + 2] = r2[0];
    dst[0 * dst_stride + 3] = r3[0];
    dst[1 * dst_stride + 0] = r0[1];
    dst[1 * dst_stride + 1] = r1[1];
    dst[1 * dst_stride + 2] = r2[1];
    dst[1 * dst_stride + 3] = r3[1];
    dst[2 * dst_stride + 0] = r0[2];
    dst[2 * dst_stride + 1] = r1[2];
    dst[2 * dst_stride + 2] = r2[2];
    dst[2 * dst_stride + 3] = r3[2];
    dst[3 * dst_stride + 0] = r0[3];
    dst[3 * dst_stride + 1] = r1[3];
    dst[3 * dst_stride + 2] = r2[3];
    dst[3 * dst_stride + 3] = r3[3];
}
#endif
#endif

} // namespace

void transpose_tensor(float* input, float* output, 
                      int dim0, int dim1, int dim2) {
    // dims are according to input shape: [dim0, dim1, dim2]
    for (int d0 = 0; d0 < dim0; ++d0) {
        const float* src_base = input + d0 * dim1 * dim2;
        float* dst_base = output + d0 * dim2 * dim1;
        int d1 = 0;
        for (; d1 + 8 <= dim1; d1 += 8) {
            int d2 = 0;
#if defined(__AVX2__)
            for (; d2 + 8 <= dim2; d2 += 8) {
                transpose4x4_store_sse(
                    src_base + (d1 + 0) * dim2 + d2,
                    dim2,
                    dst_base + (d2 + 0) * dim1 + d1,
                    dim1);
                transpose4x4_store_sse(
                    src_base + (d1 + 4) * dim2 + d2,
                    dim2,
                    dst_base + (d2 + 0) * dim1 + d1 + 4,
                    dim1);
                transpose4x4_store_sse(
                    src_base + (d1 + 0) * dim2 + d2 + 4,
                    dim2,
                    dst_base + (d2 + 4) * dim1 + d1,
                    dim1);
                transpose4x4_store_sse(
                    src_base + (d1 + 4) * dim2 + d2 + 4,
                    dim2,
                    dst_base + (d2 + 4) * dim1 + d1 + 4,
                    dim1);
            }
#elif defined(__NEON__)
            for (; d2 + 8 <= dim2; d2 += 8) {
                transpose4x4_store_neon(
                    src_base + (d1 + 0) * dim2 + d2,
                    dim2,
                    dst_base + (d2 + 0) * dim1 + d1,
                    dim1);
                transpose4x4_store_neon(
                    src_base + (d1 + 4) * dim2 + d2,
                    dim2,
                    dst_base + (d2 + 0) * dim1 + d1 + 4,
                    dim1);
                transpose4x4_store_neon(
                    src_base + (d1 + 0) * dim2 + d2 + 4,
                    dim2,
                    dst_base + (d2 + 4) * dim1 + d1,
                    dim1);
                transpose4x4_store_neon(
                    src_base + (d1 + 4) * dim2 + d2 + 4,
                    dim2,
                    dst_base + (d2 + 4) * dim1 + d1 + 4,
                    dim1);
            }
#endif
            for (; d2 + 4 <= dim2; d2 += 4) {
#if defined(__AVX2__)
                transpose4x4_store_sse(
                    src_base + d1 * dim2 + d2,
                    dim2,
                    dst_base + d2 * dim1 + d1,
                    dim1);
                transpose4x4_store_sse(
                    src_base + (d1 + 4) * dim2 + d2,
                    dim2,
                    dst_base + d2 * dim1 + d1 + 4,
                    dim1);
#elif defined(__NEON__)
                transpose4x4_store_neon(
                    src_base + d1 * dim2 + d2,
                    dim2,
                    dst_base + d2 * dim1 + d1,
                    dim1);
                transpose4x4_store_neon(
                    src_base + (d1 + 4) * dim2 + d2,
                    dim2,
                    dst_base + d2 * dim1 + d1 + 4,
                    dim1);
#endif
            }
            for (; d2 < dim2; ++d2) {
                dst_base[d2 * dim1 + d1 + 0] = src_base[(d1 + 0) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 1] = src_base[(d1 + 1) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 2] = src_base[(d1 + 2) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 3] = src_base[(d1 + 3) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 4] = src_base[(d1 + 4) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 5] = src_base[(d1 + 5) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 6] = src_base[(d1 + 6) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 7] = src_base[(d1 + 7) * dim2 + d2];
            }
        }
        for (; d1 + 4 <= dim1; d1 += 4) {
            int d2 = 0;
#if defined(__AVX2__)
            for (; d2 + 4 <= dim2; d2 += 4) {
                transpose4x4_store_sse(
                    src_base + d1 * dim2 + d2,
                    dim2,
                    dst_base + d2 * dim1 + d1,
                    dim1);
            }
#elif defined(__NEON__)
            for (; d2 + 4 <= dim2; d2 += 4) {
                transpose4x4_store_neon(
                    src_base + d1 * dim2 + d2,
                    dim2,
                    dst_base + d2 * dim1 + d1,
                    dim1);
            }
#endif
            for (; d2 < dim2; ++d2) {
                dst_base[d2 * dim1 + d1 + 0] = src_base[(d1 + 0) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 1] = src_base[(d1 + 1) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 2] = src_base[(d1 + 2) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 3] = src_base[(d1 + 3) * dim2 + d2];
            }
        }
        for (; d1 < dim1; ++d1) {
            for (int d2 = 0; d2 < dim2; ++d2) {
                dst_base[d2 * dim1 + d1] = src_base[d1 * dim2 + d2];
            }
        }
    }
}

void transpose_tensor(uint8_t* input, uint8_t* output,
                      int dim0, int dim1, int dim2) {
    for (int d0 = 0; d0 < dim0; ++d0) {
        const uint8_t* src_base = input + d0 * dim1 * dim2;
        uint8_t* dst_base = output + d0 * dim2 * dim1;
        int d1 = 0;
        for (; d1 + 32 <= dim1; d1 += 32) {
            int d2 = 0;
            for (; d2 + 32 <= dim2; d2 += 32) {
                for (int bi = 0; bi < 32; bi += 4) {
                    for (int bj = 0; bj < 32; bj += 4) {
                        const uint8_t* block_src =
                            src_base + (d1 + bi) * dim2 + (d2 + bj);
                        uint8_t* block_dst =
                            dst_base + (d2 + bj) * dim1 + (d1 + bi);
                        block_dst[0 * dim1 + 0] = block_src[0 * dim2 + 0];
                        block_dst[0 * dim1 + 1] = block_src[1 * dim2 + 0];
                        block_dst[0 * dim1 + 2] = block_src[2 * dim2 + 0];
                        block_dst[0 * dim1 + 3] = block_src[3 * dim2 + 0];
                        block_dst[1 * dim1 + 0] = block_src[0 * dim2 + 1];
                        block_dst[1 * dim1 + 1] = block_src[1 * dim2 + 1];
                        block_dst[1 * dim1 + 2] = block_src[2 * dim2 + 1];
                        block_dst[1 * dim1 + 3] = block_src[3 * dim2 + 1];
                        block_dst[2 * dim1 + 0] = block_src[0 * dim2 + 2];
                        block_dst[2 * dim1 + 1] = block_src[1 * dim2 + 2];
                        block_dst[2 * dim1 + 2] = block_src[2 * dim2 + 2];
                        block_dst[2 * dim1 + 3] = block_src[3 * dim2 + 2];
                        block_dst[3 * dim1 + 0] = block_src[0 * dim2 + 3];
                        block_dst[3 * dim1 + 1] = block_src[1 * dim2 + 3];
                        block_dst[3 * dim1 + 2] = block_src[2 * dim2 + 3];
                        block_dst[3 * dim1 + 3] = block_src[3 * dim2 + 3];
                    }
                }
            }
            for (; d2 + 4 <= dim2; d2 += 4) {
                for (int bi = 0; bi < 32; bi += 4) {
                    const uint8_t* block_src =
                        src_base + (d1 + bi) * dim2 + d2;
                    uint8_t* block_dst =
                        dst_base + d2 * dim1 + (d1 + bi);
                    block_dst[0 * dim1 + 0] = block_src[0 * dim2 + 0];
                    block_dst[0 * dim1 + 1] = block_src[1 * dim2 + 0];
                    block_dst[0 * dim1 + 2] = block_src[2 * dim2 + 0];
                    block_dst[0 * dim1 + 3] = block_src[3 * dim2 + 0];
                    block_dst[1 * dim1 + 0] = block_src[0 * dim2 + 1];
                    block_dst[1 * dim1 + 1] = block_src[1 * dim2 + 1];
                    block_dst[1 * dim1 + 2] = block_src[2 * dim2 + 1];
                    block_dst[1 * dim1 + 3] = block_src[3 * dim2 + 1];
                    block_dst[2 * dim1 + 0] = block_src[0 * dim2 + 2];
                    block_dst[2 * dim1 + 1] = block_src[1 * dim2 + 2];
                    block_dst[2 * dim1 + 2] = block_src[2 * dim2 + 2];
                    block_dst[2 * dim1 + 3] = block_src[3 * dim2 + 2];
                    block_dst[3 * dim1 + 0] = block_src[0 * dim2 + 3];
                    block_dst[3 * dim1 + 1] = block_src[1 * dim2 + 3];
                    block_dst[3 * dim1 + 2] = block_src[2 * dim2 + 3];
                    block_dst[3 * dim1 + 3] = block_src[3 * dim2 + 3];
                }
            }
            for (; d2 < dim2; ++d2) {
                for (int bi = 0; bi < 32; ++bi) {
                    dst_base[d2 * dim1 + d1 + bi] =
                        src_base[(d1 + bi) * dim2 + d2];
                }
            }
        }
        for (; d1 + 4 <= dim1; d1 += 4) {
            int d2 = 0;
            for (; d2 + 4 <= dim2; d2 += 4) {
                const uint8_t* block_src = src_base + d1 * dim2 + d2;
                uint8_t* block_dst = dst_base + d2 * dim1 + d1;
                block_dst[0 * dim1 + 0] = block_src[0 * dim2 + 0];
                block_dst[0 * dim1 + 1] = block_src[1 * dim2 + 0];
                block_dst[0 * dim1 + 2] = block_src[2 * dim2 + 0];
                block_dst[0 * dim1 + 3] = block_src[3 * dim2 + 0];
                block_dst[1 * dim1 + 0] = block_src[0 * dim2 + 1];
                block_dst[1 * dim1 + 1] = block_src[1 * dim2 + 1];
                block_dst[1 * dim1 + 2] = block_src[2 * dim2 + 1];
                block_dst[1 * dim1 + 3] = block_src[3 * dim2 + 1];
                block_dst[2 * dim1 + 0] = block_src[0 * dim2 + 2];
                block_dst[2 * dim1 + 1] = block_src[1 * dim2 + 2];
                block_dst[2 * dim1 + 2] = block_src[2 * dim2 + 2];
                block_dst[2 * dim1 + 3] = block_src[3 * dim2 + 2];
                block_dst[3 * dim1 + 0] = block_src[0 * dim2 + 3];
                block_dst[3 * dim1 + 1] = block_src[1 * dim2 + 3];
                block_dst[3 * dim1 + 2] = block_src[2 * dim2 + 3];
                block_dst[3 * dim1 + 3] = block_src[3 * dim2 + 3];
            }
            for (; d2 < dim2; ++d2) {
                dst_base[d2 * dim1 + d1 + 0] = src_base[(d1 + 0) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 1] = src_base[(d1 + 1) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 2] = src_base[(d1 + 2) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 3] = src_base[(d1 + 3) * dim2 + d2];
            }
        }
        for (; d1 < dim1; ++d1) {
            for (int d2 = 0; d2 < dim2; ++d2) {
                dst_base[d2 * dim1 + d1] = src_base[d1 * dim2 + d2];
            }
        }
    }
}

#if defined(LMSTORE_HAS_FP16)
void transpose_tensor(fp16_t* input, fp16_t* output,
                      int dim0, int dim1, int dim2) {
    for (int d0 = 0; d0 < dim0; ++d0) {
        const fp16_t* src_base = input + d0 * dim1 * dim2;
        fp16_t* dst_base = output + d0 * dim2 * dim1;
        int d1 = 0;
        for (; d1 + 8 <= dim1; d1 += 8) {
            int d2 = 0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            for (; d2 + 8 <= dim2; d2 += 8) {
                transpose4x4_store_fp16_neon(
                    src_base + (d1 + 0) * dim2 + d2,
                    dim2,
                    dst_base + (d2 + 0) * dim1 + d1,
                    dim1);
                transpose4x4_store_fp16_neon(
                    src_base + (d1 + 4) * dim2 + d2,
                    dim2,
                    dst_base + (d2 + 0) * dim1 + d1 + 4,
                    dim1);
                transpose4x4_store_fp16_neon(
                    src_base + (d1 + 0) * dim2 + d2 + 4,
                    dim2,
                    dst_base + (d2 + 4) * dim1 + d1,
                    dim1);
                transpose4x4_store_fp16_neon(
                    src_base + (d1 + 4) * dim2 + d2 + 4,
                    dim2,
                    dst_base + (d2 + 4) * dim1 + d1 + 4,
                    dim1);
            }
            for (; d2 + 4 <= dim2; d2 += 4) {
                transpose4x4_store_fp16_neon(
                    src_base + d1 * dim2 + d2,
                    dim2,
                    dst_base + d2 * dim1 + d1,
                    dim1);
                transpose4x4_store_fp16_neon(
                    src_base + (d1 + 4) * dim2 + d2,
                    dim2,
                    dst_base + d2 * dim1 + d1 + 4,
                    dim1);
            }
#endif
            for (; d2 < dim2; ++d2) {
                dst_base[d2 * dim1 + d1 + 0] = src_base[(d1 + 0) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 1] = src_base[(d1 + 1) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 2] = src_base[(d1 + 2) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 3] = src_base[(d1 + 3) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 4] = src_base[(d1 + 4) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 5] = src_base[(d1 + 5) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 6] = src_base[(d1 + 6) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 7] = src_base[(d1 + 7) * dim2 + d2];
            }
        }
        for (; d1 + 4 <= dim1; d1 += 4) {
            int d2 = 0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            for (; d2 + 4 <= dim2; d2 += 4) {
                transpose4x4_store_fp16_neon(
                    src_base + d1 * dim2 + d2,
                    dim2,
                    dst_base + d2 * dim1 + d1,
                    dim1);
            }
#endif
            for (; d2 < dim2; ++d2) {
                dst_base[d2 * dim1 + d1 + 0] = src_base[(d1 + 0) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 1] = src_base[(d1 + 1) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 2] = src_base[(d1 + 2) * dim2 + d2];
                dst_base[d2 * dim1 + d1 + 3] = src_base[(d1 + 3) * dim2 + d2];
            }
        }
        for (; d1 < dim1; ++d1) {
            for (int d2 = 0; d2 < dim2; ++d2) {
                dst_base[d2 * dim1 + d1] = src_base[d1 * dim2 + d2];
            }
        }
    }
}
#endif

}
