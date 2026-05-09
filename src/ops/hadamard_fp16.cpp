#include <cstddef>
#include <cstring>

#include "ops/ops.h"

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#endif

namespace LMStore {

#if defined(LMSTORE_HAS_FP16)
namespace {

void fwht_fp16(fp16_t* tensor, int n) {
  if (n >= 2) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    int i = 0;
    for (; i + 8 <= n; i += 8) {
      float16x8x2_t x = vld2q_f16(tensor + i);
      float16x8_t sum = vaddq_f16(x.val[0], x.val[1]);
      float16x8_t diff = vsubq_f16(x.val[0], x.val[1]);
      float16x8x2_t y;
      y.val[0] = sum;
      y.val[1] = diff;
      vst2q_f16(tensor + i, y);
    }
    for (; i < n; i += 2) {
      fp16_t u = tensor[i];
      fp16_t v = tensor[i + 1];
      tensor[i] = u + v;
      tensor[i + 1] = u - v;
    }
#else
    for (int i = 0; i < n; i += 2) {
      fp16_t u = tensor[i];
      fp16_t v = tensor[i + 1];
      tensor[i] = u + v;
      tensor[i + 1] = u - v;
    }
#endif
  }

  if (n >= 4) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    int i = 0;
    for (; i + 8 <= n; i += 8) {
      float16x8_t x = vld1q_f16(tensor + i);
      float16x4_t lo = vget_low_f16(x);
      float16x4_t hi = vget_high_f16(x);
      vst1_f16(tensor + i, vadd_f16(lo, hi));
      vst1_f16(tensor + i + 4, vsub_f16(lo, hi));
    }
    for (; i < n; i += 4) {
      fp16_t u0 = tensor[i];
      fp16_t u1 = tensor[i + 1];
      fp16_t v0 = tensor[i + 2];
      fp16_t v1 = tensor[i + 3];
      tensor[i] = u0 + v0;
      tensor[i + 1] = u1 + v1;
      tensor[i + 2] = u0 - v0;
      tensor[i + 3] = u1 - v1;
    }
#else
    for (int i = 0; i < n; i += 4) {
      fp16_t u0 = tensor[i];
      fp16_t u1 = tensor[i + 1];
      fp16_t v0 = tensor[i + 2];
      fp16_t v1 = tensor[i + 3];
      tensor[i] = u0 + v0;
      tensor[i + 1] = u1 + v1;
      tensor[i + 2] = u0 - v0;
      tensor[i + 3] = u1 - v1;
    }
#endif
  }

  if (n >= 8) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    int i = 0;
    for (; i + 8 <= n; i += 8) {
      float16x4_t a = vld1_f16(tensor + i);
      float16x4_t b = vld1_f16(tensor + i + 4);
      vst1_f16(tensor + i, vadd_f16(a, b));
      vst1_f16(tensor + i + 4, vsub_f16(a, b));
    }
    for (; i < n; i += 8) {
      for (int j = 0; j < 4; ++j) {
        fp16_t u = tensor[i + j];
        fp16_t v = tensor[i + j + 4];
        tensor[i + j] = u + v;
        tensor[i + j + 4] = u - v;
      }
    }
#else
    for (int i = 0; i < n; i += 8) {
      for (int j = 0; j < 4; ++j) {
        fp16_t u = tensor[i + j];
        fp16_t v = tensor[i + j + 4];
        tensor[i + j] = u + v;
        tensor[i + j + 4] = u - v;
      }
    }
#endif
  }

  for (int len = 8; len < n; len *= 2) {
    for (int i = 0; i < n; i += 2 * len) {
      int j = 0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
      for (; j + 8 <= len; j += 8) {
        float16x8_t u = vld1q_f16(tensor + i + j);
        float16x8_t v = vld1q_f16(tensor + i + j + len);
        vst1q_f16(tensor + i + j, vaddq_f16(u, v));
        vst1q_f16(tensor + i + j + len, vsubq_f16(u, v));
      }
#endif
      for (; j < len; ++j) {
        fp16_t u = tensor[i + j];
        fp16_t v = tensor[i + j + len];
        tensor[i + j] = u + v;
        tensor[i + j + len] = u - v;
      }
    }
  }
}

}  // namespace

void fwht_tensor(fp16_t* input, fp16_t* output, int B, int L, int D, bool transpose) {
  if (transpose) {
    transpose_tensor(input, output, B, D, L);
    for (int b = 0; b < B; ++b) {
      for (int l = 0; l < L; ++l) {
        fwht_fp16(output + (b * L + l) * D, D);
      }
    }
  } else {
    const bool inplace = (input == output);
    for (int b = 0; b < B; ++b) {
      for (int l = 0; l < L; ++l) {
        fp16_t* slice_in = input + (b * L + l) * D;
        fp16_t* slice_out = output + (b * L + l) * D;
        if (!inplace) {
          ::memcpy(slice_out, slice_in, D * sizeof(fp16_t));
        }
        fwht_fp16(slice_out, D);
      }
    }
  }
}

void fwht_tensor(fp16_t* input, fp16_t* output, int B, int L, int D, fp16_t scale,
                 bool transpose) {
  if (transpose) {
    transpose_tensor(input, output, B, D, L);
    for (int b = 0; b < B; ++b) {
      for (int l = 0; l < L; ++l) {
        fp16_t* dst = output + (b * L + l) * D;
        int d = 0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        const float16x8_t scale_vec = vdupq_n_f16(scale);
        for (; d + 8 <= D; d += 8) {
          float16x8_t in = vld1q_f16(dst + d);
          vst1q_f16(dst + d, vmulq_f16(in, scale_vec));
        }
#endif
        for (; d < D; ++d) {
          dst[d] = dst[d] * scale;
        }
        fwht_fp16(dst, D);
      }
    }
  } else {
    for (int b = 0; b < B; ++b) {
      for (int l = 0; l < L; ++l) {
        fp16_t* slice_in = input + (b * L + l) * D;
        fp16_t* slice_out = output + (b * L + l) * D;
        int d = 0;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        const float16x8_t scale_vec = vdupq_n_f16(scale);
        for (; d + 8 <= D; d += 8) {
          float16x8_t in = vld1q_f16(slice_in + d);
          vst1q_f16(slice_out + d, vmulq_f16(in, scale_vec));
        }
#endif
        for (; d < D; ++d) {
          slice_out[d] = slice_in[d] * scale;
        }
        fwht_fp16(slice_out, D);
      }
    }
  }
}

#endif

}  // namespace LMStore
