#include <cstddef>
#include <cstring>
#include "ops/ops.h"
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__NEON__)
#include <arm_neon.h>
#endif

namespace LMStore {

void fwht(float* tensor, int n) {
    // input can equal to output for in-place FWHT
    for (int len = 1; len < n; len *= 2) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
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
            for (int l = 0; l < L; ++l) {
                for (int d = 0; d < D; ++d) {
                    output[b * L * D + l * D + d] = input[b * D * L + d * L + l] * scale;
                }
                float* slice_out = output + b * L * D + l * D;
                fwht(slice_out, D);
            }
        }
    }
    else {
        // normal case: [B,L,D] -> [B,L,D]
        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                float* slice_in = input + (b * L + l) * D;
                float* slice_out = output + (b * L + l) * D;
                for (int d = 0; d < D; ++d) {
                    slice_out[d] = slice_in[d] * scale;
                }
                fwht(slice_out, D);
            }
        }
    }
}




}