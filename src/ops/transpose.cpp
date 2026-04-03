#include <cstddef>
#include <cstring>
#include "ops/ops.h"
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__NEON__)
#include <arm_neon.h>
#endif

namespace LMStore {

void transpose_tensor(float* input, float* output, 
                      int dim0, int dim1, int dim2) {
    // dims are according to input shape: [dim0, dim1, dim2]
    for (int d0 = 0; d0 < dim0; ++d0) {
        for (int d1 = 0; d1 < dim1; ++d1) {
            for (int d2 = 0; d2 < dim2; ++d2) {
                output[d0 * dim2 * dim1 + d2 * dim1 + d1] = input[d0 * dim1 * dim2 + d1 * dim2 + d2];
            }
        }
    }
}

}