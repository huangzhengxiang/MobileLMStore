#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "ops/ops.h"

namespace {

void fwht_scalar(float* tensor, int n) {
  for (int len = 1; len < n; len *= 2) {
    for (int i = 0; i < n; i += 2 * len) {
      for (int j = 0; j < len; ++j) {
        const float u = tensor[i + j];
        const float v = tensor[i + j + len];
        tensor[i + j] = u + v;
        tensor[i + j + len] = u - v;
      }
    }
  }
}

void fwht_tensor_scalar(
    const float* input,
    float* output,
    int B,
    int L,
    int D,
    bool transpose) {
  if (transpose) {
    for (int b = 0; b < B; ++b) {
      for (int l = 0; l < L; ++l) {
        for (int d = 0; d < D; ++d) {
          output[b * D * L + l * D + d] = input[b * L * D + d * L + l];
        }
        fwht_scalar(output + b * D * L + l * D, D);
      }
    }
  } else {
    const bool inplace = input == output;
    for (int b = 0; b < B; ++b) {
      for (int l = 0; l < L; ++l) {
        const float* slice_in = input + (b * L + l) * D;
        float* slice_out = output + (b * L + l) * D;
        if (!inplace) {
          std::memcpy(slice_out, slice_in, static_cast<size_t>(D) * sizeof(float));
        }
        fwht_scalar(slice_out, D);
      }
    }
  }
}

void fwht_tensor_scalar(
    const float* input,
    float* output,
    int B,
    int L,
    int D,
    float scale,
    bool transpose) {
  if (transpose) {
    for (int b = 0; b < B; ++b) {
      for (int l = 0; l < L; ++l) {
        for (int d = 0; d < D; ++d) {
          output[b * L * D + l * D + d] =
              input[b * D * L + d * L + l] * scale;
        }
        fwht_scalar(output + b * L * D + l * D, D);
      }
    }
  } else {
    for (int b = 0; b < B; ++b) {
      for (int l = 0; l < L; ++l) {
        const float* slice_in = input + (b * L + l) * D;
        float* slice_out = output + (b * L + l) * D;
        for (int d = 0; d < D; ++d) {
          slice_out[d] = slice_in[d] * scale;
        }
        fwht_scalar(slice_out, D);
      }
    }
  }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float diff = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    diff = std::max(diff, std::abs(a[i] - b[i]));
  }
  return diff;
}

bool run_case(int B, int L, int D, bool transpose, bool use_scale) {
  std::mt19937 rng(0x1234 + B * 97 + L * 13 + D * 3 + transpose * 5 + use_scale * 11);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const size_t elems = static_cast<size_t>(B) * L * D;
  std::vector<float> input(elems);
  for (float& v : input) {
    v = dist(rng);
  }

  std::vector<float> expected(elems, 0.0f);
  std::vector<float> actual(elems, 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  if (use_scale) {
    fwht_tensor_scalar(input.data(), expected.data(), B, L, D, scale, transpose);
    LMStore::fwht_tensor(input.data(), actual.data(), B, L, D, scale, transpose);
  } else {
    fwht_tensor_scalar(input.data(), expected.data(), B, L, D, transpose);
    LMStore::fwht_tensor(input.data(), actual.data(), B, L, D, transpose);
  }

  const float diff = max_abs_diff(expected, actual);
  if (diff > 1e-5f) {
    std::cerr << "Mismatch: B=" << B << " L=" << L << " D=" << D
              << " transpose=" << transpose
              << " use_scale=" << use_scale
              << " max_abs_diff=" << diff << "\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  const int dims[] = {8, 16, 32, 64, 128};
  for (int B : {1, 2}) {
    for (int L : {1, 3, 7}) {
      for (int D : dims) {
        for (bool transpose : {false, true}) {
          for (bool use_scale : {false, true}) {
            if (!run_case(B, L, D, transpose, use_scale)) {
              return 1;
            }
          }
        }
      }
    }
  }
  std::cout << "test_fwht_simd passed\n";
  return 0;
}
