/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace LMStore {

enum class ChunkPromptType : int32_t {
  kPrefix = 0,
  kNonPrefix = 1,
  kNew = 2,
};

enum class ChunkGraphType : int32_t {
  kSelectiveRecompute = 0,
  kPrompt = 1,
  kPrefixReuse = 2,
};

struct ChunkMergeSpan {
  int32_t start;
  int32_t end;
  ChunkPromptType type;
};

struct ChunkMergeInterval {
  int32_t start;
  int32_t end;
};

struct ChunkMergeResult {
  std::vector<ChunkMergeInterval> chunks;
  std::vector<ChunkGraphType> groups;
  double min_time{0.0};
};

ChunkMergeResult chunk_merge(
    const std::vector<ChunkMergeSpan>& spans,
    int32_t selective_recompute_budget,
    int32_t prompt_budget,
    double selective_recompute_time,
    double prompt_time,
    double reuse_ratio);

ChunkMergeResult chunk_merge(
    const std::vector<ChunkMergeSpan>& spans,
    int32_t selective_recompute_budget,
    int32_t prompt_budget,
    double d,
    double reuse_ratio);

} // namespace LMStore
