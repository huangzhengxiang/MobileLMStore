/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chunk_merge.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace LMStore {
namespace {

struct ConsumeResult {
  int32_t prev_end;
};

struct Decision {
  int32_t prev_end{-1};
  ChunkGraphType group{ChunkGraphType::kPrompt};
  bool valid{false};
};

bool is_base_state(
    int32_t span_index,
    const std::vector<ChunkMergeSpan>& spans) {
  return span_index < 0 ||
      (span_index == 0 && spans[0].type == ChunkPromptType::kPrefix);
}

void validate_spans(const std::vector<ChunkMergeSpan>& spans) {
  if (spans.empty()) {
    return;
  }
  for (size_t i = 0; i < spans.size(); ++i) {
    const auto& span = spans[i];
    if (span.start < 0 || span.end < span.start) {
      throw std::invalid_argument("chunk_merge spans must have 0 <= start <= end.");
    }
    if (i > 0 && span.start != spans[i - 1].end + 1) {
      throw std::invalid_argument(
          "chunk_merge spans must be contiguous and sorted.");
    }
  }
}

std::vector<int32_t> build_owner_index(
    const std::vector<ChunkMergeSpan>& spans) {
  if (spans.empty()) {
    return {};
  }
  const int32_t length = spans.back().end + 1;
  std::vector<int32_t> owner(length, -1);
  for (size_t i = 0; i < spans.size(); ++i) {
    for (int32_t pos = spans[i].start; pos <= spans[i].end; ++pos) {
      owner[pos] = static_cast<int32_t>(i);
    }
  }
  return owner;
}

ConsumeResult consume_backward(
    int32_t end_pos,
    int32_t budget,
    double reuse_ratio,
    bool selective_recompute_mode,
    const std::vector<ChunkMergeSpan>& spans,
    const std::vector<int32_t>& owner) {
  int32_t span_index = owner[end_pos];
  int32_t cur = end_pos;
  int32_t consumed = 0;

  while (budget > 0 && !is_base_state(span_index, spans)) {
    const auto& span = spans[span_index];
    const int32_t available = cur - span.start + 1;
    if (available <= 0) {
      --span_index;
      continue;
    }

    int32_t delta = 0;
    if (selective_recompute_mode && span.type == ChunkPromptType::kNew) {
      const int32_t capped =
          std::max<int32_t>(0, static_cast<int32_t>(std::floor(budget * reuse_ratio)));
      delta = std::min(capped, available);
      if (delta <= 0) {
        break;
      }
      budget -= static_cast<int32_t>(std::ceil(delta / reuse_ratio));
    } else {
      delta = std::min(budget, available);
      if (delta <= 0) {
        break;
      }
      budget -= delta;
    }

    consumed += delta;
    cur -= delta;
    if (cur < span.start) {
      --span_index;
    }
  }

  return ConsumeResult{end_pos - consumed};
}

double dp_value_at(const std::vector<double>& dp, int32_t prev_end) {
  return prev_end < 0 ? 0.0 : dp[prev_end];
}

} // namespace

ChunkMergeResult chunk_merge(
    const std::vector<ChunkMergeSpan>& spans,
    int32_t selective_recompute_budget,
    int32_t prompt_budget,
    double selective_recompute_time,
    double prompt_time,
    double reuse_ratio) {
  validate_spans(spans);
  if (spans.empty()) {
    return {};
  }
  if (selective_recompute_budget < 0 || prompt_budget < 0) {
    throw std::invalid_argument("chunk_merge budgets must be non-negative.");
  }
  if (reuse_ratio <= 0.0) {
    throw std::invalid_argument("chunk_merge reuse_ratio must be positive.");
  }

  const int32_t length = spans.back().end + 1;
  const std::vector<int32_t> owner = build_owner_index(spans);
  const double inf = std::numeric_limits<double>::infinity();

  std::vector<double> dp(length, inf);
  std::vector<Decision> decision(length);

  for (int32_t end_pos = 0; end_pos < length; ++end_pos) {
    const int32_t span_index = owner[end_pos];
    if (is_base_state(span_index, spans)) {
      dp[end_pos] = 0.0;
      continue;
    }

    const ConsumeResult selective_recompute = consume_backward(
        end_pos,
        selective_recompute_budget,
        reuse_ratio,
        true,
        spans,
        owner);
    const ConsumeResult prompt = consume_backward(
        end_pos,
        prompt_budget,
        reuse_ratio,
        false,
        spans,
        owner);

    const double selective_recompute_total =
        dp_value_at(dp, selective_recompute.prev_end) + selective_recompute_time;
    const double prompt_total = dp_value_at(dp, prompt.prev_end) + prompt_time;

    if (selective_recompute_total < prompt_total) {
      dp[end_pos] = selective_recompute_total;
      decision[end_pos].prev_end = selective_recompute.prev_end;
      decision[end_pos].group = ChunkGraphType::kSelectiveRecompute;
      decision[end_pos].valid = true;
    } else {
      dp[end_pos] = prompt_total;
      decision[end_pos].prev_end = prompt.prev_end;
      decision[end_pos].group = ChunkGraphType::kPrompt;
      decision[end_pos].valid = true;
    }
  }

  ChunkMergeResult result;
  result.min_time = dp.back();

  for (int32_t end_pos = length - 1; end_pos >= 0;) {
    if (!decision[end_pos].valid) {
      break;
    }
    const Decision& step = decision[end_pos];
    result.chunks.push_back({step.prev_end + 1, end_pos});
    result.groups.push_back(step.group);
    end_pos = step.prev_end;
  }

  std::reverse(result.chunks.begin(), result.chunks.end());
  std::reverse(result.groups.begin(), result.groups.end());
  return result;
}

ChunkMergeResult chunk_merge(
    const std::vector<ChunkMergeSpan>& spans,
    int32_t selective_recompute_budget,
    int32_t prompt_budget,
    double d,
    double reuse_ratio) {
  return chunk_merge(
      spans,
      selective_recompute_budget,
      prompt_budget,
      1.0 + d,
      1.0,
      reuse_ratio);
}

} // namespace LMStore
