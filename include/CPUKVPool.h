#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace LMStore {

template <typename T>
class CPUKVPool {
 public:
  struct Entry {
    int row_id{-1};
    int seq_len{0};
    size_t bytes{0};
    std::vector<std::vector<T>> k_layers;
    std::vector<std::vector<T>> v_layers;
    std::vector<T*> k_ptrs;
    std::vector<T*> v_ptrs;
  };

  CPUKVPool(
      int num_layers,
      int num_heads,
      int head_dim,
      size_t capacity_bytes = 512ull * 1024ull * 1024ull);

  std::pair<bool, Entry*> get_or_create(int row_id, int seq_len);

 private:
  struct Slot {
    std::unique_ptr<Entry> entry;
    std::list<int>::iterator lru_it;
  };

  size_t entry_bytes(int seq_len) const;
  void touch(int row_id);
  void evict_if_needed(size_t incoming_bytes);

  int num_layers_;
  int num_heads_;
  int head_dim_;
  size_t capacity_bytes_;
  size_t current_bytes_{0};
  std::list<int> lru_;
  std::unordered_map<int, Slot> entries_;
};

template <typename T>
CPUKVPool<T>::CPUKVPool(
    int num_layers,
    int num_heads,
    int head_dim,
    size_t capacity_bytes)
    : num_layers_(num_layers),
      num_heads_(num_heads),
      head_dim_(head_dim),
      capacity_bytes_(capacity_bytes) {}

template <typename T>
std::pair<bool, typename CPUKVPool<T>::Entry*> CPUKVPool<T>::get_or_create(
    int row_id,
    int seq_len) {
  auto it = entries_.find(row_id);
  if (it != entries_.end()) {
    touch(row_id);
    return {true, it->second.entry.get()};
  }

  auto entry = std::make_unique<Entry>();
  entry->row_id = row_id;
  entry->seq_len = seq_len;
  entry->bytes = entry_bytes(seq_len);
  entry->k_layers.resize(num_layers_);
  entry->v_layers.resize(num_layers_);
  entry->k_ptrs.resize(num_layers_);
  entry->v_ptrs.resize(num_layers_);

  const size_t k_elems =
      static_cast<size_t>(num_heads_) * head_dim_ * seq_len;
  const size_t v_elems =
      static_cast<size_t>(num_heads_) * seq_len * head_dim_;
  for (int layer = 0; layer < num_layers_; ++layer) {
    entry->k_layers[layer].resize(k_elems);
    entry->v_layers[layer].resize(v_elems);
    entry->k_ptrs[layer] = entry->k_layers[layer].data();
    entry->v_ptrs[layer] = entry->v_layers[layer].data();
  }

  evict_if_needed(entry->bytes);
  lru_.push_front(row_id);
  Entry* raw = entry.get();
  entries_[row_id] = {std::move(entry), lru_.begin()};
  current_bytes_ += raw->bytes;
  return {false, raw};
}

template <typename T>
size_t CPUKVPool<T>::entry_bytes(int seq_len) const {
  const size_t k_bytes =
      static_cast<size_t>(num_layers_) * num_heads_ * head_dim_ * seq_len *
      sizeof(T);
  const size_t v_bytes =
      static_cast<size_t>(num_layers_) * num_heads_ * seq_len * head_dim_ *
      sizeof(T);
  return k_bytes + v_bytes;
}

template <typename T>
void CPUKVPool<T>::touch(int row_id) {
  auto it = entries_.find(row_id);
  if (it == entries_.end()) {
    return;
  }
  lru_.erase(it->second.lru_it);
  lru_.push_front(row_id);
  it->second.lru_it = lru_.begin();
}

template <typename T>
void CPUKVPool<T>::evict_if_needed(size_t incoming_bytes) {
  while (!lru_.empty() && current_bytes_ + incoming_bytes > capacity_bytes_) {
    const int evict_row_id = lru_.back();
    auto it = entries_.find(evict_row_id);
    if (it == entries_.end()) {
      lru_.pop_back();
      continue;
    }
    current_bytes_ -= it->second.entry->bytes;
    lru_.pop_back();
    entries_.erase(it);
  }
}

} // namespace LMStore
