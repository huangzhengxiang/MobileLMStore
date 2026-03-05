#pragma once

namespace LMStore {

struct KVStoreMeta {
    int num_layers = 0;
    int num_heads = 0;
    int head_dim = 0;
    int seq_dim = 0;
    int valid_seq_len = 0; // pad right for prompt_tokens
    KVStoreMeta() {}
    KVStoreMeta(int n_layer, int n_head, int n_head_dim, int n_seq_dim, int n_valid_len)
        : num_layers(n_layer), num_heads(n_head), head_dim(n_head_dim), seq_dim(n_seq_dim), valid_seq_len(n_valid_len) {}
};

}