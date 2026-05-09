## MobileLMStore

### 0. Installation


### 1. Use

#### 1.1 MNN use



#### 1.2 ExecuTorch use
Always call `write_prompt_tokens` first for single write.

```c
// store kv
if (use_kv_store_ && !matched) {
ET_LOG(
    Info,
    "Storing KV caches to disk...\n");
kv_store_->write_prompt_tokens(prompt_tokens);
kv_store_->write_next_token(cur_token);
kv_manager_->transfer_cache(k_store, v_store, prompt_tokens.size(),
    prompt_tokens.size(), prompt_tokens.size(), true);
kv_store_->transfer_cache(k_store, v_store, true);
}
```

### 2. Support

#### 2.1 KVStore Support
- [ ] MNN format KV
    - [ ] SQLite KV + CPU Attention
    - [ ] SQLite KV + KVIO CPU Attention
    - [ ] Hierarchical KV + KVIO CPU Attention
    - [ ] Production KV + KVIO CPU Attention
    - [ ] SQLite KV + KVIO QNN Attention

- [ ] ExecuTorch-QNN format KV
    - [x] Naive KV + ExecuTorch
    - [x] SQLite KV + ExecuTorch
    - [ ] Hierarchical KV + ExecuTorch
    - [ ] Blender KV + ExecuTorch
    - [ ] Production KV + ExecuTorch

- [ ] llama.cpp format KV
    - [ ] SQLite KV + llama.cpp
    - [ ] Hierarchical KV + llama.cpp
    - [ ] Production KV + llama.cpp

#### 2.2 VectorDB Support
- [ ] SQLite Flat Index DB
    - [ ] FP32 matching
    - [ ] Batched query
    - [ ] SQ (INT4/INT8/UINT4/UINT8) matching

- [ ] SQLite MicroNN Index DB
    - [ ] No update version
    - [ ] Updatable version

- [ ] SQLite PQ Index DB
    - [ ] Support PQ op
    - [ ] PQ Index DB

