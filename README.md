## MobileLMStore

### 0. Installation


### 1. Use

#### 1.1 ExecuTorch use
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
- [ ] ExecuTorch format KV
    - [x] Naive KV + ExecuTorch
    - [x] SQLite KV + ExecuTorch
    - [ ] Hierarchical KV + ExecuTorch
    - [ ] Blender KV + ExecuTorch
    - [ ] Production KV + ExecuTorch