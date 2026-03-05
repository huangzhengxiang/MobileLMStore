
#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include <cstring>

#include <KVStoreMeta.h>

namespace LMStore {

template<typename T>
class NaiveKVStore {
private:
    std::string filename_;
    int fd_;
    KVStoreMeta meta_;

public:
    NaiveKVStore(const std::string& filename,
                 int num_layers, int num_heads, int head_dim, int seq_dim) 
        : filename_(filename), fd_(-1) {
        open_write();
        read_header(); // read header
        if (meta_.num_layers != num_layers ||
            meta_.num_heads != num_heads ||
            meta_.head_dim != head_dim ||
            meta_.seq_dim != seq_dim) {
            ftruncate(fd_, 0); // clear cache
            meta_.num_layers = num_layers;
            meta_.num_heads = num_heads;
            meta_.head_dim = head_dim;
            meta_.seq_dim = seq_dim;
            meta_.valid_seq_len = 0;
            write_header(); // serialize header info
        }
    }

    bool open_write() {
        fd_ = ::open(filename_.c_str(), O_CREAT | O_RDWR, 0644);
        return fd_ >= 0;
    }
    bool open_read() {
        fd_ = ::open(filename_.c_str(), O_RDONLY);
        return fd_ >= 0;
    }
    void close_fd() {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
    }

    // header
    bool write_header() {
        return write_lower(0, &meta_, sizeof(KVStoreMeta));
    }
    bool read_header() {
        return read_lower(0, &meta_, sizeof(KVStoreMeta));
    }

    // tokens
    bool write_prompt_tokens(const std::vector<uint64_t>& prompt_tokens) {
        meta_.valid_seq_len = prompt_tokens.size();
        write_header();
        return write_lower(sizeof(KVStoreMeta), prompt_tokens.data(), meta_.valid_seq_len * sizeof(uint64_t));
    }
    bool read_prompt_tokens(std::vector<uint64_t>& prompt_tokens) {
        if (meta_.valid_seq_len == 0) { return false; } // empty
        prompt_tokens.resize(meta_.valid_seq_len);
        return read_lower(sizeof(KVStoreMeta), prompt_tokens.data(), meta_.valid_seq_len * sizeof(uint64_t));
    }
    bool write_next_token(uint64_t next_token) {
        off_t offset = sizeof(KVStoreMeta) + meta_.seq_dim * sizeof(uint64_t);
        return write_lower(offset, &next_token, sizeof(uint64_t));
    }
    bool read_next_token(uint64_t* next_token) {
        off_t offset = sizeof(KVStoreMeta) + meta_.seq_dim * sizeof(uint64_t);
        return read_lower(offset, next_token, sizeof(uint64_t));
    }


    // kv
    bool write_layer_k(
        int layer,
        const T* k_ptr)
    {
        size_t elems = (size_t)meta_.num_heads * meta_.head_dim * meta_.seq_dim;
        size_t bytes = elems * sizeof(T);
        off_t offset = sizeof(KVStoreMeta) + (meta_.seq_dim+1) * sizeof(uint64_t) \
             + layer * (2 * bytes);
        return write_lower(offset, k_ptr, bytes);
    }
    bool read_layer_k(
        int layer,
        T* k_ptr)
    {
        size_t elems = (size_t)meta_.num_heads * meta_.head_dim * meta_.seq_dim;
        size_t bytes = elems * sizeof(T);
        off_t offset = sizeof(KVStoreMeta) + (meta_.seq_dim+1) * sizeof(uint64_t) \
             + layer * (2 * bytes);
        return read_lower(offset, k_ptr, bytes);
    }
    bool write_layer_v(
        int layer,
        const T* v_ptr)
    {
        size_t elems = (size_t)meta_.num_heads * meta_.head_dim * meta_.seq_dim;
        size_t bytes = elems * sizeof(T);
        off_t offset = sizeof(KVStoreMeta) + (meta_.seq_dim+1) * sizeof(uint64_t) \
             + bytes + layer * (2 * bytes);
        return write_lower(offset, v_ptr, bytes);
    }
    bool read_layer_v(
        int layer,
        T* v_ptr)
    {
        size_t elems = (size_t)meta_.num_heads * meta_.head_dim * meta_.seq_dim;
        size_t bytes = elems * sizeof(T);
        off_t offset = sizeof(KVStoreMeta) + (meta_.seq_dim+1) * sizeof(uint64_t) \
             + bytes + layer * (2 * bytes);
        return read_lower(offset, v_ptr, bytes);
    }
    bool transfer_cache(const std::vector<T*>& k_store,
                        const std::vector<T*>& v_store,
                        bool directionM2D = true) {
        bool flag = true;
        for (int layer = 0; layer < meta_.num_layers; ++layer) {
            if (directionM2D) {
                // M2D
                flag = write_layer_k(layer, k_store[layer]); if (!flag) { return false; }
                flag = write_layer_v(layer, v_store[layer]); if (!flag) { return false; }
            } else {
                // D2M
                flag = read_layer_k(layer, k_store[layer]); if (!flag) { return false; }
                flag = read_layer_v(layer, v_store[layer]); if (!flag) { return false; }
            }
        }
        return true;
    }

    // prefix matching
    bool whole_matching(const std::vector<uint64_t>& prompt_tokens) {
        if (meta_.valid_seq_len == 0) { return false; }
        std::vector<uint64_t> compare_tokens;
        read_prompt_tokens(compare_tokens);
        if (prompt_tokens.size() != compare_tokens.size()) { return false; }
        for (int j = 0; j < prompt_tokens.size(); ++j) {
            if (prompt_tokens[j] != compare_tokens[j]) { return false; }
        }
        return true;
    }

    ~NaiveKVStore() { close_fd(); }

private:
    // lower system api
    bool write_lower(off_t offset, const void* data, size_t bytes) {
        if (lseek(fd_, offset, SEEK_SET) < 0) return false;
        const char* p = (const char*)data;
        size_t written = 0;
        while (written < bytes) {
            ssize_t ret = ::write(fd_, p + written, bytes - written);
            if (ret <= 0) return false;
            written += ret;
        }
        return true;
    }

    bool read_lower(off_t offset, void* data, size_t bytes) {
        if (lseek(fd_, offset, SEEK_SET) < 0) return false;
        char* p = (char*)data;
        size_t read_bytes = 0;
        while (read_bytes < bytes) {
            ssize_t ret = ::read(fd_, p + read_bytes, bytes - read_bytes);
            if (ret <= 0) return false;
            read_bytes += ret;
        }
        return true;
    }
};

}