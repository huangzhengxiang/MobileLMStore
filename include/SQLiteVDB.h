#pragma once

#include <sqlite3.h>

#include <cstdint>
#include <string>
#include <vector>

#include <VectorDBMeta.h>

namespace LMStore {

struct VDBSearchResult {
    int vec_id = -1;
    float distance = 0.0f;
    std::string prompt;
    std::string kv_cache_file;
    int kv_id = -1;
};

class SQLiteVDB {
private:
    std::string filename_;
    sqlite3* db_ = nullptr;
    VectorDBMeta meta_;

public:
    explicit SQLiteVDB(const std::string& filename)
        : filename_(filename) {}

    ~SQLiteVDB() { close_fd(); }

    bool open_write();
    bool open_read();
    void close_fd();

    bool init_tables();
    bool reset();

    bool write_meta(const VectorDBMeta& meta);
    bool read_meta();
    const VectorDBMeta& get_meta() const { return meta_; }

    bool insert_prompt(const std::string& prompt, int* vec_id);
    bool insert_vector(
        int vec_id,
        const void* vector_blob,
        int num_bytes,
        float scale = 0.0f,
        int zero_point = 0);
    bool insert_kv(int vec_id, int kv_id, const std::string& kv_cache_file);

    bool read_prompt(int vec_id, std::string* prompt) const;
    bool read_vector(int vec_id, std::vector<std::uint8_t>* vector_blob) const;
    bool read_kv(int vec_id, int* kv_id, std::string* kv_cache_file) const;

    bool update_prompt(int vec_id, const std::string& prompt);
    bool update_vector(
        int vec_id,
        const void* vector_blob,
        int num_bytes,
        float scale = 0.0f,
        int zero_point = 0);
    bool update_kv(int vec_id, int kv_id, const std::string& kv_cache_file);

    bool delete_entry(int vec_id);

    bool list_vec_ids(std::vector<int>* vec_ids) const;
    bool count(int* num_vectors) const;

    bool search(
        const float* query_vector,
        int top_k,
        std::vector<VDBSearchResult>* results,
        bool include_payload = true) const;
};

} // namespace LMStore
