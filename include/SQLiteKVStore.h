#pragma once
#include <sqlite3.h>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>

#include <KVStoreMeta.h>

namespace LMStore {

template<typename T>
class SQLiteKVStore {
// seq_dim and valid_seq_len aren't read or used from disk.
private:
    std::string filename_;
    sqlite3* db_ = nullptr;
    sqlite3_blob* blob_ = nullptr;
    KVStoreMeta meta_;
    int kv_id_ = -1;
    int seq_len_ = 0;

public:
    SQLiteKVStore(const std::string& filename,
                  int num_layers,
                  int num_heads,
                  int head_dim,
                  int seq_dim)
        : filename_(filename)
    {
        open_write();
        init_tables();
        read_header();
        if (meta_.num_layers != num_layers ||
            meta_.num_heads != num_heads ||
            meta_.head_dim != head_dim)
        {
            reset_cache();
            meta_.num_layers = num_layers;
            meta_.num_heads = num_heads;
            meta_.head_dim = head_dim;
            write_header();
        }
    }

    bool open_write() {
        return sqlite3_open(filename_.c_str(), &db_) == SQLITE_OK;
    }
    bool open_read() {
        return sqlite3_open_v2(
            filename_.c_str(),
            &db_,
            SQLITE_OPEN_READONLY,
            nullptr) == SQLITE_OK;
    }
    void close_fd() {
        if (blob_) {
            sqlite3_blob_close(blob_);
            blob_ = nullptr;
        }
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
    }
    int get_cur_kv_id() { return kv_id_; }

private:
    void init_tables() {
        const char* sql_meta =
            "CREATE TABLE IF NOT EXISTS kv_meta("
            "id INTEGER PRIMARY KEY CHECK(id=1),"
            "num_layers INTEGER,"
            "num_heads INTEGER,"
            "head_dim INTEGER,"
            "seq_dim INTEGER,"
            "valid_seq_len INTEGER);";
        sqlite3_exec(db_, sql_meta, nullptr, nullptr, nullptr);

        const char* sql_tokens =
            "CREATE TABLE IF NOT EXISTS kv_tokens("
            "kv_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "prompt_len INTEGER,"
            "prompt_tokens BLOB,"
            "next_token INTEGER);";
        sqlite3_exec(db_, sql_tokens, nullptr, nullptr, nullptr);

        const char* sql_cache =
            "CREATE TABLE IF NOT EXISTS kv_caches("
            "kv_id INTEGER PRIMARY KEY,"
            "prompt TEXT,"
            "kv_blob BLOB,"
            "FOREIGN KEY(kv_id) REFERENCES kv_tokens(kv_id));";
        sqlite3_exec(db_, sql_cache, nullptr, nullptr, nullptr);
    }

    void reset_cache() {
        sqlite3_exec(db_, "DELETE FROM kv_tokens;", nullptr, nullptr, nullptr);
        sqlite3_exec(db_, "DELETE FROM kv_caches;", nullptr, nullptr, nullptr);
    }

public:
    bool write_header() {
        sqlite3_stmt* stmt;
        const char* sql =
            "INSERT OR REPLACE INTO kv_meta "
            "(id,num_layers,num_heads,head_dim,seq_dim,valid_seq_len)"
            "VALUES(1,?,?,?,?,?);";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

        sqlite3_bind_int(stmt,1,meta_.num_layers);
        sqlite3_bind_int(stmt,2,meta_.num_heads);
        sqlite3_bind_int(stmt,3,meta_.head_dim);
        sqlite3_bind_int(stmt,4,meta_.seq_dim);
        sqlite3_bind_int(stmt,5,meta_.valid_seq_len);

        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    bool read_header() {
        sqlite3_stmt* stmt;
        const char* sql =
            "SELECT num_layers,num_heads,head_dim "
            "FROM kv_meta WHERE id=1;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            meta_.num_layers = sqlite3_column_int(stmt,0);
            meta_.num_heads  = sqlite3_column_int(stmt,1);
            meta_.head_dim   = sqlite3_column_int(stmt,2);
        }

        sqlite3_finalize(stmt);
        return true;
    }

public:
    bool write_prompt_tokens(const std::vector<uint64_t>& prompt_tokens) {
        write_header();
        sqlite3_stmt* stmt;
        const char* sql =
            "INSERT INTO kv_tokens(prompt_len, prompt_tokens) VALUES(?,?);";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

        sqlite3_bind_int(stmt,1,(int)prompt_tokens.size());
        sqlite3_bind_blob(
            stmt,
            2,
            prompt_tokens.data(),
            prompt_tokens.size()*sizeof(uint64_t),
            SQLITE_TRANSIENT);

        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        kv_id_ = sqlite3_last_insert_rowid(db_);
        seq_len_ = prompt_tokens.size();
        create_blob();
        return ok;
    }

    bool read_prompt_tokens(std::vector<uint64_t>& prompt_tokens) {
        sqlite3_stmt* stmt;
        const char* sql =
            "SELECT prompt_tokens FROM kv_tokens "
            "WHERE kv_id=?;";

        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt,1,kv_id_);

        if (sqlite3_step(stmt) != SQLITE_ROW)
            return false;

        const void* blob = sqlite3_column_blob(stmt,0);
        int bytes = sqlite3_column_bytes(stmt,0);
        int n = bytes / sizeof(uint64_t);

        prompt_tokens.resize(n);
        memcpy(prompt_tokens.data(), blob, bytes);
        sqlite3_finalize(stmt);

        return true;
    }

    bool write_next_token(uint64_t next_token) {
        sqlite3_stmt* stmt;
        const char* sql =
            "UPDATE kv_tokens SET next_token=? "
            "WHERE kv_id=?;";

        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int64(stmt,1,next_token);
        sqlite3_bind_int(stmt,2,kv_id_);

        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    bool read_next_token(uint64_t* next_token) {
        sqlite3_stmt* stmt;
        const char* sql =
            "SELECT next_token FROM kv_tokens "
            "WHERE kv_id=?;";

        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt,1,kv_id_);

        if (sqlite3_step(stmt) != SQLITE_ROW)
            return false;

        *next_token = sqlite3_column_int64(stmt,0);
        sqlite3_finalize(stmt);
        return true;
    }

private:
    int get_prompt_len() {
        sqlite3_stmt* stmt;
        const char* sql =
            "SELECT prompt_len FROM kv_tokens "
            "WHERE kv_id=?;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt,1,kv_id_);
        int prompt_len = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            prompt_len = sqlite3_column_int(stmt,0);
        }
        sqlite3_finalize(stmt);
        return prompt_len;
    }

    size_t tensor_bytes() {
        size_t elems =
            (size_t)meta_.num_heads *
            meta_.head_dim *
            seq_len_;
        return elems * sizeof(T);
    }

    void create_blob() {
        size_t bytes = meta_.num_layers * 2 * tensor_bytes();
        sqlite3_stmt* stmt;
        const char* sql =
            "INSERT INTO kv_caches(kv_id,kv_blob) VALUES(?,zeroblob(?));";

        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt,1,kv_id_);
        sqlite3_bind_int(stmt,2,bytes);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        sqlite3_blob_open(
            db_,
            "main",
            "kv_caches",
            "kv_blob",
            kv_id_,
            1,
            &blob_);
    }

public:

    bool write_layer_k(int layer,const T* k_ptr) {
        size_t bytes = tensor_bytes();
        int offset = layer * 2 * bytes;
        return sqlite3_blob_write(blob_,k_ptr,bytes,offset) == SQLITE_OK;
    }

    bool read_layer_k(int layer,T* k_ptr) {
        size_t bytes = tensor_bytes();
        int offset = layer * 2 * bytes;
        return sqlite3_blob_read(blob_,k_ptr,bytes,offset) == SQLITE_OK;
    }

    bool write_layer_v(int layer,const T* v_ptr) {
        size_t bytes = tensor_bytes();
        int offset = layer * 2 * bytes + bytes;
        return sqlite3_blob_write(blob_,v_ptr,bytes,offset) == SQLITE_OK;
    }

    bool read_layer_v(int layer,T* v_ptr) {
        size_t bytes = tensor_bytes();
        int offset = layer * 2 * bytes + bytes;
        return sqlite3_blob_read(blob_,v_ptr,bytes,offset) == SQLITE_OK;
    }

    bool transfer_cache(const std::vector<T*>& k_store,
                        const std::vector<T*>& v_store,
                        bool directionM2D = true)
    {
        for (int layer = 0; layer < meta_.num_layers; ++layer) {
            if (directionM2D) {
                if (!write_layer_k(layer,k_store[layer])) return false;
                if (!write_layer_v(layer,v_store[layer])) return false;
            } else {
                if (!read_layer_k(layer,k_store[layer])) return false;
                if (!read_layer_v(layer,v_store[layer])) return false;
            }
        }
        return true;
    }

    bool whole_matching(const std::vector<uint64_t>& prompt_tokens) {
        sqlite3_stmt* stmt;
        const char* sql =
            "SELECT kv_id, prompt_tokens FROM kv_tokens;";

        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK)
            return false;

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int kv_id = sqlite3_column_int(stmt, 0);
            const void* blob = sqlite3_column_blob(stmt, 1);
            int bytes = sqlite3_column_bytes(stmt, 1);
            size_t n = bytes / sizeof(uint64_t);

            if (n != prompt_tokens.size())
                continue;

            const uint64_t* tokens = static_cast<const uint64_t*>(blob);
            bool match = true;

            for (size_t i = 0; i < n; i++) {
                if (tokens[i] != prompt_tokens[i]) {
                    match = false;
                    break;
                }
            }

            if (match) {
                kv_id_ = kv_id;
                seq_len_ = get_prompt_len();
                if (blob_) {
                    sqlite3_blob_close(blob_);
                    blob_ = nullptr;
                }
                sqlite3_blob_open(
                    db_,
                    "main",
                    "kv_caches",
                    "kv_blob",
                    kv_id_,
                    1,
                    &blob_);
                sqlite3_finalize(stmt);
                return true;
            }
        }
        sqlite3_finalize(stmt);
        return false;
    }

    ~SQLiteKVStore() {
        close_fd();
    }
};

}