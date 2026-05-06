#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sqlite3.h>
#include <string>
#include <vector>

#include <KVStoreMeta.h>
#include <chunk_merge.h>

namespace LMStore {

constexpr int kKVStoreChunkLimit = 512;

struct BuildInputMatch {
    int row_id = -1;
    int cur_start_pos = 0;
    int ori_start_pos = -1;
    int matched_len = 0;
    ChunkPromptType type = ChunkPromptType::kNew;
};

struct RecipeOrigin {
    int row_id = -1;
    int cur_start_pos = 0;
    int ori_start_pos = -1;
    int matched_len = 0;
    ChunkPromptType type = ChunkPromptType::kNew;
};

struct ChunkRecipe {
    int start = 0;
    int end = -1;
    ChunkGraphType group = ChunkGraphType::kPrompt;
    std::vector<RecipeOrigin> origins;
};

struct BuildInputResult {
    std::vector<BuildInputMatch> raw_matches;
    std::vector<BuildInputMatch> segments;
    ChunkMergeResult merge_result;
    std::vector<ChunkRecipe> recipes;
};

template<typename T>
class SQLiteKVStore {
private:
    struct RowInfo {
        int kv_id = -1;
        int prompt_len = 0;
        std::vector<uint64_t> prompt_tokens;
        uint64_t next_token = 0;
        int64_t ori_pos = 0;
        int prev_kv_id = -1;
        int succ_kv_id = -1;
        int root_kv_id = -1;
        int chunk_start = 0;
    };

    std::string filename_;
    sqlite3* db_ = nullptr;
    sqlite3_blob* blob_ = nullptr;
    KVStoreMeta meta_;
    int kv_id_ = -1;
    int seq_len_ = 0;
    int active_root_kv_id_ = -1;

public:
    SQLiteKVStore(const std::string& filename,
                  int num_layers,
                  int num_heads,
                  int head_dim,
                  int seq_dim)
        : filename_(filename) {
        open_write();
        init_tables();
        read_header();
        if (meta_.num_layers != num_layers ||
            meta_.num_heads != num_heads ||
            meta_.head_dim != head_dim) {
            reset_cache();
            meta_.num_layers = num_layers;
            meta_.num_heads = num_heads;
            meta_.head_dim = head_dim;
            meta_.seq_dim = seq_dim;
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
        close_blob();
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
    }
    int get_cur_kv_id() const { return kv_id_; }

private:
    void close_blob() {
        if (blob_) {
            sqlite3_blob_close(blob_);
            blob_ = nullptr;
        }
    }

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
            "next_token INTEGER,"
            "ori_pos INTEGER DEFAULT 0,"
            "prev_kv_id INTEGER DEFAULT -1,"
            "succ_kv_id INTEGER DEFAULT -1,"
            "root_kv_id INTEGER DEFAULT -1,"
            "chunk_start INTEGER DEFAULT 0);";
        sqlite3_exec(db_, sql_tokens, nullptr, nullptr, nullptr);

        const char* sql_cache =
            "CREATE TABLE IF NOT EXISTS kv_caches("
            "kv_id INTEGER PRIMARY KEY,"
            "prompt TEXT,"
            "kv_blob BLOB,"
            "FOREIGN KEY(kv_id) REFERENCES kv_tokens(kv_id));";
        sqlite3_exec(db_, sql_cache, nullptr, nullptr, nullptr);

        ensure_kv_tokens_columns();
    }

    void ensure_kv_tokens_columns() {
        struct ColumnDef {
            const char* name;
            const char* alter_sql;
        };
        static const ColumnDef kColumns[] = {
            {"prev_kv_id", "ALTER TABLE kv_tokens ADD COLUMN prev_kv_id INTEGER DEFAULT -1;"},
            {"succ_kv_id", "ALTER TABLE kv_tokens ADD COLUMN succ_kv_id INTEGER DEFAULT -1;"},
            {"root_kv_id", "ALTER TABLE kv_tokens ADD COLUMN root_kv_id INTEGER DEFAULT -1;"},
            {"chunk_start", "ALTER TABLE kv_tokens ADD COLUMN chunk_start INTEGER DEFAULT 0;"},
        };
        std::vector<std::string> existing;
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db_, "PRAGMA table_info(kv_tokens);", -1, &stmt, nullptr) ==
            SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                const unsigned char* text = sqlite3_column_text(stmt, 1);
                if (text != nullptr) {
                    existing.emplace_back(reinterpret_cast<const char*>(text));
                }
            }
        }
        sqlite3_finalize(stmt);
        for (size_t i = 0; i < sizeof(kColumns) / sizeof(kColumns[0]); ++i) {
            if (std::find(existing.begin(), existing.end(), kColumns[i].name) ==
                existing.end()) {
                sqlite3_exec(db_, kColumns[i].alter_sql, nullptr, nullptr, nullptr);
            }
        }
    }

    void reset_cache() {
        sqlite3_exec(db_, "DELETE FROM kv_tokens;", nullptr, nullptr, nullptr);
        sqlite3_exec(db_, "DELETE FROM kv_caches;", nullptr, nullptr, nullptr);
    }

    bool read_row_by_id(int kv_id, RowInfo* row) {
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "SELECT kv_id,prompt_len,prompt_tokens,next_token,ori_pos,prev_kv_id,"
            "succ_kv_id,root_kv_id,chunk_start FROM kv_tokens WHERE kv_id=?;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, kv_id);
        if (sqlite3_step(stmt) != SQLITE_ROW) {
            sqlite3_finalize(stmt);
            return false;
        }
        row->kv_id = sqlite3_column_int(stmt, 0);
        row->prompt_len = sqlite3_column_int(stmt, 1);
        const void* blob = sqlite3_column_blob(stmt, 2);
        int bytes = sqlite3_column_bytes(stmt, 2);
        row->prompt_tokens.resize(bytes / static_cast<int>(sizeof(uint64_t)));
        if (blob != nullptr && bytes > 0) {
            std::memcpy(row->prompt_tokens.data(), blob, bytes);
        }
        row->next_token = static_cast<uint64_t>(sqlite3_column_int64(stmt, 3));
        row->ori_pos = sqlite3_column_type(stmt, 4) == SQLITE_NULL
            ? 0
            : sqlite3_column_int64(stmt, 4);
        row->prev_kv_id = sqlite3_column_int(stmt, 5);
        row->succ_kv_id = sqlite3_column_int(stmt, 6);
        row->root_kv_id = sqlite3_column_int(stmt, 7);
        row->chunk_start = sqlite3_column_int(stmt, 8);
        sqlite3_finalize(stmt);
        return true;
    }

    std::vector<RowInfo> read_all_rows() {
        std::vector<RowInfo> rows;
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "SELECT kv_id,prompt_len,prompt_tokens,next_token,ori_pos,prev_kv_id,"
            "succ_kv_id,root_kv_id,chunk_start FROM kv_tokens ORDER BY kv_id ASC;";
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            return rows;
        }
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            RowInfo row;
            row.kv_id = sqlite3_column_int(stmt, 0);
            row.prompt_len = sqlite3_column_int(stmt, 1);
            const void* blob = sqlite3_column_blob(stmt, 2);
            int bytes = sqlite3_column_bytes(stmt, 2);
            row.prompt_tokens.resize(bytes / static_cast<int>(sizeof(uint64_t)));
            if (blob != nullptr && bytes > 0) {
                std::memcpy(row.prompt_tokens.data(), blob, bytes);
            }
            row.next_token = static_cast<uint64_t>(sqlite3_column_int64(stmt, 3));
            row.ori_pos = sqlite3_column_type(stmt, 4) == SQLITE_NULL
                ? 0
                : sqlite3_column_int64(stmt, 4);
            row.prev_kv_id = sqlite3_column_int(stmt, 5);
            row.succ_kv_id = sqlite3_column_int(stmt, 6);
            row.root_kv_id = sqlite3_column_int(stmt, 7);
            row.chunk_start = sqlite3_column_int(stmt, 8);
            rows.push_back(row);
        }
        sqlite3_finalize(stmt);
        return rows;
    }

    bool update_succ_kv_id(int kv_id, int succ_kv_id) {
        sqlite3_stmt* stmt = nullptr;
        const char* sql = "UPDATE kv_tokens SET succ_kv_id=? WHERE kv_id=?;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, succ_kv_id);
        sqlite3_bind_int(stmt, 2, kv_id);
        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    std::vector<uint64_t> reconstruct_chain_prompt_tokens(int root_kv_id) {
        std::vector<uint64_t> tokens;
        RowInfo row;
        if (!read_row_by_id(root_kv_id, &row)) {
            return tokens;
        }
        while (true) {
            tokens.insert(tokens.end(), row.prompt_tokens.begin(), row.prompt_tokens.end());
            if (row.succ_kv_id < 0) {
                break;
            }
            if (!read_row_by_id(row.succ_kv_id, &row)) {
                break;
            }
        }
        return tokens;
    }

    int get_root_prompt_len(int root_kv_id) {
        return static_cast<int>(reconstruct_chain_prompt_tokens(root_kv_id).size());
    }

    size_t tensor_bytes() const {
        size_t elems =
            static_cast<size_t>(meta_.num_heads) *
            meta_.head_dim *
            seq_len_;
        return elems * sizeof(T);
    }

    bool open_blob_for_row(int kv_id, int seq_len) {
        close_blob();
        kv_id_ = kv_id;
        seq_len_ = seq_len;
        return sqlite3_blob_open(
            db_,
            "main",
            "kv_caches",
            "kv_blob",
            kv_id_,
            1,
            &blob_) == SQLITE_OK;
    }

    bool create_blob_for_active_row() {
        size_t bytes = static_cast<size_t>(meta_.num_layers) * 2 * tensor_bytes();
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "INSERT INTO kv_caches(kv_id,kv_blob) VALUES(?,zeroblob(?));";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, kv_id_);
        sqlite3_bind_int(stmt, 2, static_cast<int>(bytes));
        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        if (!ok) {
            return false;
        }
        return open_blob_for_row(kv_id_, seq_len_);
    }

    int insert_prompt_row(const std::vector<uint64_t>& prompt_tokens,
                          uint64_t next_token,
                          int64_t ori_pos,
                          int prev_kv_id,
                          int root_kv_id,
                          int chunk_start) {
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "INSERT INTO kv_tokens("
            "prompt_len,prompt_tokens,next_token,ori_pos,prev_kv_id,succ_kv_id,root_kv_id,chunk_start)"
            " VALUES(?,?,?,?,?,-1,?,?);";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, static_cast<int>(prompt_tokens.size()));
        sqlite3_bind_blob(
            stmt,
            2,
            prompt_tokens.empty() ? nullptr : prompt_tokens.data(),
            static_cast<int>(prompt_tokens.size() * sizeof(uint64_t)),
            SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 3, static_cast<sqlite3_int64>(next_token));
        sqlite3_bind_int64(stmt, 4, static_cast<sqlite3_int64>(ori_pos));
        sqlite3_bind_int(stmt, 5, prev_kv_id);
        sqlite3_bind_int(stmt, 6, root_kv_id);
        sqlite3_bind_int(stmt, 7, chunk_start);
        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        if (!ok) {
            return -1;
        }
        return static_cast<int>(sqlite3_last_insert_rowid(db_));
    }

    bool update_root_kv_id(int kv_id, int root_kv_id) {
        sqlite3_stmt* stmt = nullptr;
        const char* sql = "UPDATE kv_tokens SET root_kv_id=? WHERE kv_id=?;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, root_kv_id);
        sqlite3_bind_int(stmt, 2, kv_id);
        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    void pack_k_chunk(
        const T* src,
        int full_len,
        int chunk_start,
        int chunk_len,
        std::vector<T>* packed) const {
        packed->resize(
            static_cast<size_t>(meta_.num_heads) * meta_.head_dim * chunk_len);
        for (int i = 0; i < meta_.num_heads * meta_.head_dim; ++i) {
            const T* read_ptr = src + static_cast<size_t>(i) * full_len + chunk_start;
            T* write_ptr = packed->data() + static_cast<size_t>(i) * chunk_len;
            std::memcpy(write_ptr, read_ptr, static_cast<size_t>(chunk_len) * sizeof(T));
        }
    }

    void pack_v_chunk(
        const T* src,
        int full_len,
        int chunk_start,
        int chunk_len,
        std::vector<T>* packed) const {
        packed->resize(
            static_cast<size_t>(meta_.num_heads) * chunk_len * meta_.head_dim);
        for (int head = 0; head < meta_.num_heads; ++head) {
            const T* read_ptr =
                src + static_cast<size_t>(head) * full_len * meta_.head_dim +
                static_cast<size_t>(chunk_start) * meta_.head_dim;
            T* write_ptr =
                packed->data() + static_cast<size_t>(head) * chunk_len * meta_.head_dim;
            std::memcpy(
                write_ptr,
                read_ptr,
                static_cast<size_t>(chunk_len) * meta_.head_dim * sizeof(T));
        }
    }

    void unpack_k_chunk(
        const std::vector<T>& packed,
        int full_len,
        int chunk_start,
        int chunk_len,
        T* dst) const {
        for (int i = 0; i < meta_.num_heads * meta_.head_dim; ++i) {
            const T* read_ptr = packed.data() + static_cast<size_t>(i) * chunk_len;
            T* write_ptr = dst + static_cast<size_t>(i) * full_len + chunk_start;
            std::memcpy(write_ptr, read_ptr, static_cast<size_t>(chunk_len) * sizeof(T));
        }
    }

    void unpack_v_chunk(
        const std::vector<T>& packed,
        int full_len,
        int chunk_start,
        int chunk_len,
        T* dst) const {
        for (int head = 0; head < meta_.num_heads; ++head) {
            const T* read_ptr =
                packed.data() + static_cast<size_t>(head) * chunk_len * meta_.head_dim;
            T* write_ptr =
                dst + static_cast<size_t>(head) * full_len * meta_.head_dim +
                static_cast<size_t>(chunk_start) * meta_.head_dim;
            std::memcpy(
                write_ptr,
                read_ptr,
                static_cast<size_t>(chunk_len) * meta_.head_dim * sizeof(T));
        }
    }

    void unpack_k_chunk_slice(
        const std::vector<T>& packed,
        int packed_chunk_len,
        int src_offset_in_chunk,
        int copy_len,
        int full_len,
        int dst_start,
        T* dst) const {
        for (int i = 0; i < meta_.num_heads * meta_.head_dim; ++i) {
            const T* read_ptr =
                packed.data() + static_cast<size_t>(i) * packed_chunk_len + src_offset_in_chunk;
            T* write_ptr = dst + static_cast<size_t>(i) * full_len + dst_start;
            std::memcpy(write_ptr, read_ptr, static_cast<size_t>(copy_len) * sizeof(T));
        }
    }

    void unpack_v_chunk_slice(
        const std::vector<T>& packed,
        int packed_chunk_len,
        int src_offset_in_chunk,
        int copy_len,
        int full_len,
        int dst_start,
        T* dst) const {
        for (int head = 0; head < meta_.num_heads; ++head) {
            const T* read_ptr =
                packed.data() +
                static_cast<size_t>(head) * packed_chunk_len * meta_.head_dim +
                static_cast<size_t>(src_offset_in_chunk) * meta_.head_dim;
            T* write_ptr =
                dst +
                static_cast<size_t>(head) * full_len * meta_.head_dim +
                static_cast<size_t>(dst_start) * meta_.head_dim;
            std::memcpy(
                write_ptr,
                read_ptr,
                static_cast<size_t>(copy_len) * meta_.head_dim * sizeof(T));
        }
    }

    bool write_chunk_cache_layer(
        const T* full_k,
        const T* full_v,
        int layer,
        int full_len,
        int chunk_start,
        int chunk_len) {
        std::vector<T> packed_k;
        std::vector<T> packed_v;
        pack_k_chunk(full_k, full_len, chunk_start, chunk_len, &packed_k);
        pack_v_chunk(full_v, full_len, chunk_start, chunk_len, &packed_v);
        return write_layer_k(layer, packed_k.data()) &&
            write_layer_v(layer, packed_v.data());
    }

    bool read_chunk_cache_layer(
        T* full_k,
        T* full_v,
        int layer,
        int full_len,
        int chunk_start,
        int chunk_len) {
        std::vector<T> packed_k(
            static_cast<size_t>(meta_.num_heads) * meta_.head_dim * chunk_len);
        std::vector<T> packed_v(
            static_cast<size_t>(meta_.num_heads) * chunk_len * meta_.head_dim);
        if (!read_layer_k(layer, packed_k.data()) || !read_layer_v(layer, packed_v.data())) {
            return false;
        }
        unpack_k_chunk(packed_k, full_len, chunk_start, chunk_len, full_k);
        unpack_v_chunk(packed_v, full_len, chunk_start, chunk_len, full_v);
        return true;
    }

    bool read_chunk_cache_layer_slice(
        T* full_k,
        T* full_v,
        int layer,
        int dst_buffer_seq_dim,
        int src_offset_in_chunk,
        int dst_start,
        int packed_chunk_len,
        int copy_len) {
        std::vector<T> packed_k(
            static_cast<size_t>(meta_.num_heads) * meta_.head_dim * packed_chunk_len);
        std::vector<T> packed_v(
            static_cast<size_t>(meta_.num_heads) * packed_chunk_len * meta_.head_dim);
        if (!read_layer_k(layer, packed_k.data()) || !read_layer_v(layer, packed_v.data())) {
            return false;
        }
        unpack_k_chunk_slice(
            packed_k,
            packed_chunk_len,
            src_offset_in_chunk,
            copy_len,
            dst_buffer_seq_dim,
            dst_start,
            full_k);
        unpack_v_chunk_slice(
            packed_v,
            packed_chunk_len,
            src_offset_in_chunk,
            copy_len,
            dst_buffer_seq_dim,
            dst_start,
            full_v);
        return true;
    }

    static int prefix_match_len(
        const std::vector<uint64_t>& prompt_tokens,
        const std::vector<uint64_t>& chunk_tokens,
        int prompt_start) {
        int matched = 0;
        while (prompt_start + matched < static_cast<int>(prompt_tokens.size()) &&
               matched < static_cast<int>(chunk_tokens.size()) &&
               prompt_tokens[prompt_start + matched] == chunk_tokens[matched]) {
            ++matched;
        }
        return matched;
    }

    static BuildInputMatch longest_common_substring_match(
        const std::vector<uint64_t>& prompt_tokens,
        const RowInfo& row) {
        BuildInputMatch best;
        best.row_id = row.kv_id;
        best.type = ChunkPromptType::kNonPrefix;
        const int n = static_cast<int>(prompt_tokens.size());
        const int m = static_cast<int>(row.prompt_tokens.size());
        if (n == 0 || m == 0) {
            return best;
        }
        std::vector<int> prev(m + 1, 0);
        std::vector<int> cur(m + 1, 0);
        int best_len = 0;
        int best_end_i = -1;
        int best_end_j = -1;
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                if (prompt_tokens[i - 1] == row.prompt_tokens[j - 1]) {
                    cur[j] = prev[j - 1] + 1;
                    if (cur[j] > best_len) {
                        best_len = cur[j];
                        best_end_i = i - 1;
                        best_end_j = j - 1;
                    }
                } else {
                    cur[j] = 0;
                }
            }
            std::swap(prev, cur);
            std::fill(cur.begin(), cur.end(), 0);
        }
        best.cur_start_pos = best_end_i - best_len + 1;
        best.ori_start_pos = row.chunk_start + best_end_j - best_len + 1;
        best.matched_len = best_len;
        return best;
    }

    static bool prefer_match(const BuildInputMatch& lhs, const BuildInputMatch& rhs) {
        if (lhs.type == ChunkPromptType::kPrefix &&
            rhs.type != ChunkPromptType::kPrefix) {
            return true;
        }
        if (lhs.type != ChunkPromptType::kPrefix &&
            rhs.type == ChunkPromptType::kPrefix) {
            return false;
        }
        if (lhs.matched_len != rhs.matched_len) {
            return lhs.matched_len > rhs.matched_len;
        }
        return lhs.row_id < rhs.row_id;
    }

public:
    bool write_header() {
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "INSERT OR REPLACE INTO kv_meta "
            "(id,num_layers,num_heads,head_dim,seq_dim,valid_seq_len)"
            "VALUES(1,?,?,?,?,?);";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, meta_.num_layers);
        sqlite3_bind_int(stmt, 2, meta_.num_heads);
        sqlite3_bind_int(stmt, 3, meta_.head_dim);
        sqlite3_bind_int(stmt, 4, meta_.seq_dim);
        sqlite3_bind_int(stmt, 5, meta_.valid_seq_len);
        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    bool read_header() {
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "SELECT num_layers,num_heads,head_dim,seq_dim,valid_seq_len "
            "FROM kv_meta WHERE id=1;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            meta_.num_layers = sqlite3_column_int(stmt, 0);
            meta_.num_heads  = sqlite3_column_int(stmt, 1);
            meta_.head_dim   = sqlite3_column_int(stmt, 2);
            meta_.seq_dim = sqlite3_column_int(stmt, 3);
            meta_.valid_seq_len = sqlite3_column_int(stmt, 4);
        }
        sqlite3_finalize(stmt);
        return true;
    }

    bool read_prompt_tokens(std::vector<uint64_t>& prompt_tokens) {
        int root_id = active_root_kv_id_ >= 0 ? active_root_kv_id_ : kv_id_;
        if (root_id < 0) {
            return false;
        }
        prompt_tokens = reconstruct_chain_prompt_tokens(root_id);
        return !prompt_tokens.empty();
    }

    bool write_next_token(uint64_t next_token) {
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "UPDATE kv_tokens SET next_token=? "
            "WHERE kv_id=?;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int64(stmt, 1, next_token);
        sqlite3_bind_int(stmt, 2, kv_id_);
        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    bool read_next_token(uint64_t* next_token) {
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "SELECT next_token FROM kv_tokens "
            "WHERE kv_id=?;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, active_root_kv_id_ >= 0 ? active_root_kv_id_ : kv_id_);
        if (sqlite3_step(stmt) != SQLITE_ROW) {
            sqlite3_finalize(stmt);
            return false;
        }
        *next_token = sqlite3_column_int64(stmt, 0);
        sqlite3_finalize(stmt);
        return true;
    }

    bool read_next_token(int row_id, uint64_t* next_token) {
        RowInfo row;
        if (!read_row_by_id(row_id, &row)) {
            return false;
        }
        int root_id = row.root_kv_id >= 0 ? row.root_kv_id : row.kv_id;
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "SELECT next_token FROM kv_tokens "
            "WHERE kv_id=?;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, root_id);
        if (sqlite3_step(stmt) != SQLITE_ROW) {
            sqlite3_finalize(stmt);
            return false;
        }
        *next_token = sqlite3_column_int64(stmt, 0);
        sqlite3_finalize(stmt);
        return true;
    }

    bool write_ori_pos(int64_t ori_pos) {
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "UPDATE kv_tokens SET ori_pos=? "
            "WHERE kv_id=?;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int64(stmt, 1, ori_pos);
        sqlite3_bind_int(stmt, 2, kv_id_);
        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    bool read_ori_pos(int64_t* ori_pos) {
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "SELECT ori_pos FROM kv_tokens "
            "WHERE kv_id=?;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, active_root_kv_id_ >= 0 ? active_root_kv_id_ : kv_id_);
        if (sqlite3_step(stmt) != SQLITE_ROW) {
            sqlite3_finalize(stmt);
            return false;
        }
        *ori_pos = sqlite3_column_type(stmt, 0) == SQLITE_NULL
            ? 0
            : sqlite3_column_int64(stmt, 0);
        sqlite3_finalize(stmt);
        return true;
    }

    bool read_ori_pos(int row_id, int64_t* ori_pos) {
        RowInfo row;
        if (!read_row_by_id(row_id, &row)) {
            return false;
        }
        int root_id = row.root_kv_id >= 0 ? row.root_kv_id : row.kv_id;
        sqlite3_stmt* stmt = nullptr;
        const char* sql =
            "SELECT ori_pos FROM kv_tokens "
            "WHERE kv_id=?;";
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, root_id);
        if (sqlite3_step(stmt) != SQLITE_ROW) {
            sqlite3_finalize(stmt);
            return false;
        }
        *ori_pos = sqlite3_column_type(stmt, 0) == SQLITE_NULL
            ? 0
            : sqlite3_column_int64(stmt, 0);
        sqlite3_finalize(stmt);
        return true;
    }

    bool store_prompt_cache(const std::vector<uint64_t>& prompt_tokens,
                            uint64_t next_token,
                            int64_t ori_pos,
                            const std::vector<T*>& k_store,
                            const std::vector<T*>& v_store,
                            int chunk_limit = kKVStoreChunkLimit) {
        write_header();
        int full_len = static_cast<int>(prompt_tokens.size());
        int root_row_id = -1;
        int prev_row_id = -1;
        for (int chunk_start = 0; chunk_start < full_len; chunk_start += chunk_limit) {
            int chunk_len = std::min(chunk_limit, full_len - chunk_start);
            std::vector<uint64_t> chunk_tokens(
                prompt_tokens.begin() + chunk_start,
                prompt_tokens.begin() + chunk_start + chunk_len);
            int row_id = insert_prompt_row(
                chunk_tokens,
                chunk_start == 0 ? next_token : 0,
                ori_pos,
                prev_row_id,
                root_row_id >= 0 ? root_row_id : -1,
                chunk_start);
            if (row_id < 0) {
                return false;
            }
            if (root_row_id < 0) {
                root_row_id = row_id;
                if (!update_root_kv_id(root_row_id, root_row_id)) {
                    return false;
                }
            } else {
                if (!update_root_kv_id(row_id, root_row_id) ||
                    !update_succ_kv_id(prev_row_id, row_id)) {
                    return false;
                }
            }

            kv_id_ = row_id;
            seq_len_ = chunk_len;
            if (!create_blob_for_active_row()) {
                return false;
            }
            for (int layer = 0; layer < meta_.num_layers; ++layer) {
                if (!write_chunk_cache_layer(
                        k_store[layer],
                        v_store[layer],
                        layer,
                        full_len,
                        chunk_start,
                        chunk_len)) {
                    return false;
                }
            }
            prev_row_id = row_id;
        }
        close_blob();
        kv_id_ = root_row_id;
        active_root_kv_id_ = root_row_id;
        return true;
    }

    bool write_layer_k(int layer,const T* k_ptr) {
        size_t bytes = tensor_bytes();
        int offset = static_cast<int>(layer * 2 * bytes);
        return sqlite3_blob_write(blob_,k_ptr,static_cast<int>(bytes),offset) == SQLITE_OK;
    }

    bool read_layer_k(int layer,T* k_ptr) {
        size_t bytes = tensor_bytes();
        int offset = static_cast<int>(layer * 2 * bytes);
        return sqlite3_blob_read(blob_,k_ptr,static_cast<int>(bytes),offset) == SQLITE_OK;
    }

    bool write_layer_v(int layer,const T* v_ptr) {
        size_t bytes = tensor_bytes();
        int offset = static_cast<int>(layer * 2 * bytes + bytes);
        return sqlite3_blob_write(blob_,v_ptr,static_cast<int>(bytes),offset) == SQLITE_OK;
    }

    bool read_layer_v(int layer,T* v_ptr) {
        size_t bytes = tensor_bytes();
        int offset = static_cast<int>(layer * 2 * bytes + bytes);
        return sqlite3_blob_read(blob_,v_ptr,static_cast<int>(bytes),offset) == SQLITE_OK;
    }

    bool transfer_cache(const std::vector<T*>& k_store,
                        const std::vector<T*>& v_store,
                        bool directionM2D = true) {
        if (directionM2D) {
            for (int layer = 0; layer < meta_.num_layers; ++layer) {
                if (!transfer_cache_layer(k_store[layer], v_store[layer], layer, true)) {
                    return false;
                }
            }
            return true;
        }
        int root_id = active_root_kv_id_ >= 0 ? active_root_kv_id_ : kv_id_;
        if (root_id < 0) {
            return false;
        }
        int full_len = get_root_prompt_len(root_id);
        RowInfo row;
        if (!read_row_by_id(root_id, &row)) {
            return false;
        }
        while (true) {
            if (!open_blob_for_row(row.kv_id, row.prompt_len)) {
                return false;
            }
            for (int layer = 0; layer < meta_.num_layers; ++layer) {
                if (!read_chunk_cache_layer(
                        k_store[layer],
                        v_store[layer],
                        layer,
                        full_len,
                        row.chunk_start,
                        row.prompt_len)) {
                    return false;
                }
            }
            if (row.succ_kv_id < 0) {
                break;
            }
            if (!read_row_by_id(row.succ_kv_id, &row)) {
                return false;
            }
        }
        close_blob();
        kv_id_ = root_id;
        active_root_kv_id_ = root_id;
        return true;
    }

    bool transfer_cache_layer(T* k_store, T* v_store,
                              int layer, bool directionM2D = true) {
        if (directionM2D) {
            if (!write_layer_k(layer,k_store)) return false;
            if (!write_layer_v(layer,v_store)) return false;
            return true;
        }
        int root_id = active_root_kv_id_ >= 0 ? active_root_kv_id_ : kv_id_;
        if (root_id < 0) {
            return false;
        }
        int full_len = get_root_prompt_len(root_id);
        RowInfo row;
        if (!read_row_by_id(root_id, &row)) {
            return false;
        }
        while (true) {
            if (!open_blob_for_row(row.kv_id, row.prompt_len)) {
                return false;
            }
            if (!read_chunk_cache_layer(
                    k_store,
                    v_store,
                    layer,
                    full_len,
                    row.chunk_start,
                    row.prompt_len)) {
                return false;
            }
            if (row.succ_kv_id < 0) {
                break;
            }
            if (!read_row_by_id(row.succ_kv_id, &row)) {
                return false;
            }
        }
        close_blob();
        kv_id_ = root_id;
        active_root_kv_id_ = root_id;
        return true;
    }

    bool transfer_cache(
        const std::vector<T*>& k_store,
        const std::vector<T*>& v_store,
        int row_id,
        int ori_start_pos,
        int cur_start_pos,
        int matched_len,
        int dst_buffer_seq_dim) {
        if (matched_len <= 0) {
            return true;
        }
        for (int layer = 0; layer < meta_.num_layers; ++layer) {
            if (!transfer_cache_layer(
                    k_store[layer],
                    v_store[layer],
                    layer,
                    row_id,
                    ori_start_pos,
                    cur_start_pos,
                    matched_len,
                    dst_buffer_seq_dim)) {
                return false;
            }
        }
        return true;
    }

    bool transfer_cache_layer(
        T* k_store,
        T* v_store,
        int layer,
        int row_id,
        int ori_start_pos,
        int cur_start_pos,
        int matched_len,
        int dst_buffer_seq_dim) {
        if (matched_len <= 0) {
            return true;
        }
        RowInfo seed_row;
        if (!read_row_by_id(row_id, &seed_row)) {
            return false;
        }
        int root_id = seed_row.root_kv_id >= 0 ? seed_row.root_kv_id : seed_row.kv_id;
        int full_len = get_root_prompt_len(root_id);
        RowInfo row;
        if (!read_row_by_id(root_id, &row)) {
            return false;
        }

        const int src_begin = ori_start_pos;
        const int src_end = ori_start_pos + matched_len;
        while (true) {
            const int row_begin = row.chunk_start;
            const int row_end = row.chunk_start + row.prompt_len;
            const int overlap_begin = std::max(src_begin, row_begin);
            const int overlap_end = std::min(src_end, row_end);
            if (overlap_begin < overlap_end) {
                const int copy_len = overlap_end - overlap_begin;
                const int src_offset_in_chunk = overlap_begin - row_begin;
                const int dst_start = cur_start_pos + (overlap_begin - src_begin);
                if (!open_blob_for_row(row.kv_id, row.prompt_len)) {
                    return false;
                }
                if (!read_chunk_cache_layer_slice(
                        k_store,
                        v_store,
                        layer,
                        dst_buffer_seq_dim,
                        src_offset_in_chunk,
                        dst_start,
                        row.prompt_len,
                        copy_len)) {
                    return false;
                }
            }
            if (row.succ_kv_id < 0 || overlap_end >= src_end) {
                break;
            }
            if (!read_row_by_id(row.succ_kv_id, &row)) {
                return false;
            }
        }
        close_blob();
        kv_id_ = root_id;
        active_root_kv_id_ = root_id;
        return true;
    }

    bool whole_matching(const std::vector<uint64_t>& prompt_tokens) {
        std::vector<RowInfo> rows = read_all_rows();
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].prev_kv_id >= 0) {
                continue;
            }
            if (reconstruct_chain_prompt_tokens(rows[i].kv_id) == prompt_tokens) {
                kv_id_ = rows[i].kv_id;
                active_root_kv_id_ = rows[i].kv_id;
                return true;
            }
        }
        return false;
    }

    BuildInputResult build_input(
        const std::vector<uint64_t>& prompt_tokens,
        int64_t cur_pos,
        int minimum_hit_thres = 16,
        int shared_reuse_budget = kKVStoreChunkLimit,
        int prompt_budget = kKVStoreChunkLimit,
        double d = 0.2,
        double reuse_ratio = 0.25) {
        BuildInputResult result;
        std::vector<RowInfo> rows = read_all_rows();
        for (size_t i = 0; i < rows.size(); ++i) {
            const RowInfo& row = rows[i];
            if (cur_pos == 0 && row.ori_pos == 0 && row.chunk_start == 0) {
                int matched_len = 0;
                int prompt_start = 0;
                RowInfo cur_row = row;
                while (true) {
                    int delta = prefix_match_len(prompt_tokens, cur_row.prompt_tokens, prompt_start);
                    matched_len += delta;
                    prompt_start += delta;
                    if (delta != cur_row.prompt_len || cur_row.succ_kv_id < 0 ||
                        prompt_start >= static_cast<int>(prompt_tokens.size())) {
                        break;
                    }
                    if (!read_row_by_id(cur_row.succ_kv_id, &cur_row)) {
                        break;
                    }
                }
                if (matched_len >= minimum_hit_thres) {
                    BuildInputMatch match;
                    match.row_id = row.kv_id;
                    match.cur_start_pos = 0;
                    match.ori_start_pos = 0;
                    match.matched_len = matched_len;
                    match.type = ChunkPromptType::kPrefix;
                    result.raw_matches.push_back(match);
                }
            }

            BuildInputMatch lcs = longest_common_substring_match(prompt_tokens, row);
            if (lcs.matched_len >= minimum_hit_thres) {
                result.raw_matches.push_back(lcs);
            }
        }

        int n = static_cast<int>(prompt_tokens.size());
        std::vector<int> chosen(n, -1);
        for (int i = 0; i < static_cast<int>(result.raw_matches.size()); ++i) {
            const BuildInputMatch& match = result.raw_matches[i];
            for (int pos = match.cur_start_pos;
                 pos < match.cur_start_pos + match.matched_len && pos < n;
                 ++pos) {
                if (chosen[pos] < 0 ||
                    prefer_match(match, result.raw_matches[chosen[pos]])) {
                    chosen[pos] = i;
                }
            }
        }

        int pos = 0;
        while (pos < n) {
            if (chosen[pos] < 0) {
                int start = pos;
                while (pos < n && chosen[pos] < 0) {
                    ++pos;
                }
                BuildInputMatch match;
                match.row_id = -1;
                match.cur_start_pos = start;
                match.ori_start_pos = -1;
                match.matched_len = pos - start;
                match.type = ChunkPromptType::kNew;
                result.segments.push_back(match);
                continue;
            }
            const BuildInputMatch& base = result.raw_matches[chosen[pos]];
            int start = pos;
            while (pos < n && chosen[pos] == chosen[start]) {
                ++pos;
            }
            BuildInputMatch seg = base;
            seg.cur_start_pos = start;
            seg.matched_len = pos - start;
            if (seg.ori_start_pos >= 0) {
                seg.ori_start_pos += start - base.cur_start_pos;
            }
            result.segments.push_back(seg);
        }

        int prefix_len = 0;
        if (!result.segments.empty() &&
            result.segments[0].type == ChunkPromptType::kPrefix &&
            result.segments[0].cur_start_pos == 0) {
            prefix_len = result.segments[0].matched_len;
        }

        std::vector<ChunkMergeSpan> spans;
        for (size_t i = 0; i < result.segments.size(); ++i) {
            const BuildInputMatch& seg = result.segments[i];
            if (seg.cur_start_pos + seg.matched_len <= prefix_len) {
                continue;
            }
            ChunkMergeSpan span;
            span.start = seg.cur_start_pos - prefix_len;
            span.end = seg.cur_start_pos + seg.matched_len - 1 - prefix_len;
            span.type = seg.type == ChunkPromptType::kNew
                ? ChunkPromptType::kNew
                : ChunkPromptType::kNonPrefix;
            spans.push_back(span);
        }
        if (!spans.empty()) {
            result.merge_result = chunk_merge(
                spans,
                shared_reuse_budget,
                prompt_budget,
                d,
                reuse_ratio);
        }

        if (prefix_len > 0) {
            ChunkRecipe prefix_recipe;
            prefix_recipe.start = 0;
            prefix_recipe.end = prefix_len - 1;
            prefix_recipe.group = ChunkGraphType::kPrefixReuse;
            prefix_recipe.origins.push_back(
                {result.segments[0].row_id, 0, result.segments[0].ori_start_pos, prefix_len,
                 ChunkPromptType::kPrefix});
            result.recipes.push_back(prefix_recipe);
        }

        for (size_t i = 0; i < result.merge_result.chunks.size(); ++i) {
            ChunkRecipe recipe;
            recipe.start = result.merge_result.chunks[i].start + prefix_len;
            recipe.end = result.merge_result.chunks[i].end + prefix_len;
            recipe.group = result.merge_result.groups[i];
            for (size_t j = 0; j < result.segments.size(); ++j) {
                const BuildInputMatch& seg = result.segments[j];
                int seg_start = seg.cur_start_pos;
                int seg_end = seg.cur_start_pos + seg.matched_len - 1;
                int overlap_start = std::max(recipe.start, seg_start);
                int overlap_end = std::min(recipe.end, seg_end);
                if (overlap_start > overlap_end) {
                    continue;
                }
                RecipeOrigin origin;
                origin.row_id = seg.row_id;
                origin.cur_start_pos = overlap_start;
                origin.matched_len = overlap_end - overlap_start + 1;
                origin.type = seg.type;
                if (seg.ori_start_pos >= 0) {
                    origin.ori_start_pos =
                        seg.ori_start_pos + (overlap_start - seg.cur_start_pos);
                }
                recipe.origins.push_back(origin);
            }
            result.recipes.push_back(recipe);
        }
        for (size_t i = 0; i < result.raw_matches.size(); ++i) {
            result.raw_matches[i].cur_start_pos += static_cast<int32_t>(cur_pos);
        }
        for (size_t i = 0; i < result.segments.size(); ++i) {
            result.segments[i].cur_start_pos += static_cast<int32_t>(cur_pos);
        }
        for (size_t i = 0; i < result.merge_result.chunks.size(); ++i) {
            result.merge_result.chunks[i].start += static_cast<int32_t>(cur_pos);
            result.merge_result.chunks[i].end += static_cast<int32_t>(cur_pos);
        }
        for (size_t i = 0; i < result.recipes.size(); ++i) {
            result.recipes[i].start += static_cast<int32_t>(cur_pos);
            result.recipes[i].end += static_cast<int32_t>(cur_pos);
            for (size_t j = 0; j < result.recipes[i].origins.size(); ++j) {
                result.recipes[i].origins[j].cur_start_pos += static_cast<int32_t>(cur_pos);
            }
        }
        return result;
    }

    ~SQLiteKVStore() {
        close_fd();
    }
};

}
