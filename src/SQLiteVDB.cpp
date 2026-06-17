#include <SQLiteVDB.h>
#include <ops/ops.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace LMStore {
namespace {

constexpr const char* kCreateMetaTableSql =
    "CREATE TABLE IF NOT EXISTS vdb_meta("
    "id INTEGER PRIMARY KEY CHECK(id=1),"
    "distance_metric INTEGER NOT NULL,"
    "index_type INTEGER NOT NULL,"
    "scalar_dtype INTEGER NOT NULL,"
    "embed_dim INTEGER NOT NULL CHECK(embed_dim > 0));";

constexpr const char* kCreatePromptsTableSql =
    "CREATE TABLE IF NOT EXISTS vdb_prompts("
    "vec_id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "prompt TEXT NOT NULL);";

constexpr const char* kCreateVectorsTableSql =
    "CREATE TABLE IF NOT EXISTS vdb_vectors("
    "vec_id INTEGER PRIMARY KEY,"
    "vector BLOB NOT NULL,"
    "scale FLOAT DEFAULT 0.0,"
    "zero_point INTEGER DEFAULT 0,"
    "FOREIGN KEY(vec_id) REFERENCES vdb_prompts(vec_id) ON DELETE CASCADE);";

constexpr const char* kCreateKVTableSql =
    "CREATE TABLE IF NOT EXISTS vdb_kv("
    "vec_id INTEGER PRIMARY KEY,"
    "kv_cache_file TEXT NOT NULL,"
    "kv_id INTEGER NOT NULL,"
    "FOREIGN KEY(vec_id) REFERENCES vdb_prompts(vec_id) ON DELETE CASCADE);";

constexpr const char* kCreateKVVecIndexSql =
    "CREATE INDEX IF NOT EXISTS idx_vdb_kv_id ON vdb_kv(kv_id);";

bool exec_sql(sqlite3* db, const char* sql) {
    return sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK;
}

float dist(
    VDBDistanceMetric metric,
    VDBScalarDType dtype,
    const void* query,
    const void* candidate,
    int dim,
    float query_scale,
    int query_zero_point,
    float candidate_scale,
    int candidate_zero_point);

float dist_from_fp32_query(
    VDBDistanceMetric metric,
    VDBScalarDType dtype,
    const float* query,
    const void* candidate,
    int dim,
    float candidate_scale = 0.0f,
    int candidate_zero_point = 0)
{
    switch (dtype) {
    case VDBScalarDType::FP32:
        return dist(metric, dtype, query, candidate, dim, 0.0f, 0, 0.0f, 0);
    case VDBScalarDType::UINT8: {
        std::vector<std::uint8_t> quantized_query(static_cast<std::size_t>(dim), 0);
        scalar_quantize_per_tensor_u8(
            query,
            quantized_query.data(),
            static_cast<std::size_t>(dim),
            candidate_scale,
            candidate_zero_point);
        return dist(
            metric,
            dtype,
            quantized_query.data(),
            candidate,
            dim,
            candidate_scale,
            candidate_zero_point,
            candidate_scale,
            candidate_zero_point);
    }
    case VDBScalarDType::INT8: {
        std::vector<std::int8_t> quantized_query(static_cast<std::size_t>(dim), 0);
        scalar_quantize_per_tensor_i8(
            query,
            quantized_query.data(),
            static_cast<std::size_t>(dim),
            candidate_scale,
            candidate_zero_point);
        return dist(
            metric,
            dtype,
            quantized_query.data(),
            candidate,
            dim,
            candidate_scale,
            candidate_zero_point,
            candidate_scale,
            candidate_zero_point);
    }
    default:
        return 0.0f;
    }
}

float dist(
    VDBDistanceMetric metric,
    VDBScalarDType dtype,
    const void* query,
    const void* candidate,
    int dim,
    float query_scale = 0.0f,
    int query_zero_point = 0,
    float candidate_scale = 0.0f,
    int candidate_zero_point = 0)
{
    switch (dtype) {
    case VDBScalarDType::FP32:
        switch (metric) {
        case VDBDistanceMetric::L2:
            return l2_dist_fp32(static_cast<const float*>(query), static_cast<const float*>(candidate), static_cast<std::size_t>(dim));
        case VDBDistanceMetric::IP:
            return dot_product_fp32(static_cast<const float*>(query), static_cast<const float*>(candidate), static_cast<std::size_t>(dim));
        case VDBDistanceMetric::COS:
            return cosine_sim_fp32(static_cast<const float*>(query), static_cast<const float*>(candidate), static_cast<std::size_t>(dim));
        default:
            return 0.0f;
        }
    case VDBScalarDType::UINT8:
        switch (metric) {
        case VDBDistanceMetric::L2:
            return l2_dist_u8(static_cast<const std::uint8_t*>(query), static_cast<const std::uint8_t*>(candidate), static_cast<std::size_t>(dim), query_scale, query_zero_point, candidate_scale, candidate_zero_point);
        case VDBDistanceMetric::IP:
            return dot_product_u8(static_cast<const std::uint8_t*>(query), static_cast<const std::uint8_t*>(candidate), static_cast<std::size_t>(dim), query_scale, query_zero_point, candidate_scale, candidate_zero_point);
        case VDBDistanceMetric::COS:
            return cosine_sim_u8(static_cast<const std::uint8_t*>(query), static_cast<const std::uint8_t*>(candidate), static_cast<std::size_t>(dim), query_scale, query_zero_point, candidate_scale, candidate_zero_point);
        default:
            return 0.0f;
        }
    case VDBScalarDType::INT8:
        switch (metric) {
        case VDBDistanceMetric::L2:
            return l2_dist_s8(static_cast<const std::int8_t*>(query), static_cast<const std::int8_t*>(candidate), static_cast<std::size_t>(dim), query_scale, query_zero_point, candidate_scale, candidate_zero_point);
        case VDBDistanceMetric::IP:
            return dot_product_s8(static_cast<const std::int8_t*>(query), static_cast<const std::int8_t*>(candidate), static_cast<std::size_t>(dim), query_scale, query_zero_point, candidate_scale, candidate_zero_point);
        case VDBDistanceMetric::COS:
            return cosine_sim_s8(static_cast<const std::int8_t*>(query), static_cast<const std::int8_t*>(candidate), static_cast<std::size_t>(dim), query_scale, query_zero_point, candidate_scale, candidate_zero_point);
        default:
            return 0.0f;
        }
    default:
        return 0.0f;
    }
}

bool higher_is_better(VDBDistanceMetric metric) {
    return metric != VDBDistanceMetric::L2;
}

} // namespace

bool SQLiteVDB::open_write() {
    if (db_ != nullptr) {
        return true;
    }

    if (sqlite3_open(filename_.c_str(), &db_) != SQLITE_OK) {
        db_ = nullptr;
        return false;
    }

    return exec_sql(db_, "PRAGMA foreign_keys = ON;");
}

bool SQLiteVDB::open_read() {
    if (db_ != nullptr) {
        return true;
    }

    if (sqlite3_open_v2(filename_.c_str(), &db_, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        db_ = nullptr;
        return false;
    }

    return true;
}

void SQLiteVDB::close_fd() {
    if (db_ != nullptr) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool SQLiteVDB::init_tables() {
    return db_ != nullptr &&
           exec_sql(db_, kCreateMetaTableSql) &&
           exec_sql(db_, kCreatePromptsTableSql) &&
           exec_sql(db_, kCreateVectorsTableSql) &&
           exec_sql(db_, kCreateKVTableSql) &&
           exec_sql(db_, kCreateKVVecIndexSql);
}

bool SQLiteVDB::reset() {
    return db_ != nullptr &&
           exec_sql(db_, "DELETE FROM vdb_kv;") &&
           exec_sql(db_, "DELETE FROM vdb_vectors;") &&
           exec_sql(db_, "DELETE FROM vdb_prompts;") &&
           exec_sql(db_, "DELETE FROM vdb_meta;");
}

bool SQLiteVDB::write_meta(const VectorDBMeta& meta) {
    if (db_ == nullptr || meta.embed_dim <= 0) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT OR REPLACE INTO vdb_meta "
        "(id, distance_metric, index_type, scalar_dtype, embed_dim) "
        "VALUES (1, ?, ?, ?, ?);";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_int(stmt, 1, static_cast<int>(meta.distance_metric));
    sqlite3_bind_int(stmt, 2, static_cast<int>(meta.index_type));
    sqlite3_bind_int(stmt, 3, static_cast<int>(meta.scalar_dtype));
    sqlite3_bind_int(stmt, 4, meta.embed_dim);

    const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);

    if (ok) {
        meta_ = meta;
    }
    return ok;
}

bool SQLiteVDB::read_meta() {
    if (db_ == nullptr) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "SELECT distance_metric, index_type, scalar_dtype, embed_dim "
        "FROM vdb_meta WHERE id = 1;";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    const bool ok = sqlite3_step(stmt) == SQLITE_ROW;
    if (ok) {
        meta_.distance_metric = static_cast<VDBDistanceMetric>(sqlite3_column_int(stmt, 0));
        meta_.index_type = static_cast<VDBIndexType>(sqlite3_column_int(stmt, 1));
        meta_.scalar_dtype = static_cast<VDBScalarDType>(sqlite3_column_int(stmt, 2));
        meta_.embed_dim = sqlite3_column_int(stmt, 3);
    }

    sqlite3_finalize(stmt);
    return ok;
}

bool SQLiteVDB::insert_prompt(const std::string& prompt, int* vec_id) {
    if (db_ == nullptr || vec_id == nullptr) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "INSERT INTO vdb_prompts(prompt) VALUES (?);";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_text(stmt, 1, prompt.c_str(), -1, SQLITE_TRANSIENT);
    const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);

    if (ok) {
        *vec_id = static_cast<int>(sqlite3_last_insert_rowid(db_));
    }
    return ok;
}

bool SQLiteVDB::insert_vector(int vec_id, const void* vector_blob, int num_bytes, float scale, int zero_point) {
    if (db_ == nullptr || vector_blob == nullptr || num_bytes <= 0) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT INTO vdb_vectors(vec_id, vector, scale, zero_point) "
        "VALUES (?, ?, ?, ?);";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_int(stmt, 1, vec_id);
    sqlite3_bind_blob(stmt, 2, vector_blob, num_bytes, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 3, static_cast<double>(scale));
    sqlite3_bind_int(stmt, 4, zero_point);
    const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    return ok;
}

bool SQLiteVDB::insert_kv(int vec_id, int kv_id, const std::string& kv_cache_file) {
    if (db_ == nullptr) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT INTO vdb_kv(vec_id, kv_cache_file, kv_id) "
        "VALUES (?, ?, ?);";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_int(stmt, 1, vec_id);
    sqlite3_bind_text(stmt, 2, kv_cache_file.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 3, kv_id);
    const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    return ok;
}

bool SQLiteVDB::read_prompt(int vec_id, std::string* prompt) const {
    if (db_ == nullptr || prompt == nullptr) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT prompt FROM vdb_prompts WHERE vec_id = ?;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_int(stmt, 1, vec_id);
    const bool ok = sqlite3_step(stmt) == SQLITE_ROW;
    if (ok) {
        const unsigned char* text = sqlite3_column_text(stmt, 0);
        *prompt = text == nullptr ? "" : reinterpret_cast<const char*>(text);
    }

    sqlite3_finalize(stmt);
    return ok;
}

bool SQLiteVDB::read_vector(int vec_id, std::vector<std::uint8_t>* vector_blob) const {
    if (db_ == nullptr || vector_blob == nullptr) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT vector FROM vdb_vectors WHERE vec_id = ?;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_int(stmt, 1, vec_id);
    const bool ok = sqlite3_step(stmt) == SQLITE_ROW;
    if (ok) {
        const auto* blob = static_cast<const std::uint8_t*>(sqlite3_column_blob(stmt, 0));
        const int bytes = sqlite3_column_bytes(stmt, 0);
        vector_blob->assign(blob, blob + bytes);
    }

    sqlite3_finalize(stmt);
    return ok;
}

bool SQLiteVDB::read_kv(int vec_id, int* kv_id, std::string* kv_cache_file) const {
    if (db_ == nullptr || kv_id == nullptr || kv_cache_file == nullptr) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "SELECT kv_id, kv_cache_file FROM vdb_kv "
        "WHERE vec_id = ?;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_int(stmt, 1, vec_id);
    const bool ok = sqlite3_step(stmt) == SQLITE_ROW;
    if (ok) {
        *kv_id = sqlite3_column_int(stmt, 0);
        const unsigned char* text = sqlite3_column_text(stmt, 1);
        *kv_cache_file = text == nullptr ? "" : reinterpret_cast<const char*>(text);
    }

    sqlite3_finalize(stmt);
    return ok;
}

bool SQLiteVDB::update_prompt(int vec_id, const std::string& prompt) {
    if (db_ == nullptr) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "UPDATE vdb_prompts SET prompt = ? WHERE vec_id = ?;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_text(stmt, 1, prompt.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, vec_id);
    const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    return ok;
}

bool SQLiteVDB::update_vector(int vec_id, const void* vector_blob, int num_bytes, float scale, int zero_point) {
    if (db_ == nullptr || vector_blob == nullptr || num_bytes <= 0) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "UPDATE vdb_vectors SET vector = ?, scale = ?, zero_point = ? "
        "WHERE vec_id = ?;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_blob(stmt, 1, vector_blob, num_bytes, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 2, static_cast<double>(scale));
    sqlite3_bind_int(stmt, 3, zero_point);
    sqlite3_bind_int(stmt, 4, vec_id);
    const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    return ok;
}

bool SQLiteVDB::update_kv(int vec_id, int kv_id, const std::string& kv_cache_file) {
    if (db_ == nullptr) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "UPDATE vdb_kv SET kv_cache_file = ?, kv_id = ? "
        "WHERE vec_id = ?;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_text(stmt, 1, kv_cache_file.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, kv_id);
    sqlite3_bind_int(stmt, 3, vec_id);
    const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    return ok;
}

bool SQLiteVDB::delete_entry(int vec_id) {
    if (db_ == nullptr) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "DELETE FROM vdb_prompts WHERE vec_id = ?;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_int(stmt, 1, vec_id);
    const bool ok = sqlite3_step(stmt) == SQLITE_DONE;
    sqlite3_finalize(stmt);
    return ok;
}

bool SQLiteVDB::list_vec_ids(std::vector<int>* vec_ids) const {
    if (db_ == nullptr || vec_ids == nullptr) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT vec_id FROM vdb_prompts ORDER BY vec_id ASC;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    vec_ids->clear();
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        vec_ids->push_back(sqlite3_column_int(stmt, 0));
    }

    sqlite3_finalize(stmt);
    return true;
}

bool SQLiteVDB::count(int* num_vectors) const {
    if (db_ == nullptr || num_vectors == nullptr) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT COUNT(*) FROM vdb_vectors;";
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    const bool ok = sqlite3_step(stmt) == SQLITE_ROW;
    if (ok) {
        *num_vectors = sqlite3_column_int(stmt, 0);
    }

    sqlite3_finalize(stmt);
    return ok;
}

bool SQLiteVDB::search(
    const float* query,
    int top_k,
    std::vector<VDBSearchResult>* results,
    bool include_payload) const
{
    if (db_ == nullptr || query == nullptr || results == nullptr || top_k <= 0 || meta_.embed_dim <= 0) {
        return false;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql = include_payload
        ? "SELECT p.vec_id, p.prompt, v.vector, v.scale, v.zero_point, "
          "k.kv_id, k.kv_cache_file "
          "FROM vdb_prompts AS p "
          "JOIN vdb_vectors AS v ON p.vec_id = v.vec_id "
          "LEFT JOIN vdb_kv AS k ON p.vec_id = k.vec_id;"
        : "SELECT p.vec_id, v.vector, v.scale, v.zero_point "
          "FROM vdb_prompts AS p "
          "JOIN vdb_vectors AS v ON p.vec_id = v.vec_id;";

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    results->clear();
    const bool descending = higher_is_better(meta_.distance_metric);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const int vec_id = sqlite3_column_int(stmt, 0);
        const int vector_col = include_payload ? 2 : 1;
        const int scale_col = include_payload ? 3 : 2;
        const int zero_point_col = include_payload ? 4 : 3;
        const void* blob = sqlite3_column_blob(stmt, vector_col);
        const float scale = static_cast<float>(sqlite3_column_double(stmt, scale_col));
        const int zero_point = sqlite3_column_int(stmt, zero_point_col);

        VDBSearchResult result;
        result.vec_id = vec_id;
        result.distance = dist_from_fp32_query(
            meta_.distance_metric,
            meta_.scalar_dtype,
            query,
            blob,
            meta_.embed_dim,
            scale,
            zero_point);
        if (include_payload) {
            const unsigned char* prompt_text = sqlite3_column_text(stmt, 1);
            result.prompt = prompt_text == nullptr ? "" : reinterpret_cast<const char*>(prompt_text);
            result.kv_id = sqlite3_column_type(stmt, 5) == SQLITE_NULL ? -1 : sqlite3_column_int(stmt, 5);

            const unsigned char* kv_path = sqlite3_column_text(stmt, 6);
            result.kv_cache_file = kv_path == nullptr ? "" : reinterpret_cast<const char*>(kv_path);
        }
        results->push_back(std::move(result));
    }

    sqlite3_finalize(stmt);

    std::sort(results->begin(), results->end(), [descending](const VDBSearchResult& lhs, const VDBSearchResult& rhs) {
        if (descending) {
            return lhs.distance > rhs.distance;
        }
        return lhs.distance < rhs.distance;
    });

    if (static_cast<int>(results->size()) > top_k) {
        results->resize(static_cast<std::size_t>(top_k));
    }
    return true;
}

} // namespace LMStore
