PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS vdb_meta(
            id INTEGER PRIMARY KEY CHECK(id=1),
            distance_metric INTEGER NOT NULL,
            index_type INTEGER NOT NULL,
            scalar_dtype INTEGER NOT NULL,
            embed_dim INTEGER NOT NULL CHECK(embed_dim > 0));

CREATE TABLE IF NOT EXISTS vdb_prompts(
            vec_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS vdb_vectors(
            vec_id INTEGER PRIMARY KEY,
            vector BLOB NOT NULL,
            scale FLOAT DEFAULT 0.0,
            zero_point INTEGER DEFAULT 0,
            FOREIGN KEY(vec_id) REFERENCES vdb_prompts(vec_id) ON DELETE CASCADE);

CREATE TABLE IF NOT EXISTS vdb_kv(
            vec_id INTEGER PRIMARY KEY,
            kv_cache_file TEXT NOT NULL,
            kv_id INTEGER NOT NULL,
            FOREIGN KEY(vec_id) REFERENCES vdb_prompts(vec_id) ON DELETE CASCADE);

CREATE INDEX IF NOT EXISTS idx_vdb_kv_id ON vdb_kv(kv_id);
