CREATE TABLE IF NOT EXISTS kv_meta(
            id INTEGER PRIMARY KEY CHECK(id=1),
            num_layers INTEGER,
            num_heads INTEGER,
            head_dim INTEGER,
            seq_dim INTEGER,
            valid_seq_len INTEGER);

CREATE TABLE IF NOT EXISTS kv_tokens(
            kv_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_tokens BLOB,
            next_token INTEGER);

CREATE TABLE IF NOT EXISTS kv_caches(
            kv_id INTEGER PRIMARY KEY,
            prompt TEXT,
            kv_blob BLOB,
            FOREIGN KEY(kv_id) REFERENCES kv_tokens(kv_id));