#ifndef ROPE_CONFIG_PARSER_H
#define ROPE_CONFIG_PARSER_H

#include <fstream>
#include <sstream>
#include <string>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

namespace LMStore {

struct rope_config_params {
    int dim = 0;
    int n_heads = 0;
    int head_dim = 0;
    float rope_theta = 10000.0f;
    bool use_scaled_rope = false;
    bool use_hf_rope = false;
    float partial_rotary_factor = 1.0f;
    int rope_scale_factor = 8;
    int high_freq_factor = 4;
};

namespace detail {

inline bool read_text_file(const std::string& path, std::string* out_text) {
    if (out_text == nullptr) {
        return false;
    }
    std::ifstream ifs(path);
    if (!ifs) {
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    *out_text = oss.str();
    return true;
}

inline void set_error(std::string* err, const std::string& msg) {
    if (err != nullptr) {
        *err = msg;
    }
}

inline bool read_int_field(const rapidjson::Value& root, const char* key, int* out_value) {
    if (out_value == nullptr || !root.HasMember(key)) {
        return false;
    }
    const auto& v = root[key];
    if (v.IsInt()) {
        *out_value = v.GetInt();
        return true;
    }
    if (v.IsUint()) {
        *out_value = static_cast<int>(v.GetUint());
        return true;
    }
    return false;
}

inline bool read_float_field(const rapidjson::Value& root, const char* key, float* out_value) {
    if (out_value == nullptr || !root.HasMember(key)) {
        return false;
    }
    const auto& v = root[key];
    if (!v.IsNumber()) {
        return false;
    }
    *out_value = static_cast<float>(v.GetDouble());
    return true;
}

inline bool read_bool_field(const rapidjson::Value& root, const char* key, bool* out_value) {
    if (out_value == nullptr || !root.HasMember(key)) {
        return false;
    }
    const auto& v = root[key];
    if (!v.IsBool()) {
        return false;
    }
    *out_value = v.GetBool();
    return true;
}

} // namespace detail

inline bool parse_rope_config_json(
    const std::string& json_text,
    rope_config_params* out_params,
    std::string* err = nullptr
) {
    if (out_params == nullptr) {
        detail::set_error(err, "out_params is null");
        return false;
    }

    rapidjson::Document doc;
    doc.Parse(json_text.c_str());
    if (doc.HasParseError()) {
        detail::set_error(
            err,
            std::string("json parse error: ") +
                rapidjson::GetParseError_En(doc.GetParseError()) +
                " at offset " + std::to_string(doc.GetErrorOffset())
        );
        return false;
    }
    if (!doc.IsObject()) {
        detail::set_error(err, "json root must be object");
        return false;
    }

    rope_config_params cfg;

    detail::read_int_field(doc, "dim", &cfg.dim);
    detail::read_int_field(doc, "n_heads", &cfg.n_heads);
    if (!detail::read_int_field(doc, "head_dim", &cfg.head_dim)) {
        if (cfg.dim > 0 && cfg.n_heads > 0 && (cfg.dim % cfg.n_heads == 0)) {
            cfg.head_dim = cfg.dim / cfg.n_heads;
        } else {
            detail::set_error(err, "missing valid head_dim (or inferable dim/n_heads)");
            return false;
        }
    }

    if (!detail::read_float_field(doc, "rope_theta", &cfg.rope_theta)) {
        if (!detail::read_float_field(doc, "rope_freq_base", &cfg.rope_theta)) {
            detail::set_error(err, "missing rope_theta/rope_freq_base");
            return false;
        }
    }

    detail::read_bool_field(doc, "use_scaled_rope", &cfg.use_scaled_rope);
    detail::read_bool_field(doc, "use_hf_rope", &cfg.use_hf_rope);
    detail::read_float_field(doc, "partial_rotary_factor", &cfg.partial_rotary_factor);
    detail::read_int_field(doc, "rope_scale_factor", &cfg.rope_scale_factor);
    detail::read_int_field(doc, "high_freq_factor", &cfg.high_freq_factor);

    if (cfg.head_dim <= 0) {
        detail::set_error(err, "head_dim must be > 0");
        return false;
    }
    if (cfg.rope_theta <= 0.0f) {
        detail::set_error(err, "rope_theta must be > 0");
        return false;
    }
    if (cfg.partial_rotary_factor <= 0.0f || cfg.partial_rotary_factor > 1.0f) {
        detail::set_error(err, "partial_rotary_factor must be in (0, 1]");
        return false;
    }
    if (cfg.rope_scale_factor <= 0) {
        detail::set_error(err, "rope_scale_factor must be > 0");
        return false;
    }
    if (cfg.high_freq_factor <= 1) {
        detail::set_error(err, "high_freq_factor must be > 1");
        return false;
    }

    *out_params = cfg;
    return true;
}

inline bool parse_rope_config_json_file(
    const std::string& json_path,
    rope_config_params* out_params,
    std::string* err = nullptr
) {
    std::string json_text;
    if (!detail::read_text_file(json_path, &json_text)) {
        detail::set_error(err, "failed to read json file: " + json_path);
        return false;
    }
    return parse_rope_config_json(json_text, out_params, err);
}

} // namespace LMStore

#endif // ROPE_CONFIG_PARSER_H
