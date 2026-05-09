#ifdef NEED_PYBIND

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ops/ops.h"
#include "SQLiteVDB.h"
#include <stdexcept>
#include <vector>
#include <string>
#include <torch/extension.h>

namespace py = pybind11;
using namespace LMStore;

template struct pybind11::detail::type_caster<at::Tensor>;

torch::Tensor apply_rope_emb_torch(
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin
) {
    auto B = k.size(1); // num_head
    auto L = k.size(2); // seq_len
    auto D = k.size(3); // head_size

    auto result = torch::empty_like(k);

    apply_rope_emb(
        k.data_ptr<float>(),
        result.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        B, L, D
    );

    return result;
}

torch::Tensor hadamard_torch(
    torch::Tensor data,
    int dim, float scale, bool transpose
) {
    // data: [1, B, L, D]
    auto B = data.size(1); // num_head
    auto L = data.size(2); // seq_len
    auto D = data.size(3); // head_size
    torch::Tensor result;
    // data: [1, B, D, L], if transpose
    if (transpose) {
        L = data.size(3); 
        D = data.size(2);
        result = torch::empty(
            {data.size(0), B, L, D},
            data.options()
        );
    } else {
        result = torch::empty_like(data);
    }

    fwht_tensor(
        data.data_ptr<float>(),
        result.data_ptr<float>(),
        B, L, D, scale, transpose
    );

    return result;
}

torch::Tensor transpose_torch(
    torch::Tensor data
) {
    int dim1 = data.size(1), dim2 = data.size(2), dim3 = data.size(3);
    auto result = torch::empty(
        {data.size(0), dim1, dim3, dim2},
        data.options()
    );

    transpose_tensor(
        data.data_ptr<float>(),
        result.data_ptr<float>(),
        dim1, dim2, dim3
    );

    return result;
}

std::tuple<torch::Tensor, torch::Tensor> compute_rerotation_cos_sin_torch(
    torch::Tensor freqs_cos, // [L, half]
    torch::Tensor freqs_sin, // [L, half]
    int64_t ori_pos,
    int64_t new_pos,
    int64_t matched_len
) {
    auto L = freqs_cos.size(0); // seq_len
    auto half = freqs_cos.size(1); // half
    auto rerotation_cos = torch::empty_like(freqs_cos);
    auto rerotation_sin = torch::empty_like(freqs_sin);

    compute_rerotation_cos_sin(
        freqs_cos.data_ptr<float>(),
        freqs_sin.data_ptr<float>(),
        rerotation_cos.data_ptr<float>(),
        rerotation_sin.data_ptr<float>(),
        L, half, ori_pos, new_pos, matched_len
    );

    return std::make_tuple(rerotation_cos, rerotation_sin);
}

torch::Tensor rerotate_k_fp32_torch(
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    bool enable_r3
) {
    // k: [1, B, D, L], output: [1, B, D, L]
    auto B = k.size(1); // num_head
    auto L = k.size(3); // seq_len
    auto D = k.size(2); // head_size

    auto output = torch::empty_like(k);

    rerotate_k_fp32(
        k.data_ptr<float>(),
        output.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        B, L, D, enable_r3
    );

    return output;
}

PYBIND11_MODULE(pylmstore, m) {
    m.doc() = "MobileLMStore Python binding";

    py::enum_<VDBDistanceMetric>(m, "VDBDistanceMetric")
        .value("L2", VDBDistanceMetric::L2)
        .value("IP", VDBDistanceMetric::IP)
        .value("COS", VDBDistanceMetric::COS);

    py::enum_<VDBIndexType>(m, "VDBIndexType")
        .value("FLAT", VDBIndexType::FLAT)
        .value("IVF", VDBIndexType::IVF)
        .value("IVF_PQ", VDBIndexType::IVF_PQ)
        .value("HNSW", VDBIndexType::HNSW);

    py::enum_<VDBScalarDType>(m, "VDBScalarDType")
        .value("FP32", VDBScalarDType::FP32)
        .value("FP16", VDBScalarDType::FP16)
        .value("BF16", VDBScalarDType::BF16)
        .value("INT8", VDBScalarDType::INT8)
        .value("UINT8", VDBScalarDType::UINT8)
        .value("INT4", VDBScalarDType::INT4)
        .value("UINT4", VDBScalarDType::UINT4);

    py::class_<VectorDBMeta>(m, "VectorDBMeta")
        .def(py::init<>())
        .def(py::init<VDBDistanceMetric, VDBIndexType, VDBScalarDType, int>(),
             py::arg("distance_metric"),
             py::arg("index_type"),
             py::arg("scalar_dtype"),
             py::arg("embed_dim"))
        .def_readwrite("distance_metric", &VectorDBMeta::distance_metric)
        .def_readwrite("index_type", &VectorDBMeta::index_type)
        .def_readwrite("scalar_dtype", &VectorDBMeta::scalar_dtype)
        .def_readwrite("embed_dim", &VectorDBMeta::embed_dim);

    py::class_<VDBSearchResult>(m, "VDBSearchResult")
        .def(py::init<>())
        .def_readwrite("vec_id", &VDBSearchResult::vec_id)
        .def_readwrite("distance", &VDBSearchResult::distance)
        .def_readwrite("prompt", &VDBSearchResult::prompt)
        .def_readwrite("kv_cache_file", &VDBSearchResult::kv_cache_file)
        .def_readwrite("kv_id", &VDBSearchResult::kv_id);

    py::class_<SQLiteVDB>(m, "SQLiteVDB")
        .def(py::init<const std::string&>(), py::arg("filename"))
        .def("open_write", &SQLiteVDB::open_write)
        .def("open_read", &SQLiteVDB::open_read)
        .def("close_fd", &SQLiteVDB::close_fd)
        .def("init_tables", &SQLiteVDB::init_tables)
        .def("reset", &SQLiteVDB::reset)
        .def("write_meta", &SQLiteVDB::write_meta, py::arg("meta"))
        .def("read_meta", &SQLiteVDB::read_meta)
        .def("get_meta", &SQLiteVDB::get_meta, py::return_value_policy::copy)
        .def("insert_prompt", [](SQLiteVDB& db, const std::string& prompt) {
            int vec_id = -1;
            if (!db.insert_prompt(prompt, &vec_id)) {
                throw std::runtime_error("insert_prompt failed");
            }
            return vec_id;
        }, py::arg("prompt"))
        .def("insert_vector", [](SQLiteVDB& db, int vec_id, py::array vector, float scale, int zero_point) {
            auto info = vector.request();
            if (info.ndim != 1) {
                throw std::runtime_error("insert_vector expects a 1D array");
            }
            if (!db.insert_vector(vec_id, info.ptr, static_cast<int>(info.size * info.itemsize), scale, zero_point)) {
                throw std::runtime_error("insert_vector failed");
            }
        }, py::arg("vec_id"), py::arg("vector"), py::arg("scale") = 0.0f, py::arg("zero_point") = 0)
        .def("insert_kv", [](SQLiteVDB& db, int vec_id, int kv_id, const std::string& kv_cache_file) {
            if (!db.insert_kv(vec_id, kv_id, kv_cache_file)) {
                throw std::runtime_error("insert_kv failed");
            }
        }, py::arg("vec_id"), py::arg("kv_id"), py::arg("kv_cache_file"))
        .def("read_prompt", [](const SQLiteVDB& db, int vec_id) {
            std::string prompt;
            if (!db.read_prompt(vec_id, &prompt)) {
                throw std::runtime_error("read_prompt failed");
            }
            return prompt;
        }, py::arg("vec_id"))
        .def("read_kv", [](const SQLiteVDB& db, int vec_id) {
            int kv_id = -1;
            std::string kv_cache_file;
            if (!db.read_kv(vec_id, &kv_id, &kv_cache_file)) {
                throw std::runtime_error("read_kv failed");
            }
            return py::make_tuple(kv_id, kv_cache_file);
        }, py::arg("vec_id"))
        .def("update_prompt", &SQLiteVDB::update_prompt, py::arg("vec_id"), py::arg("prompt"))
        .def("update_kv", &SQLiteVDB::update_kv, py::arg("vec_id"), py::arg("kv_id"), py::arg("kv_cache_file"))
        .def("delete_entry", &SQLiteVDB::delete_entry, py::arg("vec_id"))
        .def("list_vec_ids", [](const SQLiteVDB& db) {
            std::vector<int> vec_ids;
            if (!db.list_vec_ids(&vec_ids)) {
                throw std::runtime_error("list_vec_ids failed");
            }
            return vec_ids;
        })
        .def("count", [](const SQLiteVDB& db) {
            int num_vectors = 0;
            if (!db.count(&num_vectors)) {
                throw std::runtime_error("count failed");
            }
            return num_vectors;
        })
        .def("search", [](const SQLiteVDB& db, py::array_t<float, py::array::c_style | py::array::forcecast> query_vector, int top_k, bool include_payload) {
            auto info = query_vector.request();
            if (info.ndim != 1) {
                throw std::runtime_error("search expects a 1D float32 query vector");
            }
            std::vector<VDBSearchResult> results;
            if (!db.search(static_cast<const float*>(info.ptr), top_k, &results, include_payload)) {
                throw std::runtime_error("search failed");
            }
            return results;
        }, py::arg("query_vector"), py::arg("top_k"), py::arg("include_payload") = true);

    m.def("apply_rope_emb_cpp", &apply_rope_emb_torch,
          "Apply RoPE embedding using C++ implementation",
          py::arg("k"), py::arg("cos"), py::arg("sin"));
    m.def("hadamard_cpp", &hadamard_torch,
          "Apply Hadamard Transform using C++ implementation",
          py::arg("data"), py::arg("dim"), py::arg("scale"), py::arg("transpose")=false);
    m.def("transpose_cpp", &transpose_torch,
          "Transpose last 2 dim using C++ implementation",
          py::arg("data"));
    m.def("rerotate_k_fp32_cpp", &rerotate_k_fp32_torch,
          "Rerotate k using C++ implementation",
          py::arg("k"), py::arg("cos"), py::arg("sin"), 
          py::arg("enable_r3")=false);
    m.def("compute_rerotation_cos_sin_cpp", &compute_rerotation_cos_sin_torch,
          "Compute rerotation cos and sin using C++ implementation",
          py::arg("freqs_cos"), py::arg("freqs_sin"),
          py::arg("ori_pos"), py::arg("new_pos"), py::arg("matched_len"));
}

#endif // NEED_PYBIND
