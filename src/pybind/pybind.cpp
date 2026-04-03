#ifdef NEED_PYBIND

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ops/ops.h"
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