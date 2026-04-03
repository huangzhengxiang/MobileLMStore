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
}

#endif // NEED_PYBIND