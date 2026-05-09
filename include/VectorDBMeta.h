#pragma once

#include <cstdint>

namespace LMStore {

enum class VDBDistanceMetric : std::uint8_t {
    L2 = 0,
    IP = 1,
    COS = 2,
};

enum class VDBIndexType : std::uint8_t {
    FLAT = 0,
    IVF = 1,
    IVF_PQ = 2,
    HNSW = 3,
};

enum class VDBScalarDType : std::uint8_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    UINT8 = 4,
    INT4 = 5,
    UINT4 = 6
};

struct VectorDBMeta {
    VDBDistanceMetric distance_metric = VDBDistanceMetric::COS;
    VDBIndexType index_type = VDBIndexType::FLAT;
    VDBScalarDType scalar_dtype = VDBScalarDType::FP32;
    int embed_dim = 0;

    VectorDBMeta() {}
    VectorDBMeta(
        VDBDistanceMetric metric,
        VDBIndexType type,
        VDBScalarDType dtype,
        int dim)
        : distance_metric(metric),
          index_type(type),
          scalar_dtype(dtype),
          embed_dim(dim) {}
};

} // namespace LMStore
