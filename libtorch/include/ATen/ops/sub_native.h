#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>
#include <ATen/ops/sub_meta.h>

namespace at {
namespace native {
struct TORCH_API structured_sub_out : public at::meta::structured_sub_Tensor {
void impl(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out);
};
TORCH_API at::Tensor sub_sparse(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1);
TORCH_API at::Tensor & sub_out_sparse(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out);
TORCH_API at::Tensor & sub_sparse_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1);
struct TORCH_API structured_sub_out_mps : public at::meta::structured_sub_Tensor {
void impl(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out);
};
TORCH_API at::Tensor sub_zerotensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1);
TORCH_API at::Tensor sub(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha=1);
TORCH_API at::Tensor & sub_Scalar_out(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out);
TORCH_API at::Tensor & sub_(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha=1);
} // namespace native
} // namespace at
