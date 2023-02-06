#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/_symeig_helper_ops.h>

namespace at {


// aten::_symeig_helper(Tensor self, bool eigenvectors, bool upper) -> (Tensor, Tensor)
inline ::std::tuple<at::Tensor,at::Tensor> _symeig_helper(const at::Tensor & self, bool eigenvectors, bool upper) {
    return at::_ops::_symeig_helper::call(self, eigenvectors, upper);
}

// aten::_symeig_helper.out(Tensor self, bool eigenvectors, bool upper, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
inline ::std::tuple<at::Tensor &,at::Tensor &> _symeig_helper_out(at::Tensor & out0, at::Tensor & out1, const at::Tensor & self, bool eigenvectors, bool upper) {
    return at::_ops::_symeig_helper_out::call(self, eigenvectors, upper, out0, out1);
}
// aten::_symeig_helper.out(Tensor self, bool eigenvectors, bool upper, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
inline ::std::tuple<at::Tensor &,at::Tensor &> _symeig_helper_outf(const at::Tensor & self, bool eigenvectors, bool upper, at::Tensor & out0, at::Tensor & out1) {
    return at::_ops::_symeig_helper_out::call(self, eigenvectors, upper, out0, out1);
}

}
