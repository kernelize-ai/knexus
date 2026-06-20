#ifndef KNEXUS_KERNEL_H
#define KNEXUS_KERNEL_H

#include <knexus-api.h>
#include <knexus/object.h>
#include <knexus/info.h>

namespace knexus {

namespace detail {
class KernelImpl;
}  // namespace detail

// System class
class Kernel : public Object<detail::KernelImpl> {
 public:
  Kernel(detail::Impl base, const std::string &kernelName, Info info = Info());
  using Object::Object;

  Info getInfo() const;

  std::optional<Property> getProperty(nxs_int prop) const override;
};

typedef Objects<Kernel> Kernels;

}  // namespace knexus

#endif  // KNEXUS_KERNEL_H