#ifndef _KNEXUS_KERNEL_IMPL_H
#define _KNEXUS_KERNEL_IMPL_H

#include <knexus/info.h>
#include <knexus/kernel.h>

namespace knexus {
namespace detail {

class KernelImpl : public Impl {
 public:
  /// @brief Construct a Platform for the current system
  KernelImpl(Impl base, const std::string &kName, Info info);

  std::optional<Property> getProperty(nxs_int prop) const;

  Info getInfo() const { return info; }

 private:
  std::string kernelName;
  Info info;
};
}  // namespace detail
}  // namespace knexus

#endif  // _KNEXUS_KERNEL_IMPL_H