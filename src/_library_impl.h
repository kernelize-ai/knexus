#ifndef _KNEXUS_LIBRARY_IMPL_H
#define _KNEXUS_LIBRARY_IMPL_H

#include <knexus/info.h>
#include <knexus/kernel.h>
#include <knexus/library.h>

#include <unordered_map>

namespace knexus {
namespace detail {

class LibraryImpl : public Impl {
 public:
  /// @brief Construct a Platform for the current system
  LibraryImpl(Impl base);
  LibraryImpl(Impl base, Info info);

  void releaseChildren() override;
  nxs_status releaseAPI() override;

  std::optional<Property> getProperty(nxs_int prop) const;

  Info getInfo() const { return info; }

  Kernel getKernel(const std::string &kernelName, Info info);

  Kernels getKernels() const { return kernels; }

 private:
  Kernels kernels;
  std::unordered_map<std::string, Kernel> kernelMap;
  Info info;
};
}  // namespace detail
}  // namespace knexus

#endif  // _KNEXUS_LIBRARY_IMPL_H