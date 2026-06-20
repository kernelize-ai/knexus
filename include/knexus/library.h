#ifndef KNEXUS_LIBRARY_H
#define KNEXUS_LIBRARY_H

#include <knexus-api.h>
#include <knexus/kernel.h>
#include <knexus/object.h>
#include <knexus/info.h>

namespace knexus {

namespace detail {
class LibraryImpl;
}  // namespace detail

// System class
class Library : public Object<detail::LibraryImpl> {
 public:
  Library(detail::Impl base);
  Library(detail::Impl base, Info info);
  using Object::Object;

  Info getInfo() const;

  std::optional<Property> getProperty(nxs_int prop) const override;

  Kernel getKernel(const std::string &kernelName, Info info = Info());

  Kernels getKernels() const;
};

typedef Objects<Library> Librarys;

}  // namespace knexus

#endif  // KNEXUS_LIBRARY_H