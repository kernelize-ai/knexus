#ifndef KNEXUS_SYSTEM_H
#define KNEXUS_SYSTEM_H

#include <knexus/buffer.h>
#include <knexus/info.h>
#include <knexus/runtime.h>

#include <memory>
#include <optional>
#include <vector>

namespace knexus {
namespace detail {
class SystemImpl;
}

// System class
class System : Object<detail::SystemImpl> {
 public:
  System(int);
  using Object::Object;

  std::optional<Property> getProperty(nxs_int prop) const override;

  Runtimes getRuntimes() const;
  Infos getCatalogs() const;
  Buffers getBuffers() const;

  Runtime getRuntime(int idx) const;
  Runtime getRuntime(const std::string &name);
  Buffer createBuffer(const Layout &layout, const void *hostData = nullptr,
                      nxs_uint settings = 0);
  Buffer copyBuffer(Buffer buf, Device dev, nxs_uint settings = 0);
  Info loadCatalog(const std::string &catalogPath);
};

extern System getSystem();
}  // namespace knexus

#endif  // KNEXUS_SYSTEM_H