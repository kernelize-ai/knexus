
#ifndef _KNEXUS_SYSTEM_IMPL_H
#define _KNEXUS_SYSTEM_IMPL_H

#include <knexus/buffer.h>
#include <knexus/info.h>
#include <knexus/runtime.h>

#include <unordered_map>

namespace knexus {
namespace detail {

class SystemImpl : public detail::Impl {
 public:
  SystemImpl(int);

  void releaseChildren() override;
  std::optional<Property> getProperty(nxs_int) const { return std::nullopt; }

  Runtime getRuntime(int idx) const { return runtimes.get(idx); }
  Runtime getRuntime(const std::string &name) { 
    auto it = runtimeMap.find(name);
    if (it != runtimeMap.end())
      return it->second;
    return Runtime();
  }
  Buffer createBuffer(const Layout &layout, const void *hostData = nullptr,
                      nxs_uint options = 0);
  Buffer copyBuffer(Buffer buf, Device dev, nxs_uint options = 0);
  Info loadCatalog(const std::string &catalogPath);

  Runtimes getRuntimes() const { return runtimes; }
  Infos getCatalogs() const { return catalogs; }
  Buffers getBuffers() const { return buffers; }

 private:
  // set of runtimes
  Runtimes runtimes;
  std::unordered_map<std::string, Runtime> runtimeMap;
  Infos catalogs;
  Buffers buffers;
};
}  // namespace detail
}  // namespace knexus

#endif  // _KNEXUS_SYSTEM_IMPL_H
