#define KNEXUS_LOG_MODULE "stream"
#include <knexus/log.h>

#include <knexus/stream.h>

#include "_device_impl.h"

namespace knexus {
namespace detail {

class StreamImpl : public Impl {
 public:
  /// @brief Construct a Platform for the current system
  StreamImpl(detail::Impl base) : detail::Impl(base) {
    NXSLOG_TRACE("Stream: {}", getId());
  }

  nxs_status releaseAPI() override {
    auto *rt = getParentOfType<RuntimeImpl>();
    if (!rt) return NXS_InvalidObject;
    return (nxs_status)rt->runAPIFunction<NF_nxsReleaseStream>(getId());
  }

  std::optional<Property> getProperty(nxs_int prop) const {
    auto *rt = getParentOfType<RuntimeImpl>();
    return rt->getAPIProperty<NF_nxsGetStreamProperty>(prop, getId());
  }

 private:
};
}  // namespace detail
}  // namespace knexus

using namespace knexus;
using namespace knexus::detail;

///////////////////////////////////////////////////////////////////////////////
Stream::Stream(detail::Impl base) : Object(base) {}

std::optional<Property> Stream::getProperty(nxs_int prop) const {
  KNEXUS_OBJ_MCALL(std::nullopt, getProperty, prop);
}
