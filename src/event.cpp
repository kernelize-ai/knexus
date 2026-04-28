#define NEXUS_LOG_MODULE "event"

#include <nexus/event.h>
#include <nexus/log.h>

#include "_runtime_impl.h"

using namespace nexus;

namespace nexus {
namespace detail {
class EventImpl : public Impl {
 public:
  /// @brief Construct a Platform for the current system
  EventImpl(Impl owner, nxs_int value) : Impl(owner), value(value) {
    NXSLOG_TRACE("CTOR: {}", getId());
  }

  ~EventImpl() {
    NXSLOG_TRACE("DTOR: {}", getId());
    (void)release();
  }

  nxs_status releaseAPI() override {
    auto *rt = getParentOfType<RuntimeImpl>();
    if (!rt) return NXS_InvalidObject;
    return (nxs_status)rt->runAPIFunction<NF_nxsReleaseEvent>(getId());
  }
  void releaseChild(Impl *) override {}

  std::optional<Property> getProperty(nxs_int prop) const {
    auto *rt = getParentOfType<RuntimeImpl>();
    return rt->getAPIProperty<NF_nxsGetEventProperty>(prop, getId());
  }

  nxs_status signal(nxs_int value) {
    auto *rt = getParentOfType<RuntimeImpl>();
    return (nxs_status)rt->runAPIFunction<NF_nxsSignalEvent>(getId(), value);
  }

  nxs_status wait(nxs_int value) {
    auto *rt = getParentOfType<RuntimeImpl>();
    return (nxs_status)rt->runAPIFunction<NF_nxsWaitEvent>(getId(), value);
  }

 private:
  nxs_int value;
};
}  // namespace detail
}  // namespace nexus

///////////////////////////////////////////////////////////////////////////////
Event::Event(detail::Impl base, nxs_int value) : Object(base, value) {}

std::optional<Property> Event::getProperty(nxs_int prop) const {
  NEXUS_OBJ_MCALL(std::nullopt, getProperty, prop);
}

nxs_status Event::signal(nxs_int value) {
  NEXUS_OBJ_MCALL(NXS_InvalidEvent, signal, value);
}

nxs_status Event::wait(nxs_int value) {
  NEXUS_OBJ_MCALL(NXS_InvalidEvent, wait, value);
}
