#ifndef KNEXUS_DEVICE_H
#define KNEXUS_DEVICE_H

#include <knexus-api.h>
#include <knexus/buffer.h>
#include <knexus/event.h>
#include <knexus/library.h>
#include <knexus/info.h>
#include <knexus/schedule.h>
#include <knexus/stream.h>

#include <optional>
#include <string>

namespace knexus {

namespace detail {
class DeviceImpl;
}  // namespace detail

// Device class
class Device : public Object<detail::DeviceImpl> {
 public:
  Device(detail::Impl base);
  using Object::Object;

  // Get Device Property Value
  std::optional<Property> getProperty(nxs_int prop) const override;

  Info getInfo() const;

  // Runtime functions
  Librarys getLibraries() const;
  Schedules getSchedules() const;
  Streams getStreams() const;
  Events getEvents() const;
  Buffers getBuffers() const;

  Stream createStream(nxs_uint settings = 0);
  Schedule createSchedule(nxs_uint settings = 0);
  Event createEvent(nxs_event_type event_type = NXS_EventType_Shared,
                    nxs_uint settings = 0);

  Library loadLibrary(Info catalog, const std::string &libraryName);
  Library createLibrary(void *libraryData, size_t librarySize,
                        nxs_uint settings = 0);
  Library createLibrary(const std::string &libraryPath, nxs_uint settings = 0);

  Buffer createBuffer(const Layout &layout, const void *data = nullptr,
                      nxs_uint settings = 0);
  Buffer copyBuffer(Buffer buf, nxs_uint settings = 0);
  
  Buffer fillBuffer(void *value, nxs_uint value_size_bytes);
};

typedef Objects<Device> Devices;

}  // namespace knexus

#endif  // KNEXUS_DEVICE_H
