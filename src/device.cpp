#define NEXUS_LOG_MODULE "device"

#include <nexus/buffer.h>
#include <nexus/device_db.h>
#include <nexus/log.h>
#include <nexus/runtime.h>
#include <nexus/utility.h>

#include <filesystem>

#include "_buffer_impl.h"
#include "_device_impl.h"
#include "_info_impl.h"
#include "_runtime_impl.h"

using namespace nexus;

#define APICALL(FUNC, ...)                                                   \
  nxs_int apiResult = getParent()->runAPIFunction<NF_##FUNC>(__VA_ARGS__)

detail::DeviceImpl::DeviceImpl(detail::Impl base) : detail::Impl(base) {
  NXSLOG_TRACE("CTOR: {}", getId());
  auto vendor = getProperty(NP_Vendor);
  auto type = getProperty(NP_Type);
  auto arch = getProperty(NP_Architecture);
  if (!vendor || !type || !arch) {
    NXSLOG_WARN("device tag properties not found");
    return;
  }
  auto devTag = vendor->getValue<NP_Vendor>() + "-" +
                type->getValue<NP_Type>() + "-" +
                arch->getValue<NP_Architecture>();
  NXSLOG_INFO("device tag: {}", devTag);
  if (auto info = nexus::lookupDeviceInfo(devTag))
    deviceInfo = info;
  else  // load defaults
    NXSLOG_WARN("device properties not found");
}

detail::DeviceImpl::~DeviceImpl() {
  NXSLOG_TRACE("DTOR: {}", getId());
  buffers.clear();
  events.clear();
  schedules.clear();
  streams.clear();
  libraries.clear();
  (void)release();
}

void detail::DeviceImpl::releaseChild(Impl *obj) {
  if (!obj) return;
  buffers.removeByImpl(obj);
  libraries.removeByImpl(obj);
  schedules.removeByImpl(obj);
  streams.removeByImpl(obj);
  events.removeByImpl(obj);
}

std::optional<Property> detail::DeviceImpl::getProperty(nxs_int prop) const {
  auto *rt = getParentOfType<RuntimeImpl>();
  return rt->getAPIProperty<NF_nxsGetDeviceProperty>(prop, getId());
}

// Runtime functions
// Private helper function to find a library in a catalog
struct LibraryInfo {
  std::string arch;
  std::string binaryData;
  nxs_long size;
  Info libraryNode;
};

static void findDeviceBinary(LibraryInfo &info, Info catalogInfo,
                             const std::string &libraryName,
                             const std::string &arch) {
  if (auto libs = catalogInfo.getNode({"Libraries"})) {
    for (auto lib : *libs) {
      try {
        auto name = lib.at("Name").get<std::string_view>();
        if (name == libraryName) {
          for (auto &narch : lib.at("Architectures")) {
            // TODO: check compatibility
            auto narchName = narch.at("Name").get<std::string_view>();
            if (narchName == arch) {
              // TODO: check
              auto binaryData = narch.at("BinaryData").get<std::string_view>();
              auto size = narch.at("FileSize").get<nxs_long>();
              info.arch = narchName.data();
              info.binaryData = binaryData.data();
              info.size = size;
              Info::Node node(lib);
              info.libraryNode = Info(node);
              break;
            }
          }
        }
      } catch (...) {
        NXSLOG_ERROR("binary not found");
      }
    }
  }
}

Library detail::DeviceImpl::loadLibrary(Info catalog,
                                        const std::string &libraryName) {
  NXSLOG_TRACE("loadLibrary {} - {}", getId(), libraryName);
  auto arch = getProperty(NP_Architecture)->getValue<std::string>();
  LibraryInfo libInfo;
  findDeviceBinary(libInfo, catalog, libraryName, arch);
  if (libInfo.arch.empty()) {
    NXSLOG_ERROR("library not found: {}", libraryName);
    return Library();
  }
  // std::vector<uint8_t> data = base64Decode(libInfo.binaryData, libInfo.size);
  APICALL(nxsCreateLibrary, getId(), (void *)libInfo.binaryData.data(),
          libInfo.binaryData.size(), 0);
  Library lib(detail::Impl(this, apiResult, 0), libInfo.libraryNode);
  libraries.add(lib);
  return lib;
}

Library detail::DeviceImpl::createLibrary(void *data, size_t size,
                                          nxs_uint settings) {
  NXSLOG_TRACE("createLibrary {} - {}", getId(), size);
  APICALL(nxsCreateLibrary, getId(), data, size, settings);
  Library lib(detail::Impl(this, apiResult, settings));
  libraries.add(lib);
  return lib;
}

Library detail::DeviceImpl::createLibrary(const std::string &path,
                                          nxs_uint settings) {
  NXSLOG_TRACE("createLibrary {} - {}", getId(), path);
  APICALL(nxsCreateLibraryFromFile, getId(), path.c_str(), settings);
  Library lib(detail::Impl(this, apiResult, settings));
  libraries.add(lib);
  return lib;
}

Schedule detail::DeviceImpl::createSchedule(nxs_uint settings) {
  NXSLOG_TRACE("createSchedule {}", getId());
  APICALL(nxsCreateSchedule, getId(), settings);
  Schedule sched(detail::Impl(this, apiResult, settings));
  schedules.add(sched);
  return sched;
}

Stream detail::DeviceImpl::createStream(nxs_uint settings) {
  NXSLOG_TRACE("createStream {}", getId());
  APICALL(nxsCreateStream, getId(), 0);
  Stream stream(detail::Impl(this, apiResult, settings));
  streams.add(stream);
  return stream;
}

Event detail::DeviceImpl::createEvent(nxs_event_type event_type,
                                      nxs_uint settings) {
  NXSLOG_TRACE("createEvent {}", getId());
  APICALL(nxsCreateEvent, getId(), event_type, settings);
  Event event(detail::Impl(this, apiResult, settings));
  events.add(event);
  return event;
}

Buffer detail::DeviceImpl::createBuffer(const Layout &layout, const void *data,
                                        nxs_uint settings) {
  NXSLOG_TRACE("createBuffer {} - {}", getId(), layout.getNumElements());
  Layout normalized_layout = layout;
  if (normalized_layout.getDataType() == NXS_DataType_Undefined) {
    auto data_type = nxsGetDataType(settings);
    if (data_type != NXS_DataType_Undefined) {
      normalized_layout.setDataType(data_type);
    }
  }
  nxs_uint buffer_settings =
      settings & (NXS_BufferSettings_OnHost | NXS_BufferSettings_OnDevice |
                  NXS_BufferSettings_Maintain);
  APICALL(nxsCreateBuffer, getId(), normalized_layout.get(), (void *)data,
          buffer_settings);
  Buffer nbuf(Impl(this, apiResult, buffer_settings), normalized_layout, data);
  buffers.add(nbuf);
  return nbuf;
}

Buffer detail::DeviceImpl::copyBuffer(Buffer buf, nxs_uint settings) {
  NXSLOG_TRACE("copyBuffer {} - {}", getId(), buf.getSizeBytes());
  settings |= buf.getSettings() & ~NXS_BufferSettings_OnDevice;
  auto *data_ptr = buf.getDataPtr();
  APICALL(nxsCreateBuffer, getId(), buf.getLayout().get(), (void *)data_ptr,
          settings);
  Buffer nbuf(Impl(this, apiResult, settings), buf.getLayout(), data_ptr);
  buffers.add(nbuf);
  return nbuf;
}

///////////////////////////////////////////////////////////////////////////////
/// Object wrapper - Device
///////////////////////////////////////////////////////////////////////////////
Device::Device(detail::Impl base) : Object(base) {}

// Get Device Property Value
std::optional<Property> Device::getProperty(nxs_int prop) const {
  NEXUS_OBJ_MCALL(std::nullopt, getProperty, prop);
}

Info Device::getInfo() const { NEXUS_OBJ_MCALL(Info(), getInfo); }

// Runtime functions
Librarys Device::getLibraries() const { NEXUS_OBJ_MCALL(Librarys(), getLibraries); }

Schedules Device::getSchedules() const { NEXUS_OBJ_MCALL(Schedules(), getSchedules); }

Streams Device::getStreams() const { NEXUS_OBJ_MCALL(Streams(), getStreams); }

Buffers Device::getBuffers() const { NEXUS_OBJ_MCALL(Buffers(), getBuffers); }

Events Device::getEvents() const { NEXUS_OBJ_MCALL(Events(), getEvents); }

Event Device::createEvent(nxs_event_type event_type, nxs_uint settings) {
  NEXUS_OBJ_MCALL(Event(), createEvent, event_type, settings);
}

Stream Device::createStream(nxs_uint settings) {
  NEXUS_OBJ_MCALL(Stream(), createStream, settings);
}

Schedule Device::createSchedule(nxs_uint settings) {
  NEXUS_OBJ_MCALL(Schedule(), createSchedule, settings);
}

Buffer Device::createBuffer(const Layout &layout, const void *data, nxs_uint settings) {
  NEXUS_OBJ_MCALL(Buffer(), createBuffer, layout, data, settings);
}

Buffer Device::copyBuffer(Buffer buf, nxs_uint settings) {
  NEXUS_OBJ_MCALL(Buffer(), copyBuffer, buf, settings);
}

Library Device::loadLibrary(Info catalog, const std::string &libraryName) {
  NEXUS_OBJ_MCALL(Library(), loadLibrary, catalog, libraryName);
}

Library Device::createLibrary(void *libraryData, size_t librarySize,
                              nxs_uint settings) {
  NEXUS_OBJ_MCALL(Library(), createLibrary, libraryData, librarySize, settings);
}

Library Device::createLibrary(const std::string &libraryPath,
                              nxs_uint settings) {
  NEXUS_OBJ_MCALL(Library(), createLibrary, libraryPath, settings);
}
