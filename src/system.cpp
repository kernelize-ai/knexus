#define NEXUS_LOG_MODULE "system"

#include <nexus/system.h>
#include <nexus/utility.h>
#include <nexus/log.h>

#include "_system_impl.h"

using namespace nexus;
using namespace nexus::detail;

/// @brief Construct a Platform for the current system
SystemImpl::SystemImpl(int) {
  NXSLOG_TRACE("CTOR");
  iterateEnvPaths("NEXUS_RUNTIME_PATH", "./runtime_libs",
                  [&](const std::string &path, const std::string &name) {
                    Runtime rt(detail::Impl(this, runtimes.size()), path);
                    runtimes.add(rt);
                    runtimeMap[rt.getProp<std::string>(NP_Name)] = rt;
                  });
}

SystemImpl::~SystemImpl() {
  NXSLOG_TRACE("DTOR");
  // for (auto rt : runtimes)
  //   rt.release();
  // for (auto buf : buffers)
  //   buf.release();
}

std::optional<Property> SystemImpl::getProperty(nxs_int prop) const {
  return std::nullopt;
}

Buffer SystemImpl::createBuffer(const Layout &layout, const void *hostData,
                                nxs_uint settings) {
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
  NXSLOG_TRACE("createBuffer {}", normalized_layout.getNumElements());
  nxs_uint id = buffers.size();
  Buffer buf(detail::Impl(this, id, buffer_settings), normalized_layout, hostData);
  buffers.add(buf);
  return buf;
}

Buffer SystemImpl::copyBuffer(Buffer buf, Device dev, nxs_uint settings) {
  NXSLOG_TRACE("copyBuffer {}", buf.getSizeBytes());
  Buffer nbuf = dev.copyBuffer(buf, settings);
  return nbuf;
}

Info SystemImpl::loadCatalog(const std::string &catalogPath) {
  NXSLOG_TRACE("loadCatalog {}", catalogPath);
  Info cat(catalogPath);
  catalogs.add(cat);
  return cat;
}

///////////////////////////////////////////////////////////////////////////////
/// @param
System::System(int i) : Object(i) {}

std::optional<Property> System::getProperty(nxs_int prop) const {
  NEXUS_OBJ_MCALL(std::nullopt, getProperty, prop);
}

Buffers System::getBuffers() const { NEXUS_OBJ_MCALL(Buffers(), getBuffers); }

Runtimes System::getRuntimes() const { NEXUS_OBJ_MCALL(Runtimes(), getRuntimes); }

Infos System::getCatalogs() const {
  NEXUS_OBJ_MCALL(Infos(), getCatalogs);
}

Runtime System::getRuntime(int idx) const { NEXUS_OBJ_MCALL(Runtime(), getRuntime, idx); }
Runtime System::getRuntime(const std::string &name) { NEXUS_OBJ_MCALL(Runtime(), getRuntime, name); }

Info System::loadCatalog(const std::string &catalogPath) {
  NEXUS_OBJ_MCALL(Info(), loadCatalog, catalogPath);
}

Buffer System::createBuffer(const Layout &layout, const void *hostData,
                            nxs_uint settings) {
  NEXUS_OBJ_MCALL(Buffer(), createBuffer, layout, hostData, settings);
}

Buffer System::copyBuffer(Buffer buf, Device dev, nxs_uint settings) {
  NEXUS_OBJ_MCALL(Buffer(), copyBuffer, buf, dev, settings);
}

/// @brief Get the System Platform
/// @return
nexus::System nexus::getSystem() {
  static System s_system(0);
  return s_system;
}
