#define KNEXUS_LOG_MODULE "system"
#include <knexus/log.h>

#include <knexus/system.h>
#include <knexus/utility.h>

#include <knexus/topology.h>
#include <knexus/pci.h>

#include "_fabric_links.h"
#include "_link_impl.h"
#include "_pcie_links.h"
#include "_system_impl.h"

using namespace knexus;
using namespace knexus::detail;

/// @brief Construct a Platform for the current system
SystemImpl::SystemImpl(int) {
  NXSLOG_TRACE("CTOR");
  iterateEnvPaths("KNEXUS_RUNTIME_PATH", "./runtime_libs",
                  [&](const std::string &path, const std::string &name) {
                    Runtime rt(detail::Impl(this, runtimes.size()), path);
                    runtimes.add(rt);
                    runtimeMap[rt.getProp<std::string>(NP_Name)] = rt;
                  });
}

void SystemImpl::releaseChildren() {
  {
    std::lock_guard<std::mutex> lock(linksMutex);
    links.clear();
    linksBuilt = false;
  }
  runtimeMap.clear();
  runtimes.clear();
  catalogs.clear();
  buffers.clear();
}

Devices SystemImpl::getDevices() const {
  Devices all;
  for (auto rt : runtimes)
    for (auto dev : rt.getDevices()) all.add(dev);
  return all;
}

Links SystemImpl::getLinks() {
  std::lock_guard<std::mutex> lock(linksMutex);
  if (!linksBuilt) {
    buildLinks();
    linksBuilt = true;
  }
  return links;
}

namespace {
template <typename T>
std::vector<T> column(const Runtime &rt, nxs_int prop) {
  if (auto p = rt.getProperty(prop))
    if (auto *v = std::get_if<std::vector<T>>(&*p)) return *v;
  return {};
}
}  // namespace

// Every PCIe link, host and path, from the core's own hwloc topology (loaded
// with I/O objects on this first call) and sysfs under $KNEXUS_HOST_ROOT, for
// every device of every runtime that states a PciBusId. The host endpoint is
// the cpu runtime's device 0, an empty Device without a cpu runtime. Then
// every runtime's fabric table (docs/Plugin_API.md "Fabric links"), read once
// here, each row as one link, stated in the direction the plugin gives.
void SystemImpl::buildLinks() {
  Devices devices = getDevices();
  std::vector<std::string> busIds;
  for (auto dev : devices) {
    std::string bus;
    if (auto prop = dev.getProperty(NP_PciBusId))
      if (auto *s = std::get_if<std::string>(&*prop)) bus = *s;
    busIds.push_back(bus);
  }
  Device host;
  auto cpu = runtimeMap.find("cpu");
  if (cpu != runtimeMap.end()) host = cpu->second.getDevice(0);

  hwloc_topology_t topo = nxs::rt::topology(nxs::rt::TopologyIo::Pci);
  for (auto &pl : buildPcieLinks(busIds, topo, nxs::rt::pciHostRoot())) {
    LinkFacts facts;
    facts.type = "PCIe";
    facts.gen_max = pl.gen_max;
    facts.gen_cur = pl.gen_cur;
    facts.width_max = pl.width_max;
    facts.width_cur = pl.width_cur;
    facts.level = pl.level;
    Device b = pl.isHost() ? host : devices.get(pl.b);
    links.add(Link(detail::Impl(this, links.size()), devices.get(pl.a), b, facts));
  }
  size_t pcie = links.size();

  nxs_int base = 0;
  for (auto rt : runtimes) {
    nxs_int size = rt.getDevices().size();
    FabricTable table;
    table.type = column<std::string>(rt, NP_LinkType);
    table.device = column<nxs_long>(rt, NP_LinkDevice);
    if (!table.type.empty()) {
      table.peer = column<nxs_long>(rt, NP_LinkPeer);
      table.peer_id = column<std::string>(rt, NP_LinkPeerId);
      table.gen_max = column<nxs_long>(rt, NP_LinkGenMax);
      table.gen_cur = column<nxs_long>(rt, NP_LinkGenCurrent);
      table.width_max = column<nxs_long>(rt, NP_LinkWidthMax);
      table.width_cur = column<nxs_long>(rt, NP_LinkWidthCurrent);
      if (table.device.size() != table.type.size())
        NXSLOG_WARN("getLinks: a runtime states {} LinkType but {} LinkDevice rows",
                    table.type.size(), table.device.size());
    }
    for (auto &fl : buildFabricLinks(table, base, size, busIds)) {
      Device b = fl.b == FabricLink::kNoDevice ? Device() : devices.get(fl.b);
      links.add(Link(detail::Impl(this, links.size()), devices.get(fl.a), b, fl.facts));
    }
    base += size;
  }
  NXSLOG_INFO("getLinks: {} links ({} PCIe, {} fabric)", links.size(), pcie,
              links.size() - pcie);
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
  KNEXUS_OBJ_MCALL(std::nullopt, getProperty, prop);
}

Buffers System::getBuffers() const { KNEXUS_OBJ_MCALL(Buffers(), getBuffers); }

Runtimes System::getRuntimes() const { KNEXUS_OBJ_MCALL(Runtimes(), getRuntimes); }

Devices System::getDevices() const { KNEXUS_OBJ_MCALL(Devices(), getDevices); }

Links System::getLinks() const { KNEXUS_OBJ_MCALL(Links(), getLinks); }

Infos System::getCatalogs() const {
  KNEXUS_OBJ_MCALL(Infos(), getCatalogs);
}

Runtime System::getRuntime(int idx) const { KNEXUS_OBJ_MCALL(Runtime(), getRuntime, idx); }
Runtime System::getRuntime(const std::string &name) { KNEXUS_OBJ_MCALL(Runtime(), getRuntime, name); }

Info System::loadCatalog(const std::string &catalogPath) {
  KNEXUS_OBJ_MCALL(Info(), loadCatalog, catalogPath);
}

Buffer System::createBuffer(const Layout &layout, const void *hostData,
                            nxs_uint settings) {
  KNEXUS_OBJ_MCALL(Buffer(), createBuffer, layout, hostData, settings);
}

Buffer System::copyBuffer(Buffer buf, Device dev, nxs_uint settings) {
  KNEXUS_OBJ_MCALL(Buffer(), copyBuffer, buf, dev, settings);
}

/// @brief Get the System Platform
/// @return
knexus::System knexus::getSystem() {
  static System s_system(0);
  return s_system;
}
