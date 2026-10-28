#include "tt_runtime.h"

#include <assert.h>
#include <dlfcn.h>
#include <knexus-api.h>
#include <rt_buffer.h>
#include <rt_object.h>
#include <rt_pci.h>
#include <rt_runtime.h>
#include <rt_topology.h>
#include <rt_utilities.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <tt-metalium/tt_backend_api_types.hpp>

using namespace nxs;

TTRuntime *getRuntime() {
  static TTRuntime s_runtime;
  return &s_runtime;
}

// The PCI node of each PCIe-attached chip, indexed by tt-metal chip id. Chip
// k is the k-th visible /dev/tenstorrent/N in ascending bus-id order: UMD
// lists the char devices sorted by bus id, and TT_VISIBLE_DEVICES filters
// them (bus-id substrings, or integer indices into that sorted list) before
// chip ids are handed out. That order is UMD's, not a documented contract,
// so the count is checked against GetNumPCIeDevices() and, on a mismatch, no
// chip states a PCI node. The class listing and the link's generation and
// width are read once from sysfs under $KNEXUS_HOST_ROOT (rt_pci.h); the
// node's hwloc object, its NUMA node and the path level come from the
// plugin's hwloc topology (rt_topology.h).
struct ChipPci {
  rt::PciNode node;           // bus id, class index, gen/width (sysfs)
  hwloc_obj_t obj = nullptr;  // hwloc's PCI device at node.bus_id, or none
  nxs_long numa_node = -1;    // its first non-I/O ancestor's one NUMA node, or -1
};

static const std::vector<ChipPci> &chipPciNodes() {
  static const std::vector<ChipPci> s_nodes = [] {
    std::vector<rt::PciNode> all = rt::readPciClass(rt::pciHostRoot(), "tenstorrent");
    std::vector<rt::PciNode> nodes = all;
    if (const char *env = std::getenv("TT_VISIBLE_DEVICES")) {
      std::set<size_t> keep;
      std::stringstream ss(env);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        bool integer = tok.find_first_not_of("0123456789") == std::string::npos;
        if (integer) {
          size_t k = std::stoul(tok);
          if (k < all.size()) keep.insert(k);
          continue;
        }
        for (size_t k = 0; k < all.size(); ++k)
          if (all[k].bus_id.find(tok) != std::string::npos) keep.insert(k);
      }
      nodes.clear();
      for (size_t k : keep) nodes.push_back(all[k]);  // set order = bus-id order
    }
    size_t pcie = 0;
    try {
      pcie = ttm::GetNumPCIeDevices();
    } catch (...) {
    }
    if (nodes.size() != pcie) {
      NXSLOG_WARN("sysfs lists {} tenstorrent PCI nodes, tt-metal {} PCIe chips: no PciBusId",
                  nodes.size(), pcie);
      nodes.clear();
    }
    hwloc_topology_t topo = rt::topology(rt::TopologyIo::Pci);
    std::vector<ChipPci> chips;
    for (const auto &n : nodes) {
      ChipPci c;
      c.node = n;
      c.obj = rt::pciObject(topo, n.bus_id);
      c.numa_node = rt::pciNumaNode(topo, c.obj);
      if (c.obj)
        NXSLOG_INFO("tenstorrent chip {} at {}: NUMA node {}", chips.size(), n.bus_id, c.numa_node);
      else
        NXSLOG_WARN("hwloc states no PCI device at {}: no NUMA node, no path rows", n.bus_id);
      chips.push_back(c);
    }
    return chips;
  }();
  return s_nodes;
}

// The tt-metal device behind a plugin device: its 1x1 submesh's one IDevice
// (the MeshDevice overloads of the queries below throw).
static ttm::IDevice *chipOf(TTDevice *dev) {
  auto mesh = dev ? dev->get() : nullptr;
  if (!mesh) return nullptr;
  try {
    auto devs = mesh->get_devices();
    return devs.empty() ? nullptr : devs[0];
  } catch (...) {
    return nullptr;
  }
}

// The chip's PCI node, or null for a chip with none (a remote chip reached
// over ethernet, or a count mismatch above).
static const ChipPci *chipPci(TTDevice *dev) {
  ttm::IDevice *chip = chipOf(dev);
  if (!chip) return nullptr;
  try {
    if (!chip->is_mmio_capable()) return nullptr;
    size_t id = chip->id();
    const auto &nodes = chipPciNodes();
    return id < nodes.size() ? &nodes[id] : nullptr;
  } catch (...) {
    return nullptr;
  }
}

// One link of a device, as the Link* properties state it: a row per link, the
// vectors indexed by row. The tenstorrent plugin states three kinds of row —
//   PCIe to the host     (peer -1): generation and width, max and current,
//                         from the chip's sysfs PCI node;
//   Ethernet to a peer   (one row per connected ethernet core): the peer's
//                         index and bus id; no rate, width or level;
//   PCIe path to a peer  (one row per other PCIe-attached device): the level
//                         of the two hwloc PCI objects' common ancestor in
//                         NVML's values (rt::pciTopologyLevel).
// -1 and "" mean the row states no such figure. A chip with no PCI node
// states no host row and no path rows; a chip whose bus id hwloc states no
// object for states its host row but no path rows, and no peer has a path
// row to it. Nothing is measured.
struct DeviceLink {
  std::string type;
  nxs_long peer = -1;
  std::string peer_id;
  nxs_long gen_max = -1, gen_cur = -1, width_max = -1, width_cur = -1;
  nxs_long level = -1;
};

static std::vector<DeviceLink> computeLinks(TTRuntime *rt, nxs_int device_id) {
  std::vector<DeviceLink> links;
  TTDevice *self = rt->getDevice(device_id);
  const ChipPci *pci = chipPci(self);

  if (pci) {
    DeviceLink host;
    host.type = "PCIe";
    host.gen_max = pci->node.gen_max;
    host.gen_cur = pci->node.gen_cur;
    host.width_max = pci->node.width_max;
    host.width_cur = pci->node.width_cur;
    links.push_back(host);
  }

  // Ethernet: each active core's far end, grouped by peer. A core whose peer
  // is not visible to this process throws ("connects to a remote mmio
  // device"); it is skipped. get_ethernet_sockets(j) is not used: it throws
  // for every unlinked j.
  nxs_int count = rt->getNumDevices();
  std::vector<std::pair<nxs_long, std::string>> eth;  // (peer chip, core)
  if (ttm::IDevice *chip = chipOf(self)) {
    decltype(chip->get_active_ethernet_cores()) cores;
    try {
      cores = chip->get_active_ethernet_cores();
    } catch (...) {
    }
    for (const auto &core : cores) {
      try {
        auto [peer, peer_core] = chip->get_connected_ethernet_core(core);
        eth.emplace_back((nxs_long)peer, core.str());
      } catch (...) {
      }
    }
  }
  std::sort(eth.begin(), eth.end());
  for (const auto &[peer, core] : eth) {
    DeviceLink e;
    e.type = "Ethernet";
    e.peer = peer >= 0 && peer < count && peer != device_id ? peer : -1;
    if (e.peer >= 0)
      if (const ChipPci *p = chipPci(rt->getDevice(e.peer))) e.peer_id = p->node.bus_id;
    links.push_back(e);
  }

  // PCIe paths to every other PCIe-attached device.
  if (pci && pci->obj) {
    hwloc_topology_t topo = rt::topology(rt::TopologyIo::Pci);
    for (nxs_int j = 0; j < count; ++j) {
      if (j == device_id) continue;
      const ChipPci *other = chipPci(rt->getDevice(j));
      if (!other || !other->obj) continue;
      DeviceLink path;
      path.type = "PCIe";
      path.peer = j;
      path.peer_id = other->node.bus_id;
      path.level = rt::pciTopologyLevel(topo, pci->obj, other->obj);
      links.push_back(path);
    }
  }
  return links;
}

// Each device's rows, computed once: the topology does not change while the
// devices are open.
static const std::vector<DeviceLink> &deviceLinks(TTRuntime *rt, nxs_int device_id) {
  static std::mutex s_mutex;
  static std::map<nxs_int, std::vector<DeviceLink>> s_links;
  std::lock_guard<std::mutex> lock(s_mutex);
  auto it = s_links.find(device_id);
  if (it == s_links.end()) it = s_links.emplace(device_id, computeLinks(rt, device_id)).first;
  return it->second;
}

#undef NXS_API_CALL
#define NXS_API_CALL __attribute__((visibility("default")))

/************************************************************************
 * @def GetRuntimeProperty
 * @brief Return Runtime properties
 * @return Error status or Succes.
 ************************************************************************/
extern "C" nxs_status NXS_API_CALL
nxsGetRuntimeProperty(nxs_uint runtime_property_id, void *property_value,
                      size_t *property_value_size) {
  auto rt = getRuntime();

  NXSLOG_INFO("getRuntimeProperty {}", runtime_property_id);

  /* lookup HIP equivalent */
  /* return value size */
  /* return value */
  switch (runtime_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Name, NP_Type, NP_Vendor, NP_Size, NP_ID};
      int keys_count = sizeof(keys) / sizeof(keys[0]);
      return rt::getPropertyVec(property_value, property_value_size, keys,
                                keys_count);
    }
    case NP_Name:
      return rt::getPropertyStr(property_value, property_value_size, "tt-metal");
    case NP_Size:
      return rt::getPropertyInt(property_value, property_value_size, rt->getNumDevices());
    case NP_Vendor: {
      std::string name = "Tenstorrent";
      return rt::getPropertyStr(property_value, property_value_size,
                                name);
    }
    case NP_Type:
      return rt::getPropertyStr(property_value, property_value_size, "npu");
    case NP_ID: {
      //return rt::getPropertyInt(property_value, property_value_size,
      //                          cpuinfo_has_arm_sme2() ? 1 : 0);
    }
    default:
      return NXS_InvalidProperty;
  }
  return NXS_Success;
}

/************************************************************************
 * @def GetDeviceProperty
 * @brief Return Device properties
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL
nxsGetDeviceProperty(nxs_int device_id, nxs_uint device_property_id,
                     void *property_value, size_t *property_value_size) {
  auto *rt = getRuntime();
  if (device_id < 0 || device_id >= rt->getNumDevices()) return NXS_InvalidDevice;
  auto dev = rt->getDevice(device_id);
  if (!dev) return NXS_InvalidDevice;

  switch (device_property_id) {
    case NP_Keys: {
      std::vector<nxs_long> keys = {NP_Name, NP_Vendor, NP_Type, NP_Architecture, NP_Size};
      if (chipPci(dev)) keys.push_back(NP_PciBusId);
      if (!deviceLinks(rt, device_id).empty())
        keys.insert(keys.end(), {NP_LinkCount, NP_LinkType, NP_LinkPeer, NP_LinkPeerId,
                                 NP_LinkGenMax, NP_LinkGenCurrent, NP_LinkWidthMax,
                                 NP_LinkWidthCurrent, NP_LinkTopologyLevel});
      return rt::getPropertyVec(property_value, property_value_size, keys.data(),
                                keys.size());
    }
    case NP_Name: {
      // return getStr(property_value, property_value_size, device->core);
    }
    case NP_Vendor:
      return rt::getPropertyStr(property_value, property_value_size, "tenstorrent");
    case NP_Type:
      return rt::getPropertyStr(property_value, property_value_size, "npu");
    case NP_Architecture: {
      TT_NOBJ_CHECK(arch, dev->get()->arch);
      auto archName = tt::arch_to_str(arch);
      return rt::getPropertyStr(property_value, property_value_size,
                                archName);
    }
    case NP_Size: {
      TT_NOBJ_CHECK(grid, dev->get()->compute_with_storage_grid_size);
      nxs_long numCores = grid.x * grid.y;
      return rt::getPropertyInt(property_value, property_value_size,
                                numCores);
    }
    // The device's own identity and its links, from sysfs and tt-metal's
    // public API — the names and units of docs/Python_API.md "Host and link
    // properties".
    case NP_PciBusId: {
      const ChipPci *pci = chipPci(dev);
      if (!pci) return NXS_InvalidProperty;
      return rt::getPropertyStr(property_value, property_value_size, pci->node.bus_id);
    }
    case NP_LinkCount:
    case NP_LinkType:
    case NP_LinkPeer:
    case NP_LinkPeerId:
    case NP_LinkGenMax:
    case NP_LinkGenCurrent:
    case NP_LinkWidthMax:
    case NP_LinkWidthCurrent:
    case NP_LinkTopologyLevel: {
      const std::vector<DeviceLink> &links = deviceLinks(rt, device_id);
      if (links.empty()) return NXS_InvalidProperty;
      if (device_property_id == NP_LinkCount)
        return rt::getPropertyInt(property_value, property_value_size, links.size());
      std::vector<std::string> strs;
      std::vector<nxs_long> ints;
      for (const auto &l : links) {
        switch (device_property_id) {
          case NP_LinkType:          strs.push_back(l.type);      break;
          case NP_LinkPeerId:        strs.push_back(l.peer_id);   break;
          case NP_LinkPeer:          ints.push_back(l.peer);      break;
          case NP_LinkGenMax:        ints.push_back(l.gen_max);   break;
          case NP_LinkGenCurrent:    ints.push_back(l.gen_cur);   break;
          case NP_LinkWidthMax:      ints.push_back(l.width_max); break;
          case NP_LinkWidthCurrent:  ints.push_back(l.width_cur); break;
          default:                   ints.push_back(l.level);     break;
        }
      }
      if (device_property_id == NP_LinkType || device_property_id == NP_LinkPeerId)
        return rt::getPropertyStrVec(property_value, property_value_size, strs);
      return rt::getPropertyVec(property_value, property_value_size, ints.data(),
                                ints.size());
    }
    default:
      return NXS_InvalidProperty;
  }

  return NXS_Success;
}

/************************************************************************
 * @def CreateBuffer
 * @brief Create a buffer on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateBuffer(nxs_int device_id, nxs_buffer_layout shape,
                                                void *host_ptr,
                                                nxs_uint settings) {
  auto rt = getRuntime();
  auto dev = rt->getDevice(device_id);
  if (!dev) return NXS_InvalidDevice;

  NXSLOG_INFO("createBuffer {}", shape.rank);
  auto *buf = rt->getBuffer(dev, shape, host_ptr, settings);
  if (!buf) return NXS_InvalidBuffer;

  return rt->addObject(buf);
}

/************************************************************************
 * @def GetBufferProperty
 * @brief Return Buffer properties
 * @return Error status or Succes.
 ***********************************************************************/
 extern "C" nxs_status NXS_API_CALL nxsGetBufferProperty(nxs_int buffer_id, nxs_uint buffer_property_id,
                                                     void *property_value, size_t *property_value_size) {
  auto rt = getRuntime();
  auto buffer = rt->get<TTBuffer>(buffer_id);
  if (!buffer) return NXS_InvalidBuffer;

  switch (buffer_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Name, NP_Type, NP_Shape, NP_Size, NP_Value};
      return rt::getPropertyVec(property_value, property_value_size, keys, 5);
    }
    case NP_Value:
      return rt::getPropertyInt(property_value, property_value_size, reinterpret_cast<nxs_long>(buffer->getAddress()));
  }
  return NXS_InvalidProperty;
}

/************************************************************************
 * @def CopyBuffer
 * @brief Copy a buffer to the host
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsCopyBuffer(nxs_int buffer_id,
                                                 void *host_ptr,
                                                 nxs_uint settings) {
  auto rt = getRuntime();
  auto buf = rt->get<TTBuffer>(buffer_id);
  if (!buf) return NXS_InvalidBuffer;

  bool blocking = false;
  //if (settings & NXS_ExecutionSettings_NonBlocking) {
  //  blocking = false;
  //}

  if (settings == NXS_BufferDeviceToHost)
    return buf->copyToHost(host_ptr);
  return buf->copyToDevice(host_ptr, blocking);
}

/************************************************************************
 * @def ReleaseBuffer
 * @brief Release a buffer on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsReleaseBuffer(nxs_int buffer_id) {
  return getRuntime()->releaseBuffer(buffer_id);
}

/************************************************************************
 * @def CreateLibrary
 * @brief Create a library on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateLibrary(nxs_int device_id,
                                                 void *library_data,
                                                 nxs_uint data_size,
                                                 nxs_uint settings) {
  NXSLOG_INFO("createLibrary {} - {} - {}", device_id, (intptr_t)library_data, data_size);
  auto rt = getRuntime();
  auto dev = rt->getObject(device_id);
  if (!dev) return NXS_InvalidDevice;

  auto lib = rt->getLibrary(library_data, data_size, settings);
  return rt->addObject(lib);
}

/************************************************************************
 * @def CreateLibraryFromFile
 * @brief Create a library from a file
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateLibraryFromFile(
    nxs_int device_id, const char *library_path, nxs_uint settings) {
  NXSLOG_INFO("createLibraryFromFile {} - {}", device_id, library_path);
  auto rt = getRuntime();
  auto dev = rt->getObject(device_id);
  if (!dev) return NXS_InvalidDevice;

  auto lib = rt->getLibrary(library_path, settings);
  return rt->addObject(lib);
}

/************************************************************************
 * @def GetLibraryProperty
 * @brief Return Library properties
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL
nxsGetLibraryProperty(nxs_int library_id, nxs_uint library_property_id,
                      void *property_value, size_t *property_value_size) {
  return NXS_InvalidProperty;
}

/************************************************************************
 * @def ReleaseLibrary
 * @brief Release a library on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsReleaseLibrary(nxs_int library_id) {
  auto rt = getRuntime();
  auto lib = rt->getObject(library_id);
  if (!lib) return NXS_InvalidLibrary;
  rt->dropObject(library_id);
  return NXS_Success;
}

/************************************************************************
 * @def GetKernel
 * @brief Lookup a kernel in a library
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsGetKernel(nxs_int library_id,
                                             const char *kernel_name) {
  NXSLOG_INFO("getKernel {} - {}", library_id, kernel_name);
  auto rt = getRuntime();
  auto lib = rt->get<TTLibrary>(library_id);
  if (!lib) return NXS_InvalidProgram;

  auto *kern = lib->getKernel();
  return rt->addObject(kern);
}

/************************************************************************
 * @def GetKernelProperty
 * @brief Return Kernel properties
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL
nxsGetKernelProperty(nxs_int kernel_id, nxs_uint kernel_property_id,
                     void *property_value, size_t *property_value_size) {
  auto rt = getRuntime();
  auto kern = rt->getObject(kernel_id);
  if (!kern) return NXS_InvalidKernel;

  switch (kernel_property_id) {
    default:
      return NXS_InvalidProperty;
  }

  return NXS_Success;
}

/************************************************************************
 * @def ReleaseKernel
 * @brief Release a kernel on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsReleaseKernel(nxs_int kernel_id) {
  auto rt = getRuntime();
  if (!rt->dropObject(kernel_id)) return NXS_InvalidKernel;
  return NXS_Success;
}

/************************************************************************
 * @def CreateStream
 * @brief Create stream on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateStream(nxs_int device_id,
                                                nxs_uint stream_settings) {
  auto rt = getRuntime();
  auto dev = rt->getObject(device_id);
  if (!dev) return NXS_InvalidDevice;

  return NXS_Success;
}

/************************************************************************
 * @def ReleaseStream
 * @brief Release the stream on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsReleaseStream(nxs_int stream_id) {
  // auto rt = getRuntime();
  // if (!rt->dropObject<MTL::CommandQueue>(stream_id))
  //   return NXS_InvalidStream;
  return NXS_Success;
}

/************************************************************************
 * @def CreateSchedule
 * @brief Create schedule on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateSchedule(nxs_int device_id,
                                                  nxs_uint schedule_settings) {
  auto rt = getRuntime();
  auto dev = rt->getDevice(device_id);
  if (!dev) return NXS_InvalidDevice;

  return rt->getSchedule(dev, schedule_settings);
}

/************************************************************************
 * @def GetScheduleProperty
 * @brief Return Schedule properties
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL
nxsGetScheduleProperty(nxs_int schedule_id, nxs_uint schedule_property_id,
                       void *property_value, size_t *property_value_size) {
  NXSLOG_INFO("getScheduleProperty {}", schedule_property_id);
  auto rt = getRuntime();
  auto schedule = rt->get<TTSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;

  switch (schedule_property_id) {
    case NP_Keys: {
      constexpr nxs_long keys[] = {NP_ElapsedTime};
      constexpr int keys_count = sizeof(keys) / sizeof(keys[0]);
      return rt::getPropertyVec(property_value, property_value_size, keys,
                                keys_count);
    }
    case NP_ElapsedTime: {
      return rt::getPropertyFlt(property_value, property_value_size,
                                schedule->getTime());
    }
  }
  return NXS_Success;
}

/************************************************************************
 * @def RunSchedule
 * @brief Run the schedule on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsRunSchedule(nxs_int schedule_id,
                                                  nxs_int stream_id,
                                                  nxs_uint run_settings) {
  auto rt = getRuntime();
  auto schedule = rt->get<TTSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;

  return schedule->run(stream_id, run_settings);
}

/************************************************************************
 * @def ReleaseSchedule
 * @brief Release the schedule on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsReleaseSchedule(nxs_int schedule_id) {
  auto rt = getRuntime();
  return rt->releaseSchedule(schedule_id);
}

/************************************************************************
 * @def CreateCommand
 * @brief Create command on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateCommand(nxs_int schedule_id,
                                                 nxs_int kernel_id,
                                                 nxs_uint settings) {
  auto rt = getRuntime();
  auto schedule = rt->get<TTSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;
  auto kernel = rt->get<TTKernel>(kernel_id);
  if (!kernel) return NXS_InvalidKernel;

  auto command = rt->getCommand(kernel, settings);
  schedule->addCommand(command);
  return rt->addObject(command);
}

/************************************************************************
 * @def SetCommandArgument
 * @brief Set command argument on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsSetCommandArgument(nxs_int command_id,
                                                         nxs_int argument_index,
                                                         nxs_int buffer_id,
                                                         const char *name,
                                                         nxs_uint argument_settings
) {
  auto rt = getRuntime();

  auto command = rt->get<TTCommand>(command_id);
  if (!command) return NXS_InvalidCommand;

  auto buffer = rt->get<TTBuffer>(buffer_id);
  if (!buffer) return NXS_InvalidBuffer;

  command->setBufferArgument(argument_index, buffer);
  return command->setScalar(argument_index, buffer->getAddress(), "Buffer", argument_settings);
}

/************************************************************************
 * @def SetCommandScalar
 * @brief Set command scalar on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsSetCommandScalar(nxs_int command_id,
                                                       nxs_int argument_index,
                                                       void *value,
                                                       const char *name,
                                                       nxs_uint argument_settings
) {
  auto rt = getRuntime();
  auto command = rt->get<TTCommand>(command_id);
  if (!command) return NXS_InvalidCommand;
  return command->setScalar(argument_index, value, name, argument_settings);
}

/************************************************************************
 * @def FinalizeCommand
 * @brief Finalize command setup
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsFinalizeCommand(nxs_int command_id,
                                                      nxs_dim3 grid_size,
                                                      nxs_dim3 group_size,
                                                      nxs_uint shared_memory_size) {

  NXSLOG_INFO("finalizeCommand {} - {{ {}, {}, {} }} - {{ {}, {}, {} }}", command_id, grid_size.x,
             grid_size.y, grid_size.z, group_size.x, group_size.y, group_size.z);

  auto rt = getRuntime();

  auto command = rt->get<TTCommand>(command_id);
  if (!command) return NXS_InvalidCommand;

  return command->finalize(grid_size, group_size, shared_memory_size);
}
