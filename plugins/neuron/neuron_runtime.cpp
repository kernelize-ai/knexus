#include "neuron_runtime.h"

#include <knexus-api.h>
#include <knexus/pci.h>
#include <rt_buffer.h>
#include <rt_utilities.h>

#include <nrt/nrt_experimental.h>
#include <nrt/nrt_version.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace nxs;
namespace fs = std::filesystem;

NeuronRuntime *getRuntime() {
  static NeuronRuntime s_runtime;
  return &s_runtime;
}

// The host's logical core ids NEURON_RT_VISIBLE_CORES names ("0-3", "1,3"),
// ascending; empty where it is unset or unparsable.
static std::vector<nxs_long> visibleCoresEnv() {
  std::set<nxs_long> ids;
  const char *env = std::getenv("NEURON_RT_VISIBLE_CORES");
  if (!env || !*env) return {};
  std::stringstream ss(env);
  std::string tok;
  try {
    while (std::getline(ss, tok, ',')) {
      if (tok.empty()) continue;
      size_t dash = tok.find('-');
      nxs_long lo = std::stoll(tok.substr(0, dash));
      nxs_long hi = dash == std::string::npos ? lo : std::stoll(tok.substr(dash + 1));
      for (nxs_long i = lo; i <= hi; ++i) ids.insert(i);
    }
  } catch (...) {
    return {};
  }
  return {ids.begin(), ids.end()};
}

// The Neuron devices sysfs lists, N of each sys/class/neuron_device/neuron<N>,
// ascending.
static std::vector<nxs_long> neuronChips(const fs::path &root) {
  std::vector<nxs_long> chips;
  std::error_code ec;
  fs::path dir = root / "sys" / "class" / "neuron_device";
  if (!fs::is_directory(dir, ec)) return chips;
  for (auto &entry : fs::directory_iterator(dir, ec)) {
    std::string name = entry.path().filename().string();
    if (name.size() <= 6 || name.compare(0, 6, "neuron") != 0) continue;
    if (name.find_first_not_of("0123456789", 6) != std::string::npos) continue;
    chips.push_back(std::stoll(name.substr(6)));
  }
  std::sort(chips.begin(), chips.end());
  return chips;
}

// The PCI nodes bound to the Neuron driver (its PCI driver name is
// "neuron-driver"), ascending by bus id. The class entries neuron<N> are
// virtual devices with no `device` link, so the bus id comes from the
// driver's listing instead.
static std::vector<std::string> neuronPciNodes(const fs::path &root) {
  std::vector<std::string> nodes;
  std::error_code ec;
  fs::path dir = root / "sys" / "bus" / "pci" / "drivers" / "neuron-driver";
  if (!fs::is_directory(dir, ec)) return nodes;
  for (auto &entry : fs::directory_iterator(dir, ec)) {
    rt::PciNode n;
    if (rt::readPciNode(entry.path(), n)) nodes.push_back(n.bus_id);
  }
  std::sort(nodes.begin(), nodes.end());
  return nodes;
}

// The Neuron devices the driver lists as NeuronLink neighbours of chip `chip`
// in neuron<chip>/info/connected_devices: "12, 1, 3, 4", one entry per
// neighbour link (east, west, north, south), empty on a host with no
// chip-to-chip links. The ids are the driver's device indices, the N of
// neuron<N>. The driver does not probe the links: it answers from a table per
// instance type (aws-neuronx-dkms, fw_io_topology_v2/v3).
static std::vector<nxs_long> connectedChips(const fs::path &root, nxs_long chip) {
  std::vector<nxs_long> ids;
  std::stringstream ss(rt::pci_detail::readLine(root / "sys" / "class" / "neuron_device" /
                                                ("neuron" + std::to_string(chip)) / "info" /
                                                "connected_devices"));
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    try {
      ids.push_back(std::stoll(tok));
    } catch (...) {
    }
  }
  return ids;
}

// The runtime's fabric table (rt_fabric.h): one NeuronLink row per entry of a
// chip's connected_devices, each chip pair once. A link joins two chips, and a
// device is a core, so a chip is stood for by its first visible core (the
// lowest device index on it); the other cores of the chip state no row, so
// the count of NeuronLinks between two chips is the count of rows between
// their first cores. A pair is stated from the lower-numbered chip's list,
// unless that list omits the higher one (an asymmetric listing), and from the
// side with a visible core: a chip with none is the peer, -1 with its bus id,
// and a pair with no visible core on either side is dropped. The driver
// states no link version or width, so none is stated. No row joins two cores
// of one chip: neither libnrt nor sysfs reports an on-chip interconnect.
static std::vector<rt::FabricLink> neuronLinks(const fs::path &root,
                                               const std::vector<nxs_long> &chips,
                                               const std::vector<std::string> &pci,
                                               const std::vector<NeuronDevice> &devices) {
  std::map<nxs_long, nxs_long> first_core;  // chip to its lowest visible device
  for (const auto &d : devices)
    if (d.chip >= 0 && !first_core.count(d.chip)) first_core[d.chip] = d.vnc;
  std::map<nxs_long, std::string> bus_of;  // chip to its PCI node
  for (size_t c = 0; c < chips.size() && c < pci.size(); ++c) bus_of[chips[c]] = pci[c];
  std::map<nxs_long, std::vector<nxs_long>> lists;
  for (nxs_long c : chips) lists[c] = connectedChips(root, c);
  auto lists_has = [&](nxs_long chip, nxs_long peer) {
    auto it = lists.find(chip);
    return it != lists.end() &&
           std::find(it->second.begin(), it->second.end(), peer) != it->second.end();
  };
  auto core_of = [&](nxs_long chip) {
    auto it = first_core.find(chip);
    return it == first_core.end() ? (nxs_long)-1 : it->second;
  };

  struct Row {
    nxs_long device, peer, peer_chip;
    bool operator<(const Row &o) const {
      return std::tie(device, peer, peer_chip) < std::tie(o.device, o.peer, o.peer_chip);
    }
  };
  std::vector<Row> rows;
  for (const auto &[a, peers] : lists) {
    for (nxs_long b : peers) {
      if (b == a) continue;
      if (b < a && lists_has(b, a)) continue;  // stated from b's list
      nxs_long ca = core_of(a), cb = core_of(b);
      if (ca < 0 && cb < 0) continue;
      if (ca < 0) rows.push_back({cb, -1, a});
      else rows.push_back({ca, cb, b});
    }
  }
  std::sort(rows.begin(), rows.end());

  std::vector<rt::FabricLink> links;
  for (const auto &r : rows) {
    rt::FabricLink l;
    l.type = "NeuronLink";
    l.device = r.device;
    l.peer = r.peer;
    auto it = bus_of.find(r.peer_chip);
    if (it != bus_of.end()) l.peer_id = it->second;
    links.push_back(l);
  }
  return links;
}

// Each visible logical core's host-wide id, chip, name and PCI node.
//
// - Core ids: NEURON_RT_VISIBLE_CORES's list where it names as many cores as
//   libnrt sees, else 0..N-1 where every core is visible; otherwise (another
//   selector, NEURON_RT_NUM_CORES say) unknown, and no core states a chip.
// - Chip: core id / logical cores per chip, the host's logical core count
//   split evenly over the chips sysfs lists (/dev/neuron<chip>).
// - PCI node: chip k is the k-th neuron-driver PCI node in bus-id order. The
//   driver numbers chips in probe order, which is PCI enumeration order; that
//   is the driver's behaviour, not a documented contract, so the counts are
//   checked and, on a mismatch, no core states a PciBusId.
// - lnc: the chip's physical cores (neuron<chip>/core_count) per logical core.
// Everything reads sysfs under $KNEXUS_HOST_ROOT (knexus/pci.h).
void NeuronRuntime::discover() {
  nrt_version_t ver{};
  if (nrt_get_version(&ver, sizeof(ver)) == NRT_SUCCESS)
    version = std::to_string(ver.rt_major) + "." + std::to_string(ver.rt_minor) + "." +
              std::to_string(ver.rt_patch) + "." + std::to_string(ver.rt_maintenance);

  uint32_t visible = 0, total = 0;
  NRT_STATUS st = nrt_get_visible_vnc_count(&visible);
  if (st != NRT_SUCCESS) {
    NXSLOG_ERROR("nrt_get_visible_vnc_count failed, 0 devices: {}", nrt_get_status_as_str(st));
    return;
  }
  if (nrt_get_total_vnc_count(&total) != NRT_SUCCESS) total = 0;

  nrt_instance_info_t info{};
  if (nrt_get_instance_info(&info, sizeof(info)) == NRT_SUCCESS)
    arch.assign(info.arch_name, strnlen(info.arch_name, sizeof(info.arch_name)));

  fs::path root = rt::pciHostRoot();
  std::vector<nxs_long> chips = neuronChips(root);
  std::vector<std::string> pci = neuronPciNodes(root);
  if (pci.size() != chips.size()) {
    NXSLOG_WARN("sysfs lists {} Neuron devices, {} neuron PCI nodes: no PciBusId",
                chips.size(), pci.size());
    pci.clear();
  }
  nxs_long per_chip = 0;
  if (!chips.empty() && total > 0 && total % chips.size() == 0)
    per_chip = total / chips.size();

  std::vector<nxs_long> ids = visibleCoresEnv();
  if (ids.size() != visible) {
    ids.clear();
    if (visible == total)
      for (nxs_long i = 0; i < (nxs_long)total; ++i) ids.push_back(i);
  }

  devices.resize(visible);
  for (uint32_t k = 0; k < visible; ++k) {
    NeuronDevice &d = devices[k];
    d.vnc = k;
    if (k < ids.size()) d.global_vnc = ids[k];
    if (d.global_vnc >= 0 && per_chip > 0 && d.global_vnc / per_chip < (nxs_long)chips.size()) {
      size_t c = d.global_vnc / per_chip;
      d.chip = chips[c];
      fs::path dev = root / "sys" / "class" / "neuron_device" / ("neuron" + std::to_string(d.chip));
      d.name = rt::pci_detail::readLine(dev / "info" / "architecture" / "device_name");
      nxs_long cores = rt::pci_detail::readLong(dev / "core_count");
      if (cores > 0 && cores % per_chip == 0) d.lnc = cores / per_chip;
      if (c < pci.size()) d.bus_id = pci[c];
    }
    NXSLOG_INFO("Neuron device {}: core {} on neuron{} ({}), lnc {}", k, d.global_vnc, d.chip,
                d.bus_id, d.lnc);
  }
  for (auto &d : devices) addObject(&d);

  links = neuronLinks(root, chips, pci, devices);
  NXSLOG_INFO("Neuron runtime: {} NeuronLink rows", links.size());
}

NeuronRuntime::~NeuronRuntime() {
  // Queued executions finish first: they use the tensor sets, the models and
  // the tensors. Then the sets (commands), which name the tensors; the
  // models; the tensors; and last nrt_close, which tears down the cores.
  for (auto &q : queues) q->drain();
  command_pool.clear();
  schedule_pool.clear();
  event_pool.clear();
  stream_pool.clear();
  library_pool.clear();
  buffer_pool.clear();
  if (initialized) {
    NXSLOG_INFO("Close Neuron runtime");
    nrt_close();
  }
}

bool NeuronRuntime::ensureInit() {
  std::lock_guard<std::mutex> lock(init_mutex);
  if (initialized) return true;
  NXSLOG_INFO("nrt_init: claiming {} logical NeuronCores", devices.size());
  NRT_STATUS st = nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", "");
  if (st != NRT_SUCCESS) {
    NXSLOG_ERROR("nrt_init failed (cores held by another process?): {}",
                 nrt_get_status_as_str(st));
    return false;
  }
  initialized = true;
  // Cross-check the sysfs chip of each core against libnrt's own mapping.
  for (auto &d : devices) {
    int chip = -1;
    if (d.chip >= 0 && nrt_get_device_id(d.vnc, &chip) == NRT_SUCCESS && chip != d.chip)
      NXSLOG_WARN("Neuron device {}: sysfs puts it on neuron{}, libnrt on neuron{}", d.vnc,
                  d.chip, chip);
  }
  return true;
}

NeuronBuffer *NeuronRuntime::getBuffer(NeuronDevice *device, nxs_buffer_layout shape,
                                       void *data_ptr, nxs_uint settings, nxs_status &status) {
  if (!ensureInit()) {
    status = NXS_DeviceNotAvailable;
    return nullptr;
  }
  auto *buf = buffer_pool.get_new(device, shape, data_ptr, settings);
  if (!buf) {
    status = NXS_OutOfHostMemory;
    return nullptr;
  }
  status = buf->allocate();
  if (nxs_failed(status)) {
    buf->freeBuffer();
    buffer_pool.release(buf);
    return nullptr;
  }
  return buf;
}

nxs_status NeuronRuntime::releaseBuffer(nxs_int buffer_id) {
  auto buf = get<NeuronBuffer>(buffer_id);
  if (!buf) return NXS_InvalidBuffer;
  // A queued execution may use the tensor, and a command's set names it.
  drainQueue(buf->getDevice());
  {
    std::lock_guard<std::mutex> lock(command_mutex);
    for (auto *cmd : live_commands)
      if (cmd->unbind(buf))
        NXSLOG_INFO("releaseBuffer {}: unbound from command {}", buffer_id, cmd->getId());
  }
  buf->freeBuffer();
  buffer_pool.release(buf);
  if (!dropObject(buffer_id)) return NXS_InvalidBuffer;
  return NXS_Success;
}

// nrt_init first (a NEFF is loaded onto a claimed core), then nrt_load from
// the bytes, or from the file read whole.
NeuronLibrary *NeuronRuntime::getLibrary(NeuronDevice *device, const std::string &path,
                                         const void *data, size_t size, nxs_status &status) {
  if (!ensureInit()) {
    status = NXS_DeviceNotAvailable;
    return nullptr;
  }
  auto *lib = library_pool.get_new(device, path);
  if (!lib) {
    status = NXS_OutOfHostMemory;
    return nullptr;
  }
  status = data ? lib->load(data, size) : lib->loadFile();
  if (nxs_failed(status)) {
    lib->unload();
    library_pool.release(lib);
    return nullptr;
  }
  return lib;
}

bool NeuronRuntime::forgetObject(nxs_int id) {
  auto obj = getObject(id);
  if (!obj || !*obj) return false;
  (*obj)->release(nullptr);  // not owned: nothing to free, the entry is cleared
  return dropObject(id);
}

// Commands still using the kernel are detached from it (they then fail to
// run with NXS_InvalidKernel) and the kernel's ids dropped (they then resolve
// to no kernel) before the model is unloaded.
nxs_status NeuronRuntime::releaseLibrary(nxs_int library_id) {
  auto lib = get<NeuronLibrary>(library_id);
  if (!lib) return NXS_InvalidLibrary;
  drainQueue(lib->getDevice());
  std::vector<nxs_int> kernels;
  {
    std::lock_guard<std::mutex> lock(command_mutex);
    for (auto *cmd : live_commands)
      if (cmd->getLibrary() == lib) {
        NXSLOG_WARN("releaseLibrary {}: command {} still uses its kernel; it can no longer run",
                    library_id, cmd->getId());
        cmd->detachKernel();
      }
    auto it = kernel_ids.find(lib);
    if (it != kernel_ids.end()) {
      kernels = std::move(it->second);
      kernel_ids.erase(it);
    }
  }
  for (nxs_int id : kernels) forgetObject(id);
  lib->unload();
  library_pool.release(lib);
  if (!dropObject(library_id)) return NXS_InvalidLibrary;
  return NXS_Success;
}

nxs_int NeuronRuntime::addKernel(NeuronKernel *kernel) {
  nxs_int id = addObject(kernel);
  if (id < 0) return id;
  std::lock_guard<std::mutex> lock(command_mutex);
  kernel_ids[kernel->getLibrary()].push_back(id);
  return id;
}

// Only an id addKernel handed out and not yet dropped is a kernel.
nxs_status NeuronRuntime::releaseKernel(nxs_int kernel_id) {
  {
    std::lock_guard<std::mutex> lock(command_mutex);
    bool found = false;
    for (auto &[lib, ids] : kernel_ids) {
      auto it = std::find(ids.begin(), ids.end(), kernel_id);
      if (it == ids.end()) continue;
      ids.erase(it);
      found = true;
      break;
    }
    if (!found) return NXS_InvalidKernel;
  }
  if (!forgetObject(kernel_id)) return NXS_InvalidKernel;
  return NXS_Success;
}

NeuronCommand *NeuronRuntime::getCommand(NeuronKernel *kernel, nxs_uint settings) {
  return command_pool.get_new(kernel, settings);
}

NeuronCommand *NeuronRuntime::getCommand(NeuronEvent *event, nxs_command_type type,
                                         nxs_int value, nxs_uint settings) {
  auto *command = command_pool.get_new(event, type, value, settings);
  if (!command) return nullptr;
  std::lock_guard<std::mutex> lock(command_mutex);
  command_events[command] = event;
  return command;
}

nxs_int NeuronRuntime::addCommand(NeuronSchedule *schedule, NeuronCommand *command) {
  nxs_int id = addObject(command);
  command->setId(id);
  schedule->addCommand(command);
  std::lock_guard<std::mutex> lock(command_mutex);
  live_commands.insert(command);
  return id;
}

nxs_int NeuronRuntime::getSchedule(NeuronDevice *device, nxs_uint settings) {
  auto schedule = schedule_pool.get_new(device, settings);
  if (!schedule) return NXS_InvalidSchedule;
  return addObject(schedule);
}

// Its commands go with it (the API has no command release): their requests
// are waited for, then their tensor sets destroyed.
nxs_status NeuronRuntime::releaseSchedule(nxs_int schedule_id) {
  auto sched = get<NeuronSchedule>(schedule_id);
  if (!sched) return NXS_InvalidSchedule;
  drainQueue(sched->getDevice());
  for (auto *cmd : sched->getCommands()) {
    {
      std::lock_guard<std::mutex> lock(command_mutex);
      live_commands.erase(cmd);
      command_events.erase(cmd);
    }
    cmd->release();
    nxs_int id = cmd->getId();
    command_pool.release(cmd);
    if (id >= 0) dropObject(id);
  }
  sched->release();
  schedule_pool.release(sched);
  if (!dropObject(schedule_id)) return NXS_InvalidSchedule;
  return NXS_Success;
}

nxs_int NeuronRuntime::getEvent(NeuronDevice *device, nxs_event_type type) {
  auto *event = event_pool.get_new(device, type);
  if (!event) return NXS_InvalidEvent;
  event->reset();
  return addObject(event);
}

// Signal and wait commands naming the event are detached from it (they then
// fail to run with NXS_InvalidEvent), so none signals or waits on the event a
// later nxsCreateEvent places in the same pool slot.
nxs_status NeuronRuntime::releaseEvent(nxs_int event_id) {
  auto *event = get<NeuronEvent>(event_id);
  if (!event) return NXS_InvalidEvent;
  {
    std::lock_guard<std::mutex> lock(command_mutex);
    for (auto it = command_events.begin(); it != command_events.end();) {
      if (it->second != event) {
        ++it;
        continue;
      }
      NXSLOG_WARN("releaseEvent {}: command {} still uses it; it can no longer run", event_id,
                  it->first->getId());
      it->first->release();  // an event command has no sets: this drops the event
      it = command_events.erase(it);
    }
  }
  event->reset();
  event_pool.release(event);
  if (!dropObject(event_id)) return NXS_InvalidEvent;
  return NXS_Success;
}

nxs_int NeuronRuntime::getStream(NeuronDevice *device, nxs_uint settings) {
  auto *stream = stream_pool.get_new(device, settings);
  if (!stream) return NXS_InvalidStream;
  return addObject(stream);
}

nxs_status NeuronRuntime::releaseStream(nxs_int stream_id) {
  auto *stream = get<NeuronStream>(stream_id);
  if (!stream || !isStream(stream)) return NXS_InvalidStream;
  stream_pool.release(stream);
  if (!dropObject(stream_id)) return NXS_InvalidStream;
  return NXS_Success;
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

  switch (runtime_property_id) {
    case NP_Keys: {
      std::vector<nxs_long> keys = {NP_Name, NP_Type, NP_Vendor, NP_Size};
      if (!rt->getVersion().empty()) keys.push_back(NP_Version);
      if (!rt->getLinks().empty())
        keys.insert(keys.end(), rt::fabricKeys().begin(), rt::fabricKeys().end());
      return rt::getPropertyVec(property_value, property_value_size, keys.data(), keys.size());
    }
    case NP_Name:
      return rt::getPropertyStr(property_value, property_value_size, "neuron");
    case NP_Size:
      return rt::getPropertyInt(property_value, property_value_size, rt->getNumDevices());
    case NP_Vendor:
      return rt::getPropertyStr(property_value, property_value_size, "AWS");
    case NP_Type:
      return rt::getPropertyStr(property_value, property_value_size, "npu");
    case NP_Version:
      if (rt->getVersion().empty()) return NXS_InvalidProperty;
      return rt::getPropertyStr(property_value, property_value_size, rt->getVersion());
    default:
      // The NeuronLink table (README.md "Links"); PCIe links are the core's.
      if (rt::isFabricProperty(runtime_property_id))
        return rt::getFabricProperty(runtime_property_id, rt->getLinks(), property_value,
                                     property_value_size);
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
  auto dev = rt->getDevice(device_id);
  if (!dev) return NXS_InvalidDevice;

  switch (device_property_id) {
    case NP_Keys: {
      std::vector<nxs_long> keys = {NP_Vendor, NP_Type};
      if (!dev->name.empty() || !rt->getArch().empty()) keys.push_back(NP_Name);
      if (!rt->getArch().empty()) keys.push_back(NP_Architecture);
      if (dev->lnc > 0) keys.push_back(NP_Size);
      if (!dev->bus_id.empty()) keys.push_back(NP_PciBusId);
      return rt::getPropertyVec(property_value, property_value_size, keys.data(), keys.size());
    }
    // The chip's name from sysfs ("Trainium2"), else the architecture.
    case NP_Name: {
      const std::string &name = dev->name.empty() ? rt->getArch() : dev->name;
      if (name.empty()) return NXS_InvalidProperty;
      return rt::getPropertyStr(property_value, property_value_size, name);
    }
    // Vendor, Type and Architecture form the device_lib tag, "aws-npu-trn2".
    case NP_Vendor:
      return rt::getPropertyStr(property_value, property_value_size, "aws");
    case NP_Type:
      return rt::getPropertyStr(property_value, property_value_size, "npu");
    case NP_Architecture:
      if (rt->getArch().empty()) return NXS_InvalidProperty;
      return rt::getPropertyStr(property_value, property_value_size, rt->getArch());
    // The physical NeuronCores this logical core fuses (the LNC degree).
    case NP_Size:
      if (dev->lnc <= 0) return NXS_InvalidProperty;
      return rt::getPropertyInt(property_value, property_value_size, dev->lnc);
    // The chip's PCI identity (docs/Python_API.md "Host and link
    // properties"); every core of a chip states the same one.
    case NP_PciBusId:
      if (dev->bus_id.empty()) return NXS_InvalidProperty;
      return rt::getPropertyStr(property_value, property_value_size, dev->bus_id);
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
                                                void *host_ptr, nxs_uint settings) {
  auto rt = getRuntime();
  auto dev = rt->getDevice(device_id);
  if (!dev) return NXS_InvalidDevice;

  NXSLOG_INFO("createBuffer {}", shape.rank);
  nxs_status status = NXS_Success;
  auto *buf = rt->getBuffer(dev, shape, host_ptr, settings, status);
  if (!buf) return status;

  return rt->addObject(buf);
}

/************************************************************************
 * @def GetBufferProperty
 * @brief Return Buffer properties
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsGetBufferProperty(nxs_int buffer_id,
                                                        nxs_uint buffer_property_id,
                                                        void *property_value,
                                                        size_t *property_value_size) {
  auto rt = getRuntime();
  auto buffer = rt->get<NeuronBuffer>(buffer_id);
  if (!buffer) return NXS_InvalidBuffer;

  switch (buffer_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Size, NP_Value};
      return rt::getPropertyVec(property_value, property_value_size, keys, 2);
    }
    case NP_Size:
      return rt::getPropertyInt(property_value, property_value_size,
                                (nxs_long)buffer->getSizeBytes());
    case NP_Value:
      return rt::getPropertyInt(property_value, property_value_size,
                                reinterpret_cast<nxs_long>(buffer->getAddress()));
  }
  return NXS_InvalidProperty;
}

/************************************************************************
 * @def CopyBuffer
 * @brief Copy a buffer to or from the host
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsCopyBuffer(nxs_int buffer_id, void *host_ptr,
                                                 nxs_uint settings) {
  auto rt = getRuntime();
  auto buf = rt->get<NeuronBuffer>(buffer_id);
  if (!buf) return NXS_InvalidBuffer;
  if (!host_ptr) return NXS_InvalidHostPtr;

  if (settings == NXS_BufferDeviceToHost) return buf->copyToHost(host_ptr);
  return buf->copyToDevice(host_ptr);
}

/************************************************************************
 * @def FillBuffer
 * @brief Fill a buffer with a repeated value
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsFillBuffer(nxs_int buffer_id, void *value,
                                                 nxs_uint value_size) {
  auto rt = getRuntime();
  auto buf = rt->get<NeuronBuffer>(buffer_id);
  if (!buf || value_size == 0) return NXS_InvalidBuffer;
  return buf->fill(value, value_size);
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
 * @brief Load a NEFF, given as bytes, onto the device's core
 * @return Negative value is an error status, else the library id.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateLibrary(nxs_int device_id, void *library_data,
                                                 nxs_uint data_size, nxs_uint settings) {
  NXSLOG_INFO("createLibrary {} - {} bytes", device_id, data_size);
  auto rt = getRuntime();
  auto dev = rt->getDevice(device_id);
  if (!dev) return NXS_InvalidDevice;
  if (!library_data || data_size == 0) return NXS_InvalidBinary;

  nxs_status status = NXS_Success;
  auto *lib = rt->getLibrary(dev, "", library_data, data_size, status);
  if (!lib) return status;
  return rt->addObject(lib);
}

/************************************************************************
 * @def CreateLibraryFromFile
 * @brief Load a NEFF file onto the device's core
 * @return Negative value is an error status, else the library id.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateLibraryFromFile(nxs_int device_id,
                                                         const char *library_path,
                                                         nxs_uint settings) {
  NXSLOG_INFO("createLibraryFromFile {} - {}", device_id, library_path ? library_path : "");
  auto rt = getRuntime();
  auto dev = rt->getDevice(device_id);
  if (!dev) return NXS_InvalidDevice;
  if (!library_path || !*library_path) return NXS_InvalidLibrary;

  nxs_status status = NXS_Success;
  auto *lib = rt->getLibrary(dev, library_path, nullptr, 0, status);
  if (!lib) return status;
  return rt->addObject(lib);
}

/************************************************************************
 * @def GetLibraryProperty
 * @brief Return Library properties: Name (the NEFF's path, for one loaded
 *        from a file), Type ("neff"), Size (NEFF bytes), Count (kernels: 1)
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsGetLibraryProperty(nxs_int library_id,
                                                         nxs_uint library_property_id,
                                                         void *property_value,
                                                         size_t *property_value_size) {
  auto rt = getRuntime();
  auto lib = rt->get<NeuronLibrary>(library_id);
  if (!lib || !lib->getModel()) return NXS_InvalidLibrary;

  switch (library_property_id) {
    case NP_Keys: {
      std::vector<nxs_long> keys = {NP_Type, NP_Size, NP_Count};
      if (!lib->getPath().empty()) keys.push_back(NP_Name);
      return rt::getPropertyVec(property_value, property_value_size, keys.data(), keys.size());
    }
    case NP_Name:
      if (lib->getPath().empty()) return NXS_InvalidProperty;
      return rt::getPropertyStr(property_value, property_value_size, lib->getPath());
    case NP_Type:
      return rt::getPropertyStr(property_value, property_value_size, "neff");
    case NP_Size:
      return rt::getPropertyInt(property_value, property_value_size, (nxs_long)lib->getNeffSize());
    case NP_Count:
      return rt::getPropertyInt(property_value, property_value_size,
                                lib->getTensors().empty() ? 0 : 1);
  }
  return NXS_InvalidProperty;
}

/************************************************************************
 * @def ReleaseLibrary
 * @brief Unload the NEFF (nrt_unload)
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsReleaseLibrary(nxs_int library_id) {
  return getRuntime()->releaseLibrary(library_id);
}

/************************************************************************
 * @def GetKernel
 * @brief The library's kernel. A NEFF has one entry point, so any name
 *        finds it; a NEFF with no input or output tensor has none.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsGetKernel(nxs_int library_id, const char *kernel_name) {
  NXSLOG_INFO("getKernel {} - {}", library_id, kernel_name ? kernel_name : "");
  auto rt = getRuntime();
  auto lib = rt->get<NeuronLibrary>(library_id);
  if (!lib || !lib->getModel()) return NXS_InvalidLibrary;
  if (lib->getTensors().empty()) {
    NXSLOG_ERROR("getKernel {}: the NEFF has no input or output tensor, no kernel",
                 kernel_name ? kernel_name : "");
    return NXS_InvalidKernelName;
  }
  auto *kern = lib->getKernel();
  if (kernel_name) kern->setName(kernel_name);
  return rt->addKernel(kern);
}

/************************************************************************
 * @def GetKernelProperty
 * @brief Return Kernel properties, its arguments in argument order
 *        (inputs, then outputs):
 *        - Size: the argument count
 *        - SubUnits: the argument (tensor) names
 *        - DataTypes: their libnrt dtype names ("float32")
 *        - Shape: their shapes, each as its rank then its dimensions
 *        - Description: "a: INPUT float32[128,512] (262144 bytes); ..."
 *        - Name: the name the kernel was first looked up by; Type: "neff"
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsGetKernelProperty(nxs_int kernel_id,
                                                        nxs_uint kernel_property_id,
                                                        void *property_value,
                                                        size_t *property_value_size) {
  auto rt = getRuntime();
  auto kern = rt->get<NeuronKernel>(kernel_id);
  if (!kern || !kern->getLibrary() || !kern->getLibrary()->getModel()) return NXS_InvalidKernel;
  const auto &tensors = kern->getLibrary()->getTensors();

  switch (kernel_property_id) {
    case NP_Keys: {
      std::vector<nxs_long> keys = {NP_Type,      NP_Size,  NP_SubUnits,
                                    NP_DataTypes, NP_Shape, NP_Description};
      if (!kern->getName().empty()) keys.push_back(NP_Name);
      return rt::getPropertyVec(property_value, property_value_size, keys.data(), keys.size());
    }
    case NP_Name:
      if (kern->getName().empty()) return NXS_InvalidProperty;
      return rt::getPropertyStr(property_value, property_value_size, kern->getName());
    case NP_Type:
      return rt::getPropertyStr(property_value, property_value_size, "neff");
    case NP_Size:
      return rt::getPropertyInt(property_value, property_value_size, (nxs_long)tensors.size());
    case NP_SubUnits: {
      std::vector<std::string> names;
      for (const auto &t : tensors) names.push_back(t.name);
      return rt::getPropertyStrVec(property_value, property_value_size, names);
    }
    case NP_DataTypes: {
      std::vector<std::string> types;
      for (const auto &t : tensors) types.push_back(nrtDtypeName(t.dtype));
      return rt::getPropertyStrVec(property_value, property_value_size, types);
    }
    case NP_Shape: {
      std::vector<nxs_long> shapes;
      for (const auto &t : tensors) {
        shapes.push_back((nxs_long)t.shape.size());
        shapes.insert(shapes.end(), t.shape.begin(), t.shape.end());
      }
      return rt::getPropertyVec(property_value, property_value_size, shapes.data(),
                                shapes.size());
    }
    case NP_Description: {
      std::string desc;
      for (const auto &t : tensors) desc += (desc.empty() ? "" : "; ") + t.describe();
      return rt::getPropertyStr(property_value, property_value_size, desc);
    }
  }
  return NXS_InvalidProperty;
}

/************************************************************************
 * @def ReleaseKernel
 * @brief Drop the kernel's id; the model goes with its library
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsReleaseKernel(nxs_int kernel_id) {
  return getRuntime()->releaseKernel(kernel_id);
}

/************************************************************************
 * @def CreateEvent
 * @brief Create an event (neuron_schedule.h): a host-side value that
 *        schedules signal when their queued executions complete
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateEvent(nxs_int device_id, nxs_event_type event_type,
                                               nxs_uint event_settings) {
  auto rt = getRuntime();
  auto dev = rt->getDevice(device_id);
  if (!dev) return NXS_InvalidDevice;
  return rt->getEvent(dev, event_type);
}

/************************************************************************
 * @def GetEventProperty
 * @brief Value: the value last signalled (completed signals only)
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsGetEventProperty(nxs_int event_id,
                                                       nxs_uint event_property_id,
                                                       void *property_value,
                                                       size_t *property_value_size) {
  auto rt = getRuntime();
  auto event = rt->get<NeuronEvent>(event_id);
  if (!event) return NXS_InvalidEvent;
  switch (event_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Value};
      return rt::getPropertyVec(property_value, property_value_size, keys, 1);
    }
    case NP_Value:
      return rt::getPropertyInt(property_value, property_value_size, event->getValue());
  }
  return NXS_InvalidProperty;
}

extern "C" nxs_status NXS_API_CALL nxsSignalEvent(nxs_int event_id, nxs_int signal_value) {
  auto rt = getRuntime();
  auto event = rt->get<NeuronEvent>(event_id);
  if (!event) return NXS_InvalidEvent;
  return event->signal(signal_value);
}

// Waits for the executions the event's pending signals follow: a failed one
// is returned here.
extern "C" nxs_status NXS_API_CALL nxsWaitEvent(nxs_int event_id, nxs_int wait_value) {
  auto rt = getRuntime();
  auto event = rt->get<NeuronEvent>(event_id);
  if (!event) return NXS_InvalidEvent;
  return event->wait(wait_value);
}

extern "C" nxs_status NXS_API_CALL nxsReleaseEvent(nxs_int event_id) {
  return getRuntime()->releaseEvent(event_id);
}

/************************************************************************
 * @def CreateStream
 * @brief A stream of the device: its core's one compute queue
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateStream(nxs_int device_id, nxs_uint stream_settings) {
  auto rt = getRuntime();
  auto dev = rt->getDevice(device_id);
  if (!dev) return NXS_InvalidDevice;
  return rt->getStream(dev, stream_settings);
}

/************************************************************************
 * @def GetStreamProperty
 * @brief Value: the libnrt queue index (always 0)
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsGetStreamProperty(nxs_int stream_id,
                                                        nxs_uint stream_property_id,
                                                        void *property_value,
                                                        size_t *property_value_size) {
  auto rt = getRuntime();
  auto stream = rt->get<NeuronStream>(stream_id);
  if (!stream || !rt->isStream(stream)) return NXS_InvalidStream;
  switch (stream_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Value};
      return rt::getPropertyVec(property_value, property_value_size, keys, 1);
    }
    case NP_Value:
      return rt::getPropertyInt(property_value, property_value_size, 0);
  }
  return NXS_InvalidProperty;
}

extern "C" nxs_status NXS_API_CALL nxsReleaseStream(nxs_int stream_id) {
  return getRuntime()->releaseStream(stream_id);
}

/************************************************************************
 * @def CreateSchedule
 * @brief Create a schedule on the device
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
 * @brief ElapsedTime: host time of the last run with
 *        NXS_ExecutionSettings_Timing (ms)
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsGetScheduleProperty(nxs_int schedule_id,
                                                          nxs_uint schedule_property_id,
                                                          void *property_value,
                                                          size_t *property_value_size) {
  auto rt = getRuntime();
  auto schedule = rt->get<NeuronSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;
  switch (schedule_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_ElapsedTime};
      return rt::getPropertyVec(property_value, property_value_size, keys, 1);
    }
    case NP_ElapsedTime:
      return rt::getPropertyFlt(property_value, property_value_size, schedule->getTime());
  }
  return NXS_InvalidProperty;
}

/************************************************************************
 * @def RunSchedule
 * @brief Run the schedule on its device's queue: blocking (nrt_execute) or
 *        with NXS_ExecutionSettings_NonBlocking enqueued
 *        (nrta_execute_schedule). Any stream of the device is that queue.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsRunSchedule(nxs_int schedule_id, nxs_int stream_id,
                                                  nxs_uint run_settings) {
  auto rt = getRuntime();
  auto schedule = rt->get<NeuronSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;
  if (stream_id >= 0) {
    auto *stream = rt->get<NeuronStream>(stream_id);
    if (stream && rt->isStream(stream) && stream->device != schedule->getDevice()) {
      NXSLOG_ERROR("runSchedule {}: stream {} is on another device", schedule_id, stream_id);
      return NXS_InvalidStream;
    }
  }
  return schedule->run(stream_id, run_settings);
}

extern "C" nxs_status NXS_API_CALL nxsReleaseSchedule(nxs_int schedule_id) {
  return getRuntime()->releaseSchedule(schedule_id);
}

/************************************************************************
 * @def CreateCommand
 * @brief A kernel execution in the schedule; the kernel's NEFF must be
 *        loaded on the schedule's core
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateCommand(nxs_int schedule_id, nxs_int kernel_id,
                                                 nxs_uint settings) {
  auto rt = getRuntime();
  auto schedule = rt->get<NeuronSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;
  auto kernel = rt->get<NeuronKernel>(kernel_id);
  if (!kernel || !kernel->getLibrary() || !kernel->getLibrary()->getModel())
    return NXS_InvalidKernel;
  if (kernel->getLibrary()->getDevice() != schedule->getDevice()) {
    NXSLOG_ERROR("createCommand: the kernel is loaded on vnc {}, the schedule is on vnc {}",
                 kernel->getLibrary()->getDevice()->vnc, schedule->getDevice()->vnc);
    return NXS_InvalidKernel;
  }
  auto *command = rt->getCommand(kernel, settings);
  if (!command) return NXS_OutOfHostMemory;
  return rt->addCommand(schedule, command);
}

extern "C" nxs_int NXS_API_CALL nxsCreateSignalCommand(nxs_int schedule_id, nxs_int event_id,
                                                       nxs_int signal_value,
                                                       nxs_uint settings) {
  NXSLOG_INFO("createSignalCommand {} - {} - {}", schedule_id, event_id, signal_value);
  auto rt = getRuntime();
  auto schedule = rt->get<NeuronSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;
  auto event = rt->get<NeuronEvent>(event_id);
  if (!event) return NXS_InvalidEvent;
  auto *command = rt->getCommand(event, NXS_CommandType_Signal, signal_value, settings);
  if (!command) return NXS_OutOfHostMemory;
  return rt->addCommand(schedule, command);
}

extern "C" nxs_int NXS_API_CALL nxsCreateWaitCommand(nxs_int schedule_id, nxs_int event_id,
                                                     nxs_int wait_value, nxs_uint settings) {
  NXSLOG_INFO("createWaitCommand {} - {} - {}", schedule_id, event_id, wait_value);
  auto rt = getRuntime();
  auto schedule = rt->get<NeuronSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;
  auto event = rt->get<NeuronEvent>(event_id);
  if (!event) return NXS_InvalidEvent;
  auto *command = rt->getCommand(event, NXS_CommandType_Wait, wait_value, settings);
  if (!command) return NXS_OutOfHostMemory;
  return rt->addCommand(schedule, command);
}

/************************************************************************
 * @def GetCommandProperty
 * @brief ElapsedTime: nrt_execute host time of a blocking, timed run (ms)
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsGetCommandProperty(nxs_int command_id,
                                                         nxs_uint command_property_id,
                                                         void *property_value,
                                                         size_t *property_value_size) {
  auto rt = getRuntime();
  auto command = rt->get<NeuronCommand>(command_id);
  if (!command) return NXS_InvalidCommand;
  switch (command_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_ElapsedTime};
      return rt::getPropertyVec(property_value, property_value_size, keys, 1);
    }
    case NP_ElapsedTime:
      return rt::getPropertyFlt(property_value, property_value_size, command->getTime());
  }
  return NXS_InvalidProperty;
}

/************************************************************************
 * @def SetCommandArgument
 * @brief Bind a buffer to a kernel tensor: by `name` where one is given,
 *        else argument `argument_index` (inputs, then outputs)
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsSetCommandArgument(nxs_int command_id,
                                                         nxs_int argument_index,
                                                         nxs_int buffer_id, const char *name,
                                                         nxs_uint argument_settings) {
  auto rt = getRuntime();
  auto command = rt->get<NeuronCommand>(command_id);
  if (!command || command->getType() != NXS_CommandType_Dispatch) return NXS_InvalidCommand;
  auto buffer = rt->get<NeuronBuffer>(buffer_id);
  if (!buffer) return NXS_InvalidBuffer;
  return command->bind(argument_index, buffer, name);
}

/************************************************************************
 * @def SetCommandScalar
 * @brief Not supported: a NEFF takes device tensors only. Pass the value in
 *        a buffer bound to the tensor that carries it.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsSetCommandScalar(nxs_int command_id,
                                                       nxs_int argument_index, void *value,
                                                       const char *name,
                                                       nxs_uint argument_settings) {
  NXSLOG_ERROR("setCommandScalar {} argument {}: a NEFF takes tensors only, bind a buffer",
               command_id, argument_index);
  return NXS_InvalidArgValue;
}

/************************************************************************
 * @def FinalizeCommand
 * @brief Check that every kernel tensor is bound and build the command's
 *        tensor sets; grid, group and shared memory size do not apply
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsFinalizeCommand(nxs_int command_id, nxs_dim3 grid_size,
                                                      nxs_dim3 group_size,
                                                      nxs_uint shared_memory_size) {
  auto rt = getRuntime();
  auto command = rt->get<NeuronCommand>(command_id);
  if (!command) return NXS_InvalidCommand;
  return command->finalize(grid_size, group_size, shared_memory_size);
}
