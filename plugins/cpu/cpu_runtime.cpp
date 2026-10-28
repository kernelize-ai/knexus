#include "cpu_runtime.h"

#include <assert.h>
#include <dlfcn.h>
#include <knexus-api.h>
#include <rt_buffer.h>
#include <rt_object.h>
#include <rt_topology.h>
#include <rt_runtime.h>
#include <rt_utilities.h>
#include <string.h>

#include <functional>
#include <magic_enum/magic_enum.hpp>
#include <algorithm>
#include <optional>
#include <string>
#include <vector>

using namespace nxs;

namespace {

// The host as hwloc states it (rt_topology.h, no I/O objects): packages,
// cores, PUs and NUMA nodes, read once per process. Nothing is measured; the
// names and units are those tabled in docs/Python_API.md under "Host and link
// properties". found is false only when hwloc fails to load or states no PU,
// and then the plugin states none of the host properties.
struct HostTopology {
  std::string name;                  // the first package's CPUModel, trimmed
  nxs_long memory_bytes = -1;        // the root's total_memory, -1 if none
  nxs_long sockets = 0, cores = 0, threads = 0, numa_nodes = 0;
  std::vector<nxs_long> thread_socket, thread_core, thread_numa_node;  // by PU os_index
  std::vector<nxs_long> numa_memory_bytes;  // by node, os_index ascending
  std::vector<nxs_long> numa_distance;      // row-major numa_nodes^2, empty if none
  std::vector<std::string> numa_cpulist;    // by node, kernel cpulist syntax
  bool found = false;
};

HostTopology readHostTopology(hwloc_topology_t topo) {
  HostTopology h;
  if (!topo) return h;
  int npu = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PU);
  if (npu <= 0) return h;
  h.found = true;
  h.sockets = std::max(0, hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PACKAGE));
  h.cores = std::max(0, hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE));
  h.threads = npu;
  int nnode = std::max(0, hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE));
  h.numa_nodes = nnode;

  // Name and memory: absent on a topology that does not state them (synthetic)
  if (auto *pkg = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PACKAGE, 0)) {
    if (const char *model = hwloc_obj_get_info_by_name(pkg, "CPUModel")) {
      std::string v(model);
      v.erase(0, v.find_first_not_of(" \t"));
      v.erase(v.find_last_not_of(" \t\r\n") + 1);
      h.name = v;
    }
  }
  if (auto *root = hwloc_get_root_obj(topo); root && root->total_memory > 0)
    h.memory_bytes = (nxs_long)root->total_memory;

  // the Thread* vectors, indexed by PU os_index
  unsigned max_pu = 0;
  for (int i = 0; i < npu; ++i)
    max_pu = std::max(max_pu, hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, i)->os_index);
  h.thread_socket.assign(max_pu + 1, -1);
  h.thread_core.assign(max_pu + 1, -1);
  h.thread_numa_node.assign(max_pu + 1, -1);
  for (int i = 0; i < npu; ++i) {
    hwloc_obj_t pu = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, i);
    unsigned n = pu->os_index;
    if (auto *pkg = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_PACKAGE, pu))
      h.thread_socket[n] = pkg->os_index;
    if (auto *core = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_CORE, pu))
      h.thread_core[n] = core->os_index;
    if (pu->nodeset) {
      int first = hwloc_bitmap_first(pu->nodeset);
      if (first >= 0) h.thread_numa_node[n] = first;
    }
  }

  // NUMA nodes in os_index order; the distance matrix in the same order
  std::vector<hwloc_obj_t> nodes;
  for (int i = 0; i < nnode; ++i)
    nodes.push_back(hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, i));
  std::sort(nodes.begin(), nodes.end(),
            [](hwloc_obj_t a, hwloc_obj_t b) { return a->os_index < b->os_index; });
  for (auto *node : nodes) {
    h.numa_memory_bytes.push_back(node->attr && node->attr->numanode.local_memory
                                      ? (nxs_long)node->attr->numanode.local_memory
                                      : -1);
    std::string list;
    if (node->cpuset) {
      int len = hwloc_bitmap_list_snprintf(nullptr, 0, node->cpuset);
      if (len > 0) {
        list.resize(len + 1);
        hwloc_bitmap_list_snprintf(list.data(), list.size(), node->cpuset);
        list.resize(len);
      }
    }
    h.numa_cpulist.push_back(list);
  }
  if (nnode == 1) {
    h.numa_distance = {10};  // the kernel's value; hwloc keeps no 1x1 matrix
  } else if (nnode > 1) {
    unsigned nr = 1;
    hwloc_distances_s *dist = nullptr;
    if (hwloc_distances_get_by_type(topo, HWLOC_OBJ_NUMANODE, &nr, &dist,
                                    HWLOC_DISTANCES_KIND_FROM_OS, 0) == 0 &&
        nr >= 1 && dist) {
      // map each matrix object to its place in the os_index order
      std::vector<int> place(dist->nbobjs, -1);
      for (unsigned k = 0; k < dist->nbobjs; ++k)
        for (size_t p = 0; p < nodes.size(); ++p)
          if (dist->objs[k] == nodes[p]) place[k] = (int)p;
      h.numa_distance.assign((size_t)nnode * nnode, -1);
      for (unsigned i = 0; i < dist->nbobjs; ++i)
        for (unsigned j = 0; j < dist->nbobjs; ++j)
          if (place[i] >= 0 && place[j] >= 0)
            h.numa_distance[(size_t)place[i] * nnode + place[j]] =
                (nxs_long)dist->values[i * dist->nbobjs + j];
      hwloc_distances_release(topo, dist);
    }
  }
  return h;
}

const HostTopology &getHostTopology() {
  static const HostTopology s_host =
      readHostTopology(rt::topology(rt::TopologyIo::None));
  return s_host;
}

}  // namespace

CpuRuntime *getRuntime() {
  static CpuRuntime s_runtime;
  return &s_runtime;
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
  auto proc = cpuinfo_get_processor(0);
  auto *arch = cpuinfo_get_uarch(0);
  auto aid = cpuinfo_get_current_uarch_index();

  NXSLOG_TRACE("getRuntimeProperty {}", nxsGetPropName(runtime_property_id));

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
      return rt::getPropertyStr(property_value, property_value_size, "cpu");
    case NP_Size:
      return rt::getPropertyInt(property_value, property_value_size, 1);
    case NP_Vendor: {
      auto name = cpuinfo_vendor_to_string(proc->core->vendor);
      assert(name);
      return rt::getPropertyStr(property_value, property_value_size,
                                name);
    }
    case NP_Type:
      return rt::getPropertyStr(property_value, property_value_size, "cpu");
    case NP_ID: {
      return rt::getPropertyInt(property_value, property_value_size,
                                cpuinfo_has_arm_sme2() ? 1 : 0);
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
  auto dev = getRuntime()->getObject(device_id);
  if (!dev) return NXS_InvalidDevice;
  auto *cpu = cpuinfo_get_processor(device_id);
  auto *arch = cpuinfo_get_uarch(device_id);
  // auto isa = device->core->isa;

  NXSLOG_TRACE("getDeviceProperty {}", nxsGetPropName(device_property_id));

  const HostTopology &host = getHostTopology();

  switch (device_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Name,         NP_Type,          NP_Architecture,
                         NP_Size,         NP_GlobalMemorySize,
                         NP_SocketCount,  NP_CoreCount,     NP_ThreadCount,
                         NP_NumaNodeCount, NP_ThreadSocket, NP_ThreadCore,
                         NP_ThreadNumaNode, NP_NumaNodeMemorySize,
                         NP_NumaDistance,  NP_NumaNodeCpuList};
      size_t count = host.found ? sizeof(keys) / sizeof(keys[0]) : 4;
      return rt::getPropertyVec(property_value, property_value_size, keys, count);
    }
    case NP_Name:
      // the package's CPUModel where hwloc states one, else "cpu"
      return rt::getPropertyStr(property_value, property_value_size,
                                host.name.empty() ? std::string("cpu") : host.name);
    case NP_Type:
      return rt::getPropertyStr(property_value, property_value_size, "cpu");
    case NP_GlobalMemorySize:
      if (host.memory_bytes < 0) return NXS_InvalidProperty;
      return rt::getPropertyInt(property_value, property_value_size, host.memory_bytes);
    // the host: packages, cores, PUs and NUMA nodes as hwloc states them,
    // absent when no topology was loaded
    case NP_SocketCount:
      if (!host.found) return NXS_InvalidProperty;
      return rt::getPropertyInt(property_value, property_value_size, host.sockets);
    case NP_CoreCount:
      if (!host.found) return NXS_InvalidProperty;
      return rt::getPropertyInt(property_value, property_value_size, host.cores);
    case NP_ThreadCount:
      if (!host.found) return NXS_InvalidProperty;
      return rt::getPropertyInt(property_value, property_value_size, host.threads);
    case NP_NumaNodeCount:
      if (!host.found) return NXS_InvalidProperty;
      return rt::getPropertyInt(property_value, property_value_size, host.numa_nodes);
    case NP_ThreadSocket:
      if (!host.found) return NXS_InvalidProperty;
      return rt::getPropertyVec(property_value, property_value_size,
                                host.thread_socket.data(), host.thread_socket.size());
    case NP_ThreadCore:
      if (!host.found) return NXS_InvalidProperty;
      return rt::getPropertyVec(property_value, property_value_size,
                                host.thread_core.data(), host.thread_core.size());
    case NP_ThreadNumaNode:
      if (!host.found) return NXS_InvalidProperty;
      return rt::getPropertyVec(property_value, property_value_size,
                                host.thread_numa_node.data(), host.thread_numa_node.size());
    case NP_NumaNodeMemorySize:
      if (!host.found) return NXS_InvalidProperty;
      return rt::getPropertyVec(property_value, property_value_size,
                                host.numa_memory_bytes.data(), host.numa_memory_bytes.size());
    case NP_NumaDistance:
      // absent where hwloc holds no FROM_OS matrix (a synthetic topology)
      if (!host.found || host.numa_distance.empty()) return NXS_InvalidProperty;
      return rt::getPropertyVec(property_value, property_value_size,
                                host.numa_distance.data(), host.numa_distance.size());
    case NP_NumaNodeCpuList:
      if (!host.found) return NXS_InvalidProperty;
      return rt::getPropertyStrVec(property_value, property_value_size, host.numa_cpulist);
    case NP_Architecture: {
      auto archName = cpuinfo_uarch_to_string(cpu->core->uarch);
      assert(archName);
      return rt::getPropertyStr(property_value, property_value_size,
                                archName);
    }
    case NP_Size:
      return rt::getPropertyInt(property_value, property_value_size,
                                cpuinfo_get_processors_count());

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
  auto dev = rt->getObject(device_id);
  if (!dev) return NXS_InvalidDevice;

  NXSLOG_TRACE("createBuffer {}", shape.rank);
  if (!(settings & NXS_BufferSettings_OnDevice)) {
    settings |= NXS_BufferSettings_Maintain | NXS_BufferSettings_OnDevice;
  }
  auto *buf = rt->getBuffer(shape, host_ptr, settings);
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
  auto buffer = rt->getObject(buffer_id);
  if (!buffer) return NXS_InvalidBuffer;
  auto bufObj = (*buffer)->get<rt::Buffer>();

  switch (buffer_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Name, NP_Type, NP_Shape, NP_Size, NP_Value};
      return rt::getPropertyVec(property_value, property_value_size, keys, 5);
    }
    case NP_Value:
      return rt::getPropertyInt(property_value, property_value_size,
                                reinterpret_cast<nxs_long>(bufObj->getData()));
  }
  return NXS_InvalidProperty;
}

/************************************************************************
 * @def CopyBuffer
 * @brief Copy a buffer to/from the host based on the settings
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsCopyBuffer(nxs_int buffer_id,
                                                 void *host_ptr,
                                                 nxs_uint settings) {
  auto rt = getRuntime();
  auto buf = rt->getObject(buffer_id);
  if (!buf) return NXS_InvalidBuffer;
  auto bufObj = (*buf)->get<rt::Buffer>();
  if (settings & NXS_BufferHostToDevice) {
    std::memcpy(bufObj->data(), host_ptr, bufObj->getSizeBytes());
  } else {
    std::memcpy(host_ptr, bufObj->data(), bufObj->getSizeBytes());
  }
  return NXS_Success;
}

/************************************************************************
 * @def FillBuffer
 * @brief Fill a buffer with a value
 * @return Error status or Succes.
 ***********************************************************************/
 extern "C" nxs_status NXS_API_CALL nxsFillBuffer(nxs_int buffer_id,
                                                  void *value,
                                                  nxs_uint value_size) {
  auto rt = getRuntime();
  auto buffer = rt->get<rt::Buffer>(buffer_id);
  if (!buffer || value_size == 0) return NXS_InvalidBuffer;

  size_t total_size = buffer->getSizeBytes();

  bool is_zero = true;
  for (nxs_uint i = 0; i < value_size; ++i) {
    if (*((unsigned char*)value + i) != 0) {
      is_zero = false;
      break;
    }
  }

  char *buffer_data = buffer->data();
  if (is_zero || value_size == 1) {
    std::memset(buffer_data, *((unsigned char*)value), total_size);
    return NXS_Success;
  }

  assert(total_size % value_size == 0 && "Buffer size must be aligned with value size");
  if (value_size == 2) {
    uint16_t val = *((uint16_t*)value);
    auto* ptr = reinterpret_cast<uint16_t*>(buffer_data);
    std::fill(ptr, ptr + total_size / 2, val);
  } else if (value_size == 4) {
    uint32_t val = *((uint32_t*)value);
    auto* ptr = reinterpret_cast<uint32_t*>(buffer_data);
    std::fill(ptr, ptr + total_size / 4, val);
  } else {
    assert(false && "Unsupported value size");
  }

  return NXS_Success;
}

/************************************************************************
 * @def ReleaseBuffer
 * @brief Release a buffer on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsReleaseBuffer(nxs_int buffer_id) {
  auto rt = getRuntime();
  NXSLOG_TRACE("releaseBuffer {}", buffer_id);
  return rt->releaseBuffer(buffer_id);
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
  auto rt = getRuntime();
  auto dev = rt->getObject(device_id);
  if (!dev) return NXS_InvalidDevice;

  // #include <sys/mman.h>
  // #include <dlfcn.h>
  // #include <unistd.h>
  // int fd = memfd_create("my_lib", MFD_CLOEXEC);
  // write(fd, library_data, library_size);
  // char fd_path[64];
  // snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", fd);
  // void *handle = dlopen(fd_path, RTLD_NOW);
  // return rt->addObject(lib);
  return NXS_InvalidLibrary;
}

/************************************************************************
 * @def CreateLibraryFromFile
 * @brief Create a library from a file
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateLibraryFromFile(
    nxs_int device_id, const char *library_path, nxs_uint settings) {
  NXSLOG_TRACE("createLibraryFromFile {} - {}", device_id, library_path);
  auto rt = getRuntime();
  auto dev = rt->getObject(device_id);
  if (!dev) return NXS_InvalidDevice;

  void *lib = dlopen(library_path, RTLD_NOW);
  if (!lib) {
    NXSLOG_ERROR("createLibraryFromFile {}", dlerror());
    return NXS_InvalidLibrary;
  }
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
  dlclose((*lib)->get<void>());
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
  auto lib = rt->getObject(library_id);
  if (!lib) return NXS_InvalidProgram;
  void *func = dlsym((*lib)->get<void>(), kernel_name);
  if (!func) {
    NXSLOG_ERROR("getKernel {}", dlerror());
    return NXS_InvalidKernel;
  }
  return rt->addObject(func);
}

/************************************************************************
 * @def GetKernelProperty
 * @brief Return Kernel properties
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL
nxsGetKernelProperty(nxs_int kernel_id, nxs_uint kernel_property_id,
                     void *property_value, size_t *property_value_size) {
  auto rt = getRuntime();
  auto func = rt->getObject(kernel_id);
  if (!func) return NXS_InvalidKernel;

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

  // 1 CPU, with many cores and 1 thread per core
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
  auto dev = rt->getObject(device_id);
  if (!dev) return NXS_InvalidDevice;

  return rt->getSchedule(device_id, schedule_settings);
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
  auto schedule = rt->get<CpuSchedule>(schedule_id);
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
  auto schedule = rt->get<CpuSchedule>(schedule_id);
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
  auto schedule = rt->get<CpuSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;
  auto kernel_v = rt->get<void>(kernel_id);
  if (!kernel_v) return NXS_InvalidKernel;
  auto kernel = reinterpret_cast<cpuFunction_t>(kernel_v);

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
                                                         nxs_uint arg_settings) {
  auto rt = getRuntime();

  auto command = rt->get<CpuCommand>(command_id);
  if (!command) return NXS_InvalidCommand;

  auto buffer = rt->get<rt::Buffer>(buffer_id);
  if (!buffer) return NXS_InvalidBuffer;

  return command->setArgument(argument_index, buffer, name, arg_settings);
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
                                                       nxs_uint arg_settings) {
  auto rt = getRuntime();
  auto command = rt->get<CpuCommand>(command_id);
  if (!command) return NXS_InvalidCommand;
  return command->setScalar(argument_index, value, name, arg_settings);
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

  auto command = rt->get<CpuCommand>(command_id);
  if (!command) return NXS_InvalidCommand;

  return command->finalize(grid_size, group_size, shared_memory_size);
}
