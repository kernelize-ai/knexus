#include <cuda_utils.h>

#include <assert.h>
#include <cuda_command.h>
#include <cuda_device.h>
#include <cuda_plugin_runtime.h>
#include <cuda_runtime.h>
#include <rt_buffer.h>
#include <rt_object.h>
#include <rt_utilities.h>
#include <string.h>

#include <string>
#include <vector>

#include <nvml.h>

using namespace nxs;

// One link of a device, as the Link* properties state it: a row per link, the
// vectors indexed by row. The cuda plugin states three kinds of row —
//   PCIe to the host     (peer -1): generation and width, max and current;
//   NVLink to a peer     (one row per active NVML link): its version as
//                         generation, the remote's bus id, and the peer index
//                         where the remote is a device of this runtime;
//   PCIe path to a peer  (one row per other device): NVML's common-ancestor
//                         level — 0 internal, 10 a single switch, 20 several,
//                         30 a host bridge, 40 a NUMA node, 50 the system.
// -1 and "" mean the row states no such figure. A plugin for another vendor
// states the same columns with its own kinds.
struct DeviceLink {
  std::string type;
  nxs_long peer = -1;
  std::string peer_id;
  nxs_long gen_max = -1, gen_cur = -1, width_max = -1, width_cur = -1;
  nxs_long level = -1;
};

static std::vector<DeviceLink> deviceLinks(nxs_int device_id, nxs_int count) {
  std::vector<DeviceLink> links;
  std::vector<nvmlDevice_t> handles(count);
  std::vector<std::string> bus_ids(count);
  for (nxs_int j = 0; j < count; ++j) {
    nvmlPciInfo_t pci;
    if (nvmlDeviceGetHandleByIndex(j, &handles[j]) != NVML_SUCCESS) return links;
    if (nvmlDeviceGetPciInfo(handles[j], &pci) == NVML_SUCCESS) bus_ids[j] = pci.busId;
  }
  nvmlDevice_t self = handles[device_id];

  DeviceLink host;
  host.type = "PCIe";
  unsigned int v = 0;
  if (nvmlDeviceGetMaxPcieLinkGeneration(self, &v) == NVML_SUCCESS) host.gen_max = v;
  if (nvmlDeviceGetCurrPcieLinkGeneration(self, &v) == NVML_SUCCESS) host.gen_cur = v;
  if (nvmlDeviceGetMaxPcieLinkWidth(self, &v) == NVML_SUCCESS) host.width_max = v;
  if (nvmlDeviceGetCurrPcieLinkWidth(self, &v) == NVML_SUCCESS) host.width_cur = v;
  links.push_back(host);

  for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link) {
    nvmlEnableState_t active;
    nvmlReturn_t rc = nvmlDeviceGetNvLinkState(self, link, &active);
    if (rc == NVML_ERROR_NOT_SUPPORTED || rc == NVML_ERROR_INVALID_ARGUMENT) break;
    if (rc != NVML_SUCCESS || active != NVML_FEATURE_ENABLED) continue;
    DeviceLink nv;
    nv.type = "NVLink";
    nvmlPciInfo_t remote;
    if (nvmlDeviceGetNvLinkRemotePciInfo(self, link, &remote) == NVML_SUCCESS) {
      nv.peer_id = remote.busId;
      for (nxs_int j = 0; j < count; ++j)
        if (j != device_id && !bus_ids[j].empty() && bus_ids[j] == remote.busId) nv.peer = j;
    }
    unsigned int version = 0;
    if (nvmlDeviceGetNvLinkVersion(self, link, &version) == NVML_SUCCESS)
      nv.gen_max = nv.gen_cur = version;
    links.push_back(nv);
  }

  for (nxs_int j = 0; j < count; ++j) {
    if (j == device_id) continue;
    DeviceLink path;
    path.type = "PCIe";
    path.peer = j;
    path.peer_id = bus_ids[j];
    nvmlGpuTopologyLevel_t level;
    if (nvmlDeviceGetTopologyCommonAncestor(self, handles[j], &level) == NVML_SUCCESS)
      path.level = (nxs_long)level;
    links.push_back(path);
  }
  return links;
}

CudaRuntime *getRuntime() {
  static CudaRuntime s_runtime;
  return &s_runtime;
}

/*
 * Get the Runtime properties
 */ 
extern "C" nxs_status NXS_API_CALL
nxsGetRuntimeProperty(nxs_uint runtime_property_id, void *property_value,
                      size_t *property_value_size) {
  auto rt = getRuntime();

  int runtime_version = 0;
  CUDA_CHECK(NXS_InvalidRuntime, cudaRuntimeGetVersion, &runtime_version);

  int major_version = runtime_version / 10000000;
  int minor_version = (runtime_version % 10000000) / 100000;
  int patch_version = runtime_version % 100000;

  NXSLOG_TRACE("getRuntimeProperty {}", nxsGetPropName(runtime_property_id));
  /* return value size */
  /* return value */
  switch (runtime_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Name,    NP_Type,         NP_Vendor,
                         NP_Version, NP_MajorVersion, NP_MinorVersion,
                         NP_Size};
      return rt::getPropertyVec(property_value, property_value_size, keys, 7);
    }
    case NP_Name:
      return rt::getPropertyStr(property_value, property_value_size, "cuda");
    case NP_Type:
      return rt::getPropertyStr(property_value, property_value_size, "gpu");
    case NP_Vendor:
      return rt::getPropertyStr(property_value, property_value_size, "NVIDIA");
    case NP_Version: {
      char version[128];
      snprintf(version, 128, "%d.%d.%d", major_version, minor_version,
               patch_version);
      return rt::getPropertyStr(property_value, property_value_size, version);
    }
    case NP_MajorVersion:
      return rt::getPropertyInt(property_value, property_value_size,
                                major_version);
    case NP_MinorVersion:
      return rt::getPropertyInt(property_value, property_value_size,
                                minor_version);
    case NP_Size: {
      nxs_long size = getRuntime()->getDeviceCount();
      return rt::getPropertyInt(property_value, property_value_size, size);
    }
    default:
      return NXS_InvalidProperty;
  }
  return NXS_Success;
}

extern "C" nxs_status NXS_API_CALL
nxsGetDeviceProperty(nxs_int device_id, nxs_uint property_id,
                     void *property_value, size_t *property_value_size) {
  auto *rt = getRuntime();
  auto *device = rt->getDevice(device_id);
  if (!device) return NXS_InvalidDevice;
  CUdevice cuda_device;
  CU_CHECK(NXS_InvalidDevice, cuDeviceGet, &cuda_device, device_id);
 
  cudaDeviceProp &props = device->props;

  NXSLOG_TRACE("getDeviceProperty {}", nxsGetPropName(property_id));

  switch (property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Name,
                         NP_Type,
                         NP_Architecture,
                         NP_MajorVersion,
                         NP_MinorVersion,
                         NP_Size,
                         NP_GlobalMemorySize,
                         NP_CoreMemorySize,
                         NP_CoreRegisterSize,
                         NP_SIMDSize,
                         NP_MemoryBusWidth,
                         NP_CoreUtilization,
                         NP_MemoryUtilization,
                         NP_PciBusId,
                         NP_LinkCount,
                         NP_LinkType,
                         NP_LinkPeer,
                         NP_LinkPeerId,
                         NP_LinkGenMax,
                         NP_LinkGenCurrent,
                         NP_LinkWidthMax,
                         NP_LinkWidthCurrent,
                         NP_LinkTopologyLevel};
      return rt::getPropertyVec(property_value, property_value_size, keys,
                                sizeof(keys) / sizeof(keys[0]));
    }
    case NP_Name:
      return rt::getPropertyStr(property_value, property_value_size, props.name);
    case NP_Type:
      return rt::getPropertyStr(property_value, property_value_size, "gpu");
    case NP_Architecture: {
      std::string name = "sm_" + std::to_string(props.major * 10 + props.minor);
      return rt::getPropertyStr(property_value, property_value_size, name);
    }
    case NP_MajorVersion:
      return rt::getPropertyInt(property_value, property_value_size,
                                props.major);
    case NP_MinorVersion:
      return rt::getPropertyInt(property_value, property_value_size,
                                props.minor);
    case NP_Size:
      return rt::getPropertyInt(property_value, property_value_size,
                                props.multiProcessorCount);
    case NP_GlobalMemorySize:
      return rt::getPropertyInt(property_value, property_value_size,
                                props.totalGlobalMem);
    case NP_CoreMemorySize: {
      int max_shared_mem = 0;
      CU_CHECK(NXS_InvalidDevice, cuDeviceGetAttribute,
          &max_shared_mem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
          cuda_device);
      return rt::getPropertyInt(property_value, property_value_size, max_shared_mem);
    }
    case NP_CoreRegisterSize:
      return rt::getPropertyInt(property_value, property_value_size,
                                props.regsPerBlock);
    case NP_SIMDSize:
      return rt::getPropertyInt(property_value, property_value_size,
                                props.warpSize);
    #if CUDA_VERSION < 13000
    case NP_CoreClockRate:
      return rt::getPropertyInt(property_value, property_value_size,
                                props.clockRate);
    case NP_MemoryClockRate:
      return rt::getPropertyInt(property_value, property_value_size,
                                props.memoryClockRate);
    #endif
    case NP_MemoryBusWidth:
      return rt::getPropertyInt(property_value, property_value_size,
                                props.memoryBusWidth);
    case NP_CoreUtilization: {
      nvmlDevice_t nvml_device;
      nvmlUtilization_t utilization;
      nxs_long value = -1;

      if (nvmlDeviceGetHandleByIndex(device_id, &nvml_device) != NVML_SUCCESS)
        return NXS_InvalidDevice;

      if (nvmlDeviceGetUtilizationRates(nvml_device, &utilization) != NVML_SUCCESS)
        return NXS_InvalidDevice;

      value = utilization.gpu;

      return rt::getPropertyInt(property_value, property_value_size, value);
    }
    case NP_MemoryUtilization: {
      nvmlDevice_t nvml_device;
      nvmlMemory_t memInfo;
      nxs_long value = -1;

      if (nvmlDeviceGetHandleByIndex(device_id, &nvml_device) != NVML_SUCCESS)
        return NXS_InvalidDevice;

      if (nvmlDeviceGetMemoryInfo(nvml_device, &memInfo) != NVML_SUCCESS)
        return NXS_InvalidDevice;

      value = (memInfo.used * 100) / memInfo.total;  // Memory usage percentage

      return rt::getPropertyInt(property_value, property_value_size, value);
    }
    // The device's own identity and its links, from NVML — the names and
    // units of docs/Python_API.md "Host and link properties". Nothing is
    // measured; each is what the driver declares.
    case NP_PciBusId: {
      nvmlDevice_t nvml_device;
      nvmlPciInfo_t pci;
      if (nvmlDeviceGetHandleByIndex(device_id, &nvml_device) != NVML_SUCCESS ||
          nvmlDeviceGetPciInfo(nvml_device, &pci) != NVML_SUCCESS)
        return NXS_InvalidProperty;
      return rt::getPropertyStr(property_value, property_value_size, pci.busId);
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
      std::vector<DeviceLink> links = deviceLinks(device_id, rt->getDeviceCount());
      if (links.empty()) return NXS_InvalidProperty;
      if (property_id == NP_LinkCount)
        return rt::getPropertyInt(property_value, property_value_size, links.size());
      std::vector<std::string> strs;
      std::vector<nxs_long> ints;
      for (const auto &l : links) {
        switch (property_id) {
          case NP_LinkType:          strs.push_back(l.type);     break;
          case NP_LinkPeerId:        strs.push_back(l.peer_id);  break;
          case NP_LinkPeer:          ints.push_back(l.peer);     break;
          case NP_LinkGenMax:        ints.push_back(l.gen_max);  break;
          case NP_LinkGenCurrent:    ints.push_back(l.gen_cur);  break;
          case NP_LinkWidthMax:      ints.push_back(l.width_max); break;
          case NP_LinkWidthCurrent:  ints.push_back(l.width_cur); break;
          default:                   ints.push_back(l.level);    break;
        }
      }
      if (property_id == NP_LinkType || property_id == NP_LinkPeerId)
        return rt::getPropertyStrVec(property_value, property_value_size, strs);
      return rt::getPropertyVec(property_value, property_value_size, ints.data(),
                                ints.size());
    }
    default:
      return NXS_InvalidProperty;
  }
  return NXS_Success;
}

/*
 * Get the number of supported platforms on this system. 
 * On POCL, this trivially reduces to 1 - POCL itself.
 */ 
extern "C" nxs_status NXS_API_CALL
nxsGetDevicePropertyFromPath(
  nxs_int device_id,
  nxs_uint property_path_count,
  nxs_uint *property_id,
  void *property_value,
  size_t* property_value_size
)
{
  // if (property_path_count == 1)
  //   return nxsGetDeviceProperty(device_id, *property_id, property_value, property_value_size);
  // switch (property_id[0]) {
  //   case NP_CoreSubsystem:
  //     break;
  //   case NP_MemorySubsystem:
  //     break;
  //   default:
  //     return NXS_InvalidProperty;
  // }
  return NXS_Success;
}

/*
 * Allocate a buffer on the device.
 */
extern "C" nxs_int NXS_API_CALL nxsCreateBuffer(nxs_int device_id, nxs_buffer_layout shape,
                                                void *data_ptr,
                                                nxs_uint buffer_settings) {
  auto rt = getRuntime();
  auto deviceObject = rt->get<CudaDevice>(device_id);
  if (!deviceObject) return NXS_InvalidDevice;

  size_t size = nxsGetNumElements(shape);
  if (auto element_size_bits = nxsGetDataTypeSizeBits(shape.data_type)) {
    size *= element_size_bits;
    size /= 8;
  }
  NXSLOG_INFO("createBuffer: {}", size);
  if (!(buffer_settings & NXS_BufferSettings_OnDevice)) {
    void *cuda_ptr = nullptr;
    CUDA_CHECK(NXS_InvalidBuffer, cudaMalloc, &cuda_ptr, size);
    if (data_ptr != nullptr)
      CUDA_CHECK(NXS_InvalidBuffer, cudaMemcpy, cuda_ptr, data_ptr, size,
                 cudaMemcpyHostToDevice);
    data_ptr = cuda_ptr;
  }

  auto *buf = rt->getBuffer(shape, data_ptr, NXS_BufferSettings_OnDevice);
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
  auto buffer = rt->get<rt::Buffer>(buffer_id);
  if (!buffer) return NXS_InvalidBuffer;

  switch (buffer_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Name, NP_Type, NP_Shape, NP_Size, NP_Value};
      return rt::getPropertyVec(property_value, property_value_size, keys, 5);
    }
    case NP_Value:
      return rt::getPropertyInt(property_value, property_value_size,
                                reinterpret_cast<nxs_long>(buffer->getData()));
  }
  return NXS_InvalidProperty;
}

extern "C" nxs_status NXS_API_CALL nxsCopyBuffer(nxs_int buffer_id,
                                                 void *host_ptr,
                                                 nxs_uint copy_settings) {
  auto rt = getRuntime();

  auto buffer = rt->get<rt::Buffer>(buffer_id);
  if (!buffer) return NXS_InvalidBuffer;
  if (!host_ptr) return NXS_InvalidHostPtr;
  if (copy_settings == NXS_BufferDeviceToHost)
    CUDA_CHECK(NXS_InvalidBuffer, cudaMemcpy, host_ptr, buffer->get(),
              buffer->getSizeBytes(), cudaMemcpyDeviceToHost);
  else
    CUDA_CHECK(NXS_InvalidBuffer, cudaMemcpy, buffer->get(), host_ptr,
              buffer->getSizeBytes(), cudaMemcpyHostToDevice);
  return NXS_Success;
}

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

  if (is_zero || value_size == 1) {
    cudaMemset(buffer->get(), *((unsigned char*)value), total_size);
    return NXS_Success;
  }

  assert(total_size % value_size == 0 && "Buffer size must be aligned with value size");
  std::vector<uint8_t> host_buffer(total_size);
  if (value_size == 2) {
    uint16_t val = *((uint16_t*)value);
    auto* ptr = reinterpret_cast<uint16_t*>(host_buffer.data());
    std::fill(ptr, ptr + total_size / 2, val);
  } else if (value_size == 4) {
    uint32_t val = *((uint32_t*)value);
    auto* ptr = reinterpret_cast<uint32_t*>(host_buffer.data());
    std::fill(ptr, ptr + total_size / 4, val);
  } else {
    assert(total_size % value_size == 0 && "Buffer size must be aligned with value size");
    for (size_t i = 0; i < total_size; i += value_size) {
      std::memcpy(host_buffer.data() + i, value, value_size);
    }
  }

  CUDA_CHECK(NXS_InvalidBuffer, cudaMemcpy,
            buffer->get(),
            host_buffer.data(),
            total_size,
            cudaMemcpyHostToDevice);

  return NXS_Success;
}

/*
 * Release a buffer on the device.
 */
extern "C" nxs_status NXS_API_CALL
nxsReleaseBuffer(
  nxs_int buffer_id
)
{
  auto rt = getRuntime();
  auto buf = rt->get<rt::Buffer>(buffer_id);
  if (!buf) return NXS_InvalidBuffer;
  CUDA_CHECK(NXS_InvalidBuffer, cudaFree, buf->get());
  rt->dropObject(buffer_id);
  return NXS_Success;
}

/*
 * Allocate a buffer on the device.
 */
extern "C" nxs_int NXS_API_CALL nxsCreateLibrary(nxs_int device_id,
                                                 void *library_data,
                                                 nxs_uint data_size,
                                                 nxs_uint library_settings) {
  auto rt = getRuntime();

  auto device = rt->getDevice(device_id);
  if (!device) return NXS_InvalidDevice;

  CUmodule module;
  CU_CHECK(NXS_InvalidLibrary, cuModuleLoadData, &module, library_data);

  return rt->addObject(module);
}

/*
 * Allocate a buffer on the device.
 */
extern "C" nxs_int NXS_API_CALL nxsCreateLibraryFromFile(
    nxs_int device_id, const char *library_path, nxs_uint library_settings) {
  auto rt = getRuntime();
  auto device = rt->getDevice(device_id);
  if (!device)
   return NXS_InvalidDevice;

  CUmodule module;
  CU_CHECK(NXS_InvalidLibrary, cuModuleLoad, &module, library_path);

  return rt->addObject(module);
}

/************************************************************************
 * @def GetKernelProperty
 * @brief Return Kernel properties
 ***********************************************************************/
 extern "C" nxs_status NXS_API_CALL nxsGetLibraryProperty(nxs_int library_id,
  nxs_uint library_property_id,
  void *property_value,
  size_t *property_value_size) {
  
  auto rt = getRuntime();
  auto library = rt->getPtr<CUmodule>(library_id);
  if (!library) return NXS_InvalidLibrary;

  NXSLOG_INFO("getLibraryProperty {}", library_property_id);

  switch (library_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Value};
      return rt::getPropertyVec(property_value, property_value_size, keys, 1);
    }
    case NP_Value: {
      return rt::getPropertyInt(property_value, property_value_size,
            (nxs_long)library);
    }
    default:
      return NXS_InvalidProperty;
  }

  return NXS_Success;
}

/*
 * Release a Library.
 */
/*
extern "C" nxs_status NXS_API_CALL
nxsReleaseLibrary(
  nxs_int library_id
)
{
  auto rt = getRuntime();
  auto lib = rt->dropObject<MTL::Library>(library_id);
  if (!lib)
    return NXS_InvalidProgram;
  (*lib)->release();
  return NXS_Success;
}
*/

/*
 * Lookup a Kernel in a Library.
 */
extern "C" nxs_int NXS_API_CALL
nxsGetKernel(nxs_int library_id, const char *kernel_name) {
  auto rt = getRuntime();

  auto library = rt->getPtr<CUmodule>(library_id);
  if (!library) return NXS_InvalidLibrary;

  CUfunction kernel = nullptr;
  CUresult result = cuModuleGetFunction(&kernel, library, kernel_name);
  if (result != CUDA_SUCCESS) {
    const char *error_string;
    cuGetErrorString(result, &error_string);
    NXSLOG_ERROR("GetKernel {} {}", kernel_name, error_string);
    return NXS_InvalidKernel;
  }

  return rt->addObject(kernel, false);
}

/************************************************************************
 * @def GetKernelProperty
 * @brief Return Kernel properties
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsGetKernelProperty(nxs_int kernel_id,
                                           nxs_uint kernel_property_id,
                                           void *property_value,
                                           size_t *property_value_size) {
  auto rt = getRuntime();
  auto kernel = rt->getPtr<CUfunction>(kernel_id);
  if (!kernel) return NXS_InvalidKernel;

  switch (kernel_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Value, NP_CoreRegisterSize, NP_SIMDSize,
                         NP_CoreMemorySize, NP_MaxThreadsPerBlock};
      return rt::getPropertyVec(property_value, property_value_size, keys, 5);
    }
    case NP_Value: {
      return rt::getPropertyInt(property_value, property_value_size,
                                (nxs_long)kernel);
    }
    case NP_CoreRegisterSize: {
      int n_regs = 0;
      CU_CHECK(NXS_InvalidKernel, cuFuncGetAttribute, &n_regs,
               CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
      return rt::getPropertyInt(property_value, property_value_size, n_regs);
    }
    case NP_SIMDSize: {
      int simd_size = 0;
      CU_CHECK(NXS_InvalidKernel, cuFuncGetAttribute, &simd_size,
               CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel);
      return rt::getPropertyInt(property_value, property_value_size, simd_size);
    }
    case NP_CoreMemorySize: {
      int shared_size = 0;
      CU_CHECK(NXS_InvalidKernel, cuFuncGetAttribute, &shared_size,
               CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
      return rt::getPropertyInt(property_value, property_value_size,
                                shared_size);
    }
    case NP_MaxThreadsPerBlock: {
      int max_threads = 0;
      CU_CHECK(NXS_InvalidKernel, cuFuncGetAttribute, &max_threads,
               CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel);
      return rt::getPropertyInt(property_value, property_value_size, max_threads);
    }
    default:
      return NXS_InvalidProperty;
  }

  return NXS_Success;
}

/************************************************************************
 * @def CreateEvent
 * @brief Create event on the device using CUDA Driver API
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateEvent(nxs_int device_id,
                                               nxs_event_type event_type,
                                               nxs_uint event_settings) {
  auto rt = getRuntime();
  auto parent = rt->getObject(device_id);
  if (!parent) return NXS_InvalidDevice;

  CUevent event;
  if (event_type == NXS_EventType_Shared) {
    CU_CHECK(NXS_InvalidEvent, cuEventCreate, &event, CU_EVENT_DEFAULT);
  } else if (event_type == NXS_EventType_Signal) {
    CU_CHECK(NXS_InvalidEvent, cuEventCreate, &event, CU_EVENT_DISABLE_TIMING);
  } else if (event_type == NXS_EventType_Fence) {
    CU_CHECK(NXS_InvalidEvent, cuEventCreate, &event, CU_EVENT_BLOCKING_SYNC);
  } else {
    return NXS_InvalidEvent; // or whatever error code you use
  }

  return rt->addObject(event);
}

/************************************************************************
 * @def SignalEvent - Record event using Driver API
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsSignalEvent(nxs_int event_id,
                                                  nxs_int signal_value) {
  auto rt = getRuntime();
  auto event = rt->getPtr<CUevent>(event_id);
  if (!event) return NXS_InvalidEvent;
  CU_CHECK(NXS_InvalidEvent, cuEventRecord, event,
           0);  // Remove the * - use event directly
  return NXS_Success;
}

/************************************************************************
 * @def WaitEvent - Synchronize event using Driver API
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsWaitEvent(nxs_int event_id,
                                                nxs_int wait_value) {
  auto rt = getRuntime();
  auto event = rt->getPtr<CUevent>(event_id);
  if (!event) return NXS_InvalidEvent;
  CU_CHECK(NXS_InvalidEvent, cuEventSynchronize,
           event);  // Remove the * - use event directly
  return NXS_Success;
}

/************************************************************************
 * @def ReleaseEvent - Destroy event using Driver API
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsReleaseEvent(nxs_int event_id) {
  auto rt = getRuntime();
  auto event = rt->getPtr<CUevent>(event_id);
  if (!event) return NXS_InvalidEvent;
  CU_CHECK(NXS_InvalidEvent, cuEventDestroy,
           event);  // Remove the * - use event directly
  if (!rt->dropObject(event_id)) return NXS_InvalidEvent;
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
  auto device = rt->get<CudaDevice>(device_id);
  if (!device) return NXS_InvalidDevice;

  // TODO: Get the default command queue for the first Stream
  NXSLOG_INFO("createStream");
  cudaStream_t stream;
  CUDA_CHECK(NXS_InvalidStream, cudaStreamCreate, &stream);
  return rt->addObject(stream, false);
}

/************************************************************************
 * @def GetStreamProperty
 * @brief Return Stream properties
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL
nxsGetStreamProperty(nxs_int stream_id, nxs_uint stream_property_id,
                     void *property_value, size_t *property_value_size) {
  NXSLOG_INFO("getStreamProperty {}", stream_property_id);
  auto rt = getRuntime();
  auto stream = rt->getPtr<cudaStream_t>(stream_id);
  if (!stream) return NXS_InvalidStream;

  switch (stream_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_Value};
      return rt::getPropertyVec(property_value, property_value_size, keys, 1);
    }
    case NP_Value: {
      return rt::getPropertyInt(property_value, property_value_size,
                                (nxs_long)stream);
    }
  }
  return NXS_Success;
}

/************************************************************************
 * @def ReleaseStream
 * @brief Release the stream on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsReleaseStream(nxs_int stream_id) {
  NXSLOG_INFO("releaseStream {}", stream_id);
  auto rt = getRuntime();
  if (!rt->dropObject(stream_id)) return NXS_InvalidStream;
  return NXS_Success;
}

/************************************************************************
 * @def CreateSchedule
 * @brief Create schedule on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateSchedule(nxs_int device_id,
                                                  nxs_uint sched_settings) {
  auto rt = getRuntime();
  auto dev = rt->getDevice(device_id);
  if (!dev) return NXS_InvalidDevice;

  auto schedule = rt->getSchedule(dev, sched_settings);
  if (!schedule) return NXS_InvalidSchedule;
  return rt->addObject(schedule);
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
  auto schedule = rt->get<CudaSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;

  switch (schedule_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_ElapsedTime};
      return rt::getPropertyVec(property_value, property_value_size, keys, 1);
    }
    case NP_ElapsedTime: {
      return rt::getPropertyFlt(property_value, property_value_size,
                                schedule->getTime());
    }
  }
  return NXS_Success;
}

/************************************************************************
 * @def ReleaseSchedule
 * @brief Release the schedule on the device
 * @return Error status or Succes.
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL nxsReleaseSchedule(nxs_int schedule_id) {
  NXSLOG_INFO("releaseSchedule {}", schedule_id);
  auto rt = getRuntime();
  auto sched = rt->get<CudaSchedule>(schedule_id);
  if (!sched) return NXS_InvalidSchedule;
  sched->release();
  if (!rt->dropObject(schedule_id)) return NXS_InvalidSchedule;
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
  NXSLOG_INFO("runSchedule {} - {} - {}", schedule_id, stream_id,
             run_settings);

  auto rt = getRuntime();

  auto schedule = rt->get<CudaSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;

  auto stream = rt->getPtr<cudaStream_t>(stream_id);
  auto status = schedule->run(stream, run_settings);
  if (!nxs_success(status)) return status;

  if (!(run_settings & NXS_ExecutionSettings_NonBlocking)) {
    if (stream)
      CUDA_CHECK(NXS_InvalidStream, cudaStreamSynchronize, stream);
    else
      CUDA_CHECK(NXS_InvalidStream, cudaDeviceSynchronize);
  }
  return NXS_Success;
}

/************************************************************************
 * @def CreateCommand
 * @brief Create command buffer on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL nxsCreateCommand(nxs_int schedule_id,
                                                 nxs_int kernel_id,
                                                 nxs_uint command_settings) {
  auto rt = getRuntime();

  auto schedule = rt->get<CudaSchedule>(schedule_id);
  if (!schedule) return NXS_InvalidSchedule;

  auto kernel = rt->getPtr<CUfunction>(kernel_id);
  if (!kernel) return NXS_InvalidKernel;

  auto command = rt->getCommand(kernel, command_settings);
  schedule->addCommand(command);
  return rt->addObject(command);
}

/************************************************************************
 * @def CreateSignalCommand
 * @brief Create signal command on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL
nxsCreateSignalCommand(nxs_int schedule_id, nxs_int event_id,
                       nxs_int signal_value, nxs_uint command_settings) {
  NXSLOG_INFO("createSignalCommand {} - {} - {}", schedule_id, event_id,
             signal_value);
  auto rt = getRuntime();
  auto sched = rt->get<CudaSchedule>(schedule_id);
  if (!sched) return NXS_InvalidSchedule;
  auto event = rt->getPtr<cudaEvent_t>(event_id);
  if (!event) {
    CUDA_CHECK(NXS_InvalidEvent, cudaEventCreateWithFlags, &event,
               cudaEventDefault);
    rt->addObject(event);
  }

  auto *cmd = rt->getCommand(event, NXS_CommandType_Signal, signal_value,
                             command_settings);
  sched->addCommand(cmd);
  return rt->addObject(cmd);
}

/************************************************************************
 * @def CreateWaitCommand
 * @brief Create wait command on the device
 * @return Negative value is an error status.
 *         Non-negative is the bufferId.
 ***********************************************************************/
extern "C" nxs_int NXS_API_CALL
nxsCreateWaitCommand(nxs_int schedule_id, nxs_int event_id, nxs_int wait_value,
                     nxs_uint command_settings) {
  NXSLOG_INFO("createWaitCommand {} - {} - {}", schedule_id, event_id,
             wait_value);
  auto rt = getRuntime();
  auto sched = rt->get<CudaSchedule>(schedule_id);
  if (!sched) return NXS_InvalidSchedule;
  auto event = rt->getPtr<cudaEvent_t>(event_id);
  if (!event) return NXS_InvalidEvent;

  //NXSAPI_LOG(NXSAPI_STATUS_NOTE, "EventQuery: " << hipEventQuery(event));
  auto *cmd =
      rt->getCommand(event, NXS_CommandType_Wait, wait_value, command_settings);
  sched->addCommand(cmd);
  return rt->addObject(cmd);
}

/************************************************************************
 * @def GetCommandProperty
 * @brief Return Command properties
 ***********************************************************************/
extern "C" nxs_status NXS_API_CALL
nxsGetCommandProperty(nxs_int command_id, nxs_uint command_property_id,
                      void *property_value, size_t *property_value_size) {
  auto rt = getRuntime();
  auto command = rt->get<CudaCommand>(command_id);
  if (!command) return NXS_InvalidCommand;

  switch (command_property_id) {
    case NP_Keys: {
      nxs_long keys[] = {NP_ElapsedTime};
      return rt::getPropertyVec(property_value, property_value_size, keys, 1);
    }
    case NP_ElapsedTime: {
      return rt::getPropertyFlt(property_value, property_value_size,
                                command->getTime());
    }
  }
  return NXS_Success;
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

  auto command = rt->get<CudaCommand>(command_id);
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
  auto command = rt->get<CudaCommand>(command_id);
  if (!command) return NXS_InvalidCommand;
  return command->setScalar(argument_index, value, name, arg_settings);
}
/************************************************************************
 * @def FinalizeCommand
 * @brief Finalize command buffer on the device
 * @return Error status or Succes.
 ***********************************************************************/

extern "C" nxs_status NXS_API_CALL nxsFinalizeCommand(nxs_int command_id,
                                                      nxs_dim3 grid_size,
                                                      nxs_dim3 block_size,
                                                      nxs_uint shared_memory_size) {

  NXSLOG_INFO("finalizeCommand {} - {{ {}, {}, {} }} - {{ {}, {}, {} }} - {}", command_id,
             grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z,
             shared_memory_size);

  auto rt = getRuntime();

  auto command = rt->get<CudaCommand>(command_id);
  if (!command) return NXS_InvalidCommand;

  return command->finalize(grid_size, block_size, shared_memory_size);
}
