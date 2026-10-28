# KNexus Plugin API Documentation

The KNexus Plugin API defines a set of C functions that must be implemented by hardware backend plugins (such as Metal, CUDA, HIP, CPU, etc.) to provide a unified, cross-platform hardware accelerator interface. This API enables the KNexus runtime to interact with different hardware backends in a consistent way.

## Overview

- **API Location:** `include/knexus-api/nxs_functions.h` (function declarations), `include/knexus-api/_nxs_functions.h` (macro-generated)
- **Plugin Implementation Example:** `plugins/metal/metal_runtime.cpp`
- **Language:** C/C++ (extern "C" linkage for plugin functions)
- **Design:** Each function operates on integer handles (IDs) for devices, buffers, libraries, kernels, etc., which are managed by the plugin.

---

## Core Concepts

| KNexus Concept      | Plugin Implementation (Metal Example)         |
|--------------------|----------------------------------------------|
| Runtime            | Singleton managing all devices                |
| Device             | Hardware device handle (e.g., MTL::Device)    |
| Buffer             | Device memory buffer (e.g., MTL::Buffer)      |
| Library            | Compiled kernel library (e.g., MTL::Library)  |
| Kernel             | Compiled kernel/function (e.g., MTL::ComputePipelineState) |
| Stream             | Command queue (e.g., MTL::CommandQueue)       |
| Schedule           | Command buffer (e.g., MTL::CommandBuffer)     |
| Command            | Command encoder (e.g., MTL::ComputeCommandEncoder) |

---

## API Function Categories

### 1. Runtime and Device Properties

- **nxsGetRuntimeProperty**: Query runtime-level properties (e.g., backend name, device count).
- **nxsGetDeviceProperty**: Query device-level properties (e.g., name, vendor, architecture).
- The host (the cpu plugin's device, from hwloc) is a set of device properties; the names are tabled in `Python_API.md` under *Host and link properties*. A plugin that needs the host topology for its own properties includes the core's hwloc wrapper, `include/knexus/topology.h`, and links `PkgConfig::HWLOC` itself, rather than loading hwloc its own way; today only the cpu plugin does.
- **PCI identity.** A plugin states each PCI-attached device's `PciBusId` (`domain:bus:device.function`, 4-hex domain; a plugin whose source writes more, as NVML does, trims it) and nothing else about PCI: no PCIe fact crosses the wire. A plugin that finds its devices by PCI class reads the sysfs listing through `include/knexus/pci.h`, which needs no hwloc (the tenstorrent plugin does this and no longer links hwloc). The core derives every `PCIe` link, to the host and the path level between any two PCI devices of any runtimes, from its own hwloc topology and sysfs; a plugin never states a PCIe generation, width or topology level.
- **Fabric links.** A plugin states the links only it knows (NVLink, Ethernet, a new kind under its own name) once, as a runtime-level table on `nxsGetRuntimeProperty`, each pair in one direction: `LinkCount` (int, the rows), and per row `LinkType` (str_vec), `LinkDevice` (int_vec, the first endpoint's device index), `LinkPeer` (int_vec, the second's, `-1` when it is not a device of this runtime), `LinkPeerId` (str_vec, the second's bus id where known, which the core resolves across runtimes), `LinkGenMax`, `LinkGenCurrent`, `LinkWidthMax`, `LinkWidthCurrent` (int_vec, `-1` where the kind states none). The runtime's `Keys` lists them only when it has a link to state. The shared encoder `plugins/include/rt_fabric.h` holds a row as `rt::FabricLink` and answers the columns: `fabricKeys()` for `Keys`, `isFabricProperty()` and `getFabricProperty()` in `nxsGetRuntimeProperty`, so a plugin carries no serialization of its own. No `Link*` key is stated on a device; the core exposes every link as a `Link` object on the System with `Device` endpoints (`Core_API.md`).
- A property's value crosses as bytes in the caller's buffer, with `property_value_size` stating the count: an int or float as itself, a string NUL-terminated, an int or float vector as its elements laid end to end, and a string vector as its strings laid end to end, each NUL-terminated (`rt::getPropertyStrVec` packs it; the core asks the byte count with a null buffer first, then fills one of that size and splits on the NULs). A null `property_value` is a size query.

**Example (Metal):**
- Returns `"metal"` for runtime name.
- Device properties map to Metal device attributes (e.g., `device->name()`).

---

### 2. Buffer Management

- **nxsCreateBuffer**: Allocate a buffer on a device.
- **nxsCopyBuffer**: Copy buffer contents to host memory.
- **nxsReleaseBuffer**: Release a buffer.

**Example (Metal):**
- Uses `MTL::Device::newBuffer` to allocate.
- Uses `memcpy` to copy data from device to host.
- Calls `release()` on the Metal buffer object.

---

### 3. Library and Kernel Management

- **nxsCreateLibrary**: Create a library from binary data.
- **nxsCreateLibraryFromFile**: Create a library from a file.
- **nxsGetLibraryProperty**: Query library properties.
- **nxsReleaseLibrary**: Release a library.
- **nxsGetKernel**: Get a kernel/function from a library.
- **nxsGetKernelProperty**: Query kernel properties.
- **nxsReleaseKernel**: Release a kernel.

**Example (Metal):**
- Loads Metal libraries from file paths.
- Retrieves functions with `newFunction` and compiles them to pipeline states.
- Manages object lifetimes with reference counting.

---

### 4. Stream and Schedule Management

- **nxsCreateStream**: Create a command queue (stream).
- **nxsReleaseStream**: Release a command queue.
- **nxsCreateSchedule**: Create a command buffer (schedule).
- **nxsRunSchedule**: Submit a command buffer for execution.
- **nxsReleaseSchedule**: Release a command buffer.

**Example (Metal):**
- Streams are `MTL::CommandQueue` objects.
- Schedules are `MTL::CommandBuffer` objects.
- Running a schedule commits and optionally waits for completion.

---

### 5. Command Management

- **nxsCreateCommand**: Create a command encoder for a kernel.
- **nxsSetCommandArgument**: Set a buffer as a kernel argument.
- **nxsFinalizeCommand**: Finalize the command (set grid/threadgroup sizes).
- **nxsReleaseCommand**: Release the command encoder.

**Example (Metal):**
- Commands are `MTL::ComputeCommandEncoder` objects.
- Arguments are set with `setBuffer`.
- Finalization dispatches threads and ends encoding.

---

## Example Usage Flow (Metal Plugin)

1. **Device Discovery**
   - `nxsGetRuntimeProperty(NP_Size, ...)` → returns number of Metal devices.
   - `nxsGetDeviceProperty(device_id, NP_Name, ...)` → returns device name.

2. **Buffer Allocation**
   - `nxsCreateBuffer(device_id, size, flags, host_ptr)` → allocates a Metal buffer.

3. **Library and Kernel**
   - `nxsCreateLibraryFromFile(device_id, path)` → loads a Metal library.
   - `nxsGetKernel(library_id, kernel_name)` → gets a function and compiles to pipeline state.

4. **Command Submission**
   - `nxsCreateStream(device_id, ...)` → creates a command queue.
   - `nxsCreateSchedule(device_id, ...)` → creates a command buffer.
   - `nxsCreateCommand(schedule_id, kernel_id)` → creates a compute command encoder.
   - `nxsSetCommandArgument(command_id, arg_index, buffer_id)` → binds buffer.
   - `nxsFinalizeCommand(command_id, group_size, grid_size)` → dispatches threads.
   - `nxsRunSchedule(schedule_id, stream_id, blocking)` → commits and waits for completion.

5. **Cleanup**
   - `nxsReleaseCommand`, `nxsReleaseSchedule`, `nxsReleaseStream`, `nxsReleaseKernel`, `nxsReleaseLibrary`, `nxsReleaseBuffer` as needed.

---

## Object Management

- Plugins maintain an internal registry mapping integer IDs to backend objects (e.g., Metal objects).
- All API calls use these IDs to refer to objects.
- Proper reference counting and cleanup are required to avoid leaks.

---

## Error Handling

- Functions return status codes (e.g., `NXS_Success`, `NXS_InvalidDevice`, `NXS_InvalidProperty`).
- Plugins should map backend errors to KNexus status codes as closely as possible.

---

## Extending for New Backends

To implement a new backend plugin:
1. Implement all required API functions as extern "C" functions.
2. Map KNexus concepts to backend-specific objects.
3. Maintain an object registry for handle management.
4. Ensure proper error handling and resource cleanup.

---

## References

- **API Declarations:** `include/knexus-api/nxs_functions.h`, `include/knexus-api/_nxs_functions.h`
- **Example Implementation:** `plugins/metal/metal_runtime.cpp`
- **Status Codes and Properties:** `include/knexus-api/nxs_propertys.h`, `include/knexus-api/nxs.h`

---

**This API enables KNexus to support multiple hardware accelerators through a unified, extensible plugin interface.** 