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