# KNexus Python API

The KNexus Python API provides a modern, object-oriented interface for cross-platform hardware accelerator programming. It enables device discovery, memory management, kernel execution, and property queries in a unified way across multiple backends (CUDA, HIP, Metal, CPU, etc.).

## Installation

1. **Build KNexus with Python bindings:**
   ```bash
   cd /path/to/knexus
   mkdir build && cd build
   cmake .. -DKNEXUS_BUILD_PYTHON_MODULE=ON
   make
   ```

2. **Install Python dependencies:**
   ```bash
   pip install numpy
   ```

3. **Import KNexus in Python:**
   ```python
   import knexus
   ```

---

## API Overview

### Top-Level Functions

- `knexus.get_runtimes()`: Returns a list of available runtime backends (CUDA, Metal, CPU, etc.).
- `knexus.get_device_info()`: Returns the device info database.
- `knexus.lookup_device_info(name)`: Lookup device info by name.
- `knexus.create_buffer(size)` / `knexus.create_buffer(data)`: Create a system buffer from size or numpy array.

### Main Classes

#### Runtime
Represents a backend runtime (e.g., CUDA, Metal, CPU).
- `get_devices()`: List all devices for this runtime.
- `get_device(id)`: Get a device by index.

#### Device
Represents a hardware device.
- `get_info()`: Get device properties (as a Properties object).
- `create_buffer(data or size)`: Create a buffer on this device.
- `copy_buffer(buffer)`: Copy a buffer to this device.
- `create_event(event_type=knexus.event_type.Shared)`: Create an event for synchronization.
- `get_events()`: Get all events associated with this device.
- `load_library(filepath)`: Load a kernel library.
- `create_schedule()`: Create a command schedule.
- `create_stream()`: Create a command stream (if supported).

#### Buffer
Represents a memory buffer.
- `copy(host_array)`: Copy buffer contents to a numpy array or host buffer.
- `get_property_*`: Query buffer properties.

#### Library
Represents a loaded kernel library.
- `get_kernel(name)`: Get a kernel by name.

#### Kernel
Represents a compiled kernel.

#### Event
Represents a synchronization primitive.
- `signal(value=1)`: Signal the event with a specific value.
- `wait(value=1)`: Wait for the event to be signaled with a specific value.
- `get_property_*`: Query event properties.

#### Schedule
Represents a command schedule.
- `create_command(kernel)`: Create a command for a kernel.
- `create_signal_command(event, value=1)`: Create a signal command for an event.
- `create_wait_command(event, value=1)`: Create a wait command for an event.
- `run(stream=None, blocking=True)`: Execute the schedule.

#### Command
Represents a kernel execution command.
- `set_buffer(index, buffer)`: Set a buffer as a kernel argument.
- `finalize(group_size, grid_size)`: Finalize the command for execution.
- `get_event()`: Get the associated event (for signal/wait commands).

#### Properties
Represents a set of properties (device, buffer, etc.).
- `get_str(name or path)`: Get a string property.
- `get_int(name or path)`: Get an integer property.
- `get_str_vec(path)`: Get a list of string properties.

---

## Property Query Methods

KNexus provides a flexible and powerful property system for querying metadata and capabilities of all major objects (Device, Buffer, Library, Kernel, Schedule, Command, etc.). The property system supports both integer property IDs (enums) and string names, as well as hierarchical property paths.

### General Pattern

All major objects expose the following property query methods:

- **`get_property_str(name_or_enum)`**  
  Returns the property value as a string.

- **`get_property_int(name_or_enum)`**  
  Returns the property value as an integer.

- **`get_property_flt(name_or_enum)`**  
  Returns the property value as a float (if applicable).

- **`get_property_int_vec(name_or_enum)`**  
  Returns the property value as a list of integers.

- **`get_property_str_vec(name_or_enum)`**  
  Returns the property value as a list of strings.

- **`get_property_str(path: List[str] or List[int])`**  
  Returns a string property from a hierarchical path.

- **`get_property_int(path: List[str] or List[int])`**  
  Returns an integer property from a hierarchical path.

- **`get_property_flt(path: List[str] or List[int])`**  
  Returns a float property from a hierarchical path.

- **`get_property_str_vec(path: List[str] or List[int])`**  
  Returns a list of string properties from a hierarchical path.

- **`get_property_int_vec(path: List[str] or List[int])`**  
  Returns a list of integer properties from a hierarchical path.

- **`get_property_flt_vec(path: List[str] or List[int])`**  
  Returns a list of float properties from a hierarchical path.

### Usage Examples

#### Querying Device Properties

```python
info = device.get_info()

# By string name
name = info.get_str("Name")
vendor = info.get_str("Vendor")

# By enum (from knexus.property)
arch = info.get_str(knexus.property.Architecture)

# By hierarchical path
l2_size = info.get_int(["MemorySubsystem", "MemoryTypes", 1, "Size"])
```

#### Querying Buffer Properties

```python
size = buffer.get_property_int("Size")
dtype = buffer.get_property_str("DataType")
```

#### Querying Library and Kernel Properties

```python
lib_type = library.get_property_str("Type")
kernel_name = kernel.get_property_str("Name")
```

#### Querying Schedule and Command Properties

```python
sched_status = schedule.get_property_str("Status")
cmd_args = command.get_property_int_vec("Arguments")
```

### Notes

- If a property is not found, a `RuntimeError` is raised.
- Property names are case-sensitive and must match the schema or backend.
- Enum values are available in the `knexus.property` submodule for convenience and type safety.
- Hierarchical paths allow access to nested properties, such as memory subsystem details or device features.

### Host and link properties

KNexus states the machine it runs on as properties by name on the objects it
already has, so a consumer (kmap's `kmap probe`, for one) reads the host and
the links the same way it reads a device's name. Every name below is an
`nxs_property` in `include/knexus-api/_nxs_propertys.h`, read through
`get_property_str`, `get_property_int`, `get_property_int_vec` or
`get_property_str_vec`; a property a
runtime or a device does not state raises, like any other, rather than
returning a guess. Nothing here is measured: each value is what the kernel or
the driver declares.

**The host** is the one device of the `cpu` runtime. On Linux the plugin reads
the kernel's sysfs and procfs under `$KNEXUS_HOST_ROOT` (default `/`), so a
test can point it at a fixture tree; on another OS the host properties are
absent.

| property | type | states |
| --- | --- | --- |
| `Name` | str | the first `model name` in `proc/cpuinfo` |
| `GlobalMemorySize` | int | `MemTotal` in `proc/meminfo`, in bytes |
| `SocketCount` | int | distinct `physical_package_id` under `sys/devices/system/cpu/cpu*/topology/` |
| `CoreCount` | int | distinct (socket, `core_id`) pairs |
| `ThreadCount` | int | logical cpus with a topology directory |
| `NumaNodeCount` | int | `sys/devices/system/node/node*/` directories holding a `cpulist` |
| `ThreadSocket` | int_vec | per logical cpu number, its socket; `-1` where the kernel states none |
| `ThreadCore` | int_vec | per logical cpu number, its `core_id`; `-1` where none |
| `ThreadNumaNode` | int_vec | per logical cpu number, the NUMA node whose `cpulist` holds it; `-1` where none |
| `NumaNodeMemorySize` | int_vec | per NUMA node, its `MemTotal` from `node*/meminfo`, in bytes; `-1` where none |
| `NumaDistance` | int_vec | the kernel's `node*/distance` table, row-major, `NumaNodeCount` squared entries |
| `NumaNodeCpuList` | str_vec | per NUMA node, its `node*/cpulist` text as the kernel writes it (`0-7,16-23`) |

The three `Thread*` vectors are indexed by the cpu number the kernel uses
(`cpu<N>`), so their length is one past the highest present cpu; an int
vector carries at most 1024 entries across the plugin boundary.

**A link** is stated on the device it belongs to, one row per link, and
every `Link*` property is a vector over those rows, `LinkCount` long. The
row's `LinkType` says what kind of link it is; the other columns hold what
that kind states and `-1` or `""` where it states nothing. The `cuda` plugin
reads NVML, which it already links; no `nvidia-smi` is run.

| property | type | states |
| --- | --- | --- |
| `PciBusId` | str | the device's own `nvmlDeviceGetPciInfo` `busId`, `domain:bus:device.function` |
| `LinkCount` | int | the rows stated, the length of every vector below |
| `LinkType` | str_vec | per row, its kind: `PCIe`, `NVLink`; a plugin for another fabric adds its own name |
| `LinkPeer` | int_vec | per row, the peer device's index in this runtime; `-1` for the host, or a remote that is not a device of this runtime |
| `LinkPeerId` | str_vec | per row, the peer's id where the kind knows one (the remote's PCI bus id); `""` where not |
| `LinkGenMax`, `LinkGenCurrent` | int_vec | per row, the generation (PCIe) or version (NVLink), maximum and current |
| `LinkWidthMax`, `LinkWidthCurrent` | int_vec | per row, the width in lanes (PCIe), maximum and current |
| `LinkTopologyLevel` | int_vec | per row, the path's common-ancestor level where the kind has one |

The `cuda` plugin states three kinds of row:

- **`PCIe` to the host**, `LinkPeer` `-1`, always the first row: generation
  and width from `nvmlDeviceGet{Max,Curr}PcieLink{Generation,Width}`.
- **`NVLink` to a peer**, one row per active NVML link: `LinkPeerId` is the
  remote's bus id, `LinkPeer` the device it names when that device is one of
  this runtime's, and `LinkGen*` the link's `nvmlDeviceGetNvLinkVersion`.
  How many NVLinks reach device `j` is the count of such rows with
  `LinkPeer == j`.
- **`PCIe` path to a peer**, one row per other device of the runtime:
  `LinkTopologyLevel` is `nvmlDeviceGetTopologyCommonAncestor` as NVML's
  `nvmlGpuTopologyLevel_t` value — 0 internal, 10 a single switch, 20
  several switches, 30 a host bridge, 40 a NUMA node, 50 the system.

So the rows of device `i` with `LinkPeer == j` are its links to device `j`,
and the pair is symmetric: device `j` states the same rows back. A hip, metal
or tenstorrent plugin states the same columns with its own kinds when it has
a link to state; until it does, the properties are absent there.

---

## Event Types and Synchronization

KNexus supports three types of events for different synchronization patterns:

### Event Types

- **`knexus.event_type.Shared`**: Shared events for cross-queue synchronization
- **`knexus.event_type.Signal`**: Signal events for simple completion notifications  
- **`knexus.event_type.Fence`**: Fence events for kernel completion synchronization

### Basic Event Usage

```python
import knexus

# Create an event
event = device.create_event(knexus.event_type.Shared)

# Signal the event
event.signal(1)

# Wait for the event
event.wait(1)
```

### Event-Based Synchronization

```python
# Create event and schedule
event = device.create_event()
schedule = device.create_schedule()

# Add kernel command
cmd = schedule.create_command(kernel)
cmd.set_buffer(0, input_buffer)
cmd.finalize([256,1,1], [1024,1,1])

# Add signal command
signal_cmd = schedule.create_signal_command(event, 1)

# Run schedule non-blocking
schedule.run(blocking=False)

# Wait for completion
event.wait(1)
```

For detailed event documentation, see [Event API](Event_API.md).

### Advanced: Property Vectors

Some properties are arrays or lists (e.g., supported memory types, available kernels). Use the `_vec` methods to retrieve these as Python lists.

```python
mem_types = info.get_str_vec(["MemorySubsystem", "SupportedMemoryTypes"])
```

---

## Summary Table

| Method                        | Return Type         | Description                                 |
|-------------------------------|--------------------|---------------------------------------------|
| `get_property_str(key)`       | `str`              | Get property as string                      |
| `get_property_int(key)`       | `int`              | Get property as integer                     |
| `get_property_flt(key)`       | `float`            | Get property as float                       |
| `get_property_str_vec(key)`   | `List[str]`        | Get property as list of strings             |
| `get_property_int_vec(key)`   | `List[int]`        | Get property as list of integers            |
| `get_property_flt_vec(key)`   | `List[float]`      | Get property as list of floats              |

- `key` can be a string, enum, or a list (for hierarchical paths).

---

## Example Usage

```python
import knexus
import numpy as np

# Get available runtimes and devices
runtimes = knexus.get_runtimes()
runtime = runtimes[0]
devices = runtime.get_devices()
device = devices[0]

# Create a buffer from numpy array
data = np.ones(1024, dtype=np.float32)
buf = device.create_buffer(data)

# Load a kernel library and get a kernel
lib = device.load_library("kernel.so")
kernel = lib.get_kernel("add_vectors")

# Create a schedule and command
schedule = device.create_schedule()
command = schedule.create_command(kernel)
command.set_buffer(0, buf)
command.finalize([32,1,1], [1024,1,1])

# Run the schedule
schedule.run()
```

---

## Status and Error Codes

KNexus exposes status codes as enums in `knexus.status` (e.g., `knexus.status.NXS_Success`) and properties as enums in `knexus.property`.

---

## Design Notes

- All objects are reference-counted and managed by the underlying C++ backend.
- Handles numpy arrays and Python buffers for efficient data transfer.
- Supports device and runtime enumeration, property queries, and kernel execution in a backend-agnostic way.

---

## Advanced Features

- **Streams:** For asynchronous execution (if supported by backend).
- **Multiple Devices:** Iterate over all runtimes and devices for multi-accelerator systems.
- **Device Info Database:** Use `knexus.get_device_info()` and `knexus.lookup_device_info(name)` for architecture-aware programming.

---

## See Also

- [Unit Tests](../test/python/README.md)

---

**KNexus Python API enables portable, high-performance accelerator programming with a modern, Pythonic interface.** 