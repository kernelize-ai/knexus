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
- `knexus.get_runtime(name)`: Returns the runtime whose `Name` property is `name` (`"cpu"`, `"cuda"`, `"tt-metal"`, ...).
- `knexus.get_devices()`: Every runtime's devices in one list, runtime order then device order.
- `knexus.get_links()`: Every link of the machine as `Link` objects (see *Host and link properties*).
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
- `get_runtime()`: The `Runtime` that owns this device (empty for an empty `Device`, such as a host link's second end without a cpu runtime).
- `get_id()`: The device's index inside its runtime; every object has `get_id()`, its index inside its parent collection.
- `get_info()`: Get device properties (as a Properties object).
- `create_buffer(data or size)`: Create a buffer on this device.
- `copy_buffer(buffer)`: Copy a buffer to this device.
- `create_event(event_type=knexus.event_type.Shared)`: Create an event for synchronization.
- `get_events()`: Get all events associated with this device.
- `load_library(filepath)`: Load a kernel library.
- `create_schedule()`: Create a command schedule.
- `create_stream()`: Create a command stream (if supported).

#### Link
Represents a link between two devices (PCIe to the host, a PCIe path, NVLink, Ethernet).
- `get_devices()`: The two endpoints as a `Devices` collection of two `Device` objects, possibly of different runtimes. Each endpoint's `get_runtime()` and `get_id()` say where it sits in the tree.
- `get_property_*`: `Type`, `LinkGenMax`, `LinkGenCurrent`, `LinkWidthMax`, `LinkWidthCurrent`, `LinkTopologyLevel`; `get_property_keys()` returns their property ids, as on every object.
- `get_property_type(key)`: The type name (`"int"`, `"str"`, as `knexus.property.get_type` spells it) of the value this link states for `key` (a name or a property id). A `Link` states the `Link*` names as scalars while `knexus.property.get_type` gives the runtime fabric table's vector type, so ask the link.

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

KNexus states the machine it runs on through the objects it already has: the
host as properties of the cpu device, and the links between devices as `Link`
objects on the System, so a consumer (kmap's `kmap probe`, for one) reads
them the same way it reads a device's name. Every name below is an
`nxs_property` in `include/knexus-api/_nxs_propertys.h`, read through
`get_property_str`, `get_property_int`, `get_property_int_vec` or
`get_property_str_vec`; a property an object does not state raises, like any
other, rather than returning a guess. Nothing here is measured: each value is
what the kernel or the driver declares.

**The host** is the one device of the `cpu` runtime. The plugin loads the
host's topology once per process with hwloc (2.x), which reads the kernel's
view (sysfs and procfs on Linux, the verified platform). It loads the whole
machine (`HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED`), so the counts are the
kernel's, not the calling cgroup's. A test points it at a fixture with hwloc's
own variables, set before the plugin's first load: `HWLOC_XMLFILE` (an
exported topology, such as `test/fixtures/hwloc-two-node.xml`) or
`HWLOC_SYNTHETIC` (a description such as `pack:2 numa:1 core:4 pu:2`). Both
are read process-wide, by tt-metal's driver layer too, so a fixture process
loads the cpu plugin alone. Where hwloc fails to load or states no PU, the
host properties are absent.

| property | type | states |
| --- | --- | --- |
| `Name` | str | the first package's `CPUModel` info, trimmed; `cpu` where hwloc states none |
| `GlobalMemorySize` | int | the root object's `total_memory`, in bytes; absent where it is 0 |
| `SocketCount` | int | `Package` objects |
| `CoreCount` | int | `Core` objects |
| `ThreadCount` | int | `PU` objects (logical cpus) |
| `NumaNodeCount` | int | `NUMANode` objects |
| `ThreadSocket` | int_vec | per logical cpu number, its package's `os_index`; `-1` where none |
| `ThreadCore` | int_vec | per logical cpu number, its core's `os_index` (the kernel's `core_id`); `-1` where none |
| `ThreadNumaNode` | int_vec | per logical cpu number, the first NUMA node in its nodeset; `-1` where none |
| `NumaNodeMemorySize` | int_vec | per NUMA node in `os_index` order, its `local_memory`, in bytes; `-1` where none |
| `NumaDistance` | int_vec | row-major, `NumaNodeCount` squared entries: `[10]` on a one-node host, else hwloc's `FROM_OS` latency matrix (the kernel's `node*/distance`); absent where hwloc holds no matrix (a synthetic topology) |
| `NumaNodeCpuList` | str_vec | per NUMA node, its cpuset in the kernel's list syntax (`0-7,16-23`) |

The three `Thread*` vectors are indexed by the cpu number the kernel uses
(`cpu<N>`), so their length is one past the highest present cpu; an int
vector carries at most 1024 entries across the plugin boundary.

**A link** is a fact about two devices, so it is stated once, on the System, as
a `Link` object: `knexus.get_links()` returns every link of the machine, and
each `Link` has `get_devices()`, its two endpoints as `Device` objects that may
belong to different runtimes. The host is the one device of the `cpu` runtime,
so a device's PCIe link to the host is a `Link` between that device and the cpu
device (an empty `Device` where no cpu runtime is loaded). `knexus.get_devices()`
lists every runtime's devices, runtime order then device order. No `Link*`
property is stated on a device.

A device states its own PCI identity:

| property | type | states |
| --- | --- | --- |
| `PciBusId` | str | the device's PCI address, `domain:bus:device.function` with a 4-hex domain (as sysfs and `lspci -D` write it; a plugin whose source writes more, as NVML does, trims it); absent on a device with no PCI node |

A `Link` states, through `get_property_*`:

| property | type | states |
| --- | --- | --- |
| `Type` | str | the kind: `PCIe`, `NVLink`, `Ethernet`; a plugin for another fabric adds its own name |
| `LinkGenMax`, `LinkGenCurrent` | int | the generation (PCIe) or version (NVLink), maximum and current; `-1` where the kind states none |
| `LinkWidthMax`, `LinkWidthCurrent` | int | the width in lanes (PCIe), maximum and current; `-1` where none |
| `LinkTopologyLevel` | int | for a `PCIe` path, the two PCI nodes' common-ancestor level; `-1` otherwise |

Three kinds of link exist. The first two are computed by the core, for every
device of every runtime that states a `PciBusId`, from one hwloc topology of
the host (loaded as for the host properties, with I/O objects kept) and the
sysfs PCI link files under `$KNEXUS_HOST_ROOT` (default `/`); no plugin
computes a PCIe fact, and no `nvidia-smi` or `tt-smi` is run.

- **`PCIe` to the host**, one per device with a `PciBusId`: the endpoints are
  the device and the cpu device; generation and width from the node's
  `{max,current}_link_{speed,width}` (the speed as its PCIe generation:
  2.5 GT/s 1 up to 64 GT/s 6).
- **`PCIe` path between two devices**, one per unordered pair of devices whose
  bus ids hwloc both resolves, across runtimes: `LinkTopologyLevel` in NVML's
  `nvmlGpuTopologyLevel_t` values, by the two PCI nodes' nearest common
  ancestor: 10 a PCI bridge with at most one more bridge between it and each
  device, 20 a deeper PCI bridge, 30 a host bridge, 40 a non-I/O object (the
  Machine on a one-node host, the Package on a multi-node one) whose nodeset
  holds exactly one NUMA node, 50 otherwise (0 is never stated). A bus id
  hwloc states no object for keeps its host link and has no path links.
- **A fabric link between two devices**, stated by the runtime's plugin, which
  alone knows it, one `Link` per lane or socket pair, the pair stated once:
  the `cuda` plugin states `NVLink` (one per active NVML link, `LinkGen*` its
  `nvmlDeviceGetNvLinkVersion`); the `tenstorrent` plugin states `Ethernet`
  (one per connected ethernet core, from tt-metal's public API; generation,
  width and level `-1`). A fabric endpoint that is not a device of any loaded
  runtime is an empty `Device`.

So the links of device `d` are the `Link`s with `d` among `get_devices()`, and
how many NVLinks or ethernet cores join `d` and `e` is the count of such links
with both. A chip with no PCI node (a remote chip reached over ethernet, say)
states no `PciBusId` and has no `PCIe` links: its way to the host is its
`Ethernet` link to a chip that has them. A hip or metal device gets its `PCIe`
links from the core as soon as its plugin states `PciBusId`; a plugin for a new
fabric adds its kind by stating a pair table (`Plugin_API.md`), never a PCIe fact.

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

## Utilities

- **`knexus.dump(obj=None, info=False, children=True, format="json", pretty=False, file=None)`** dumps `obj`'s node and subtree: the System by default, or a Runtime, Device, Link or any other knexus object. `format` is `"json"` (the default) or `"minimal"`, indented text with one object per line as `collection[index] Class label` and its properties beneath it, a Link line showing its type and both endpoints as `runtime/device (PciBusId)`. Without `pretty` the value comes back, a JSON-serializable dict or the minimal text; with `pretty=True` the rendering (indented, key-sorted JSON, or the text) is printed to `file` (stdout by default) and nothing is returned. The dict carries each object's class, index, property keys and values, and children keyed by collection name; the System's `links` are `Link` nodes with their properties and `endpoints`, two references `{runtime, device, PciBusId}` into the runtimes' device lists (`null` for an empty endpoint). It only reads, never creates an object. **`python -m knexus.dump [--runtime NAME] [--info] [--no-children] [--format json|minimal]`** prints the System (or one runtime) the same way; two runs on one host are byte-identical, so `diff` compares two hosts or two builds. Run it from an installed knexus, or from the repo root as `PYTHONPATH=python python3 -m knexus.dump` where knexus is not installed. `from knexus.dump import to_text, to_json` formats a dict you already hold (the name `knexus.dump` is the function, not the module).

---

## See Also

- [Unit Tests](../test/python/README.md)

---

**KNexus Python API enables portable, high-performance accelerator programming with a modern, Pythonic interface.** 