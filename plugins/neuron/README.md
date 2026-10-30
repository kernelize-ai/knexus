# AWS Neuron Plugin for KNexus

The neuron plugin is a KNexus runtime backend for AWS Trainium (and other
NeuronCore) instances, built on the Neuron runtime library `libnrt`: discovery,
properties, device buffers, and execution of compiled NEFFs (libraries,
kernels, commands, schedules, streams and events).

## Device model: one device per logical NeuronCore

A KNexus device is one **logical NeuronCore** visible to the process (an NRT
"vnc"), not one chip. Device `k` is local vnc `k`, the index libnrt takes in
`nrt_tensor_allocate`, `nrt_load` and friends.

Why the core and not the chip:

- **libnrt addresses everything by core.** A tensor is allocated on a core's
  HBM, a NEFF is loaded onto a core (`nrt_load(neff, vnc, 1, ...)`) and runs
  there. The KNexus API has no sub-device argument on `createBuffer`,
  `createLibrary` or `createSchedule`, so a chip device would need a hidden
  core-choice policy for every buffer and kernel.
- **Memory is per core.** On Trainium2 with the default LNC=2 each logical core
  owns one 24 GiB HBM bank (`nrt_get_vnc_memory_stats` reports a 24 GiB limit
  on each of the four cores of a 96 GiB chip). A chip device's buffers would
  all land in one bank.
- **Frameworks do the same.** torch-neuronx and JAX expose each NeuronCore as a
  device, and `NEURON_RT_VISIBLE_CORES` selects cores, not chips, so a KNexus
  device index lines up with the index users already use.

The cost is that the cores of one chip share the chip's identity: each states
the same `PciBusId`, `Name` and `Architecture`, and the device_lib entry
(`device_lib/aws-npu-trn2.json`) describes the whole chip, not the core.

The logical core size follows the instance's LNC setting
(`NEURON_LOGICAL_NC_CONFIG`): on a trn2.3xlarge (one Trainium2, eight physical
cores) the default LNC=2 gives 4 devices, LNC=1 gives 8.

## Discovery and initialization

| KNexus Concept | Neuron Implementation |
|---------------|------------------------|
| Runtime | Singleton `NeuronRuntime`, devices from `nrt_get_visible_vnc_count` |
| Device | `NeuronDevice`: local vnc index plus its chip and PCI node |
| Buffer | `NeuronBuffer`: one `nrt_tensor_t` on the device's core (`NRT_TENSOR_PLACEMENT_DEVICE`) |
| Library | `NeuronLibrary`: one NEFF loaded on the device's core (`nrt_load`) |
| Kernel | `NeuronKernel`: the loaded model (`nrt_model_t`), a NEFF's one entry point |
| Command | `NeuronCommand`: one input and one output `nrt_tensor_set_t`, its own |
| Schedule | `NeuronSchedule`: commands in order; `nrt_execute`, or `nrta_execute_schedule` non-blocking |
| Stream | `NeuronStream`: the core's one async queue (queue 0) |
| Event | `NeuronEvent`: a host value, signalled when queued executions complete (`nrta_is_completed`) |

- **Loading the plugin does not initialize libnrt.** The core counts, the
  instance architecture (`nrt_get_instance_info`) and the library version are
  answered before `nrt_init`, and the rest comes from sysfs. So enumerating the
  system (`python -m knexus.dump`, say) claims no NeuronCore and does not pay
  `nrt_init`'s cost (9 to 20 s measured here).
- **`nrt_init` runs on the first buffer or library.** It claims every visible
  core for the process. While another process holds one of them it fails
  (libnrt logs "cores busy"), `createBuffer` and `createLibrary` return
  `NXS_DeviceNotAvailable`, and the next call retries. Restrict the process with `NEURON_RT_VISIBLE_CORES` to share
  a chip between processes.
- **Teardown** waits for every queued execution, destroys the commands'
  tensor sets, unloads the models, frees every live tensor, then calls
  `nrt_close`.

### Identity

- **Chip of a core:** its host-wide core id over the logical cores per chip
  (total logical cores split evenly over the `sys/class/neuron_device/neuron<N>`
  entries). The host-wide id is the core's position in
  `NEURON_RT_VISIBLE_CORES` when that names as many cores as libnrt sees, else
  its local index when every core is visible; otherwise (`NEURON_RT_NUM_CORES`,
  say) it is unknown and the core states no chip facts. After `nrt_init` the
  plugin cross-checks the chip against `nrt_get_device_id` and warns on a
  mismatch.
- **`PciBusId`:** the `neuron<N>` class entries are virtual devices with no
  `device` link, so chip `N` is the `N`-th PCI node bound to the
  `neuron-driver` PCI driver in bus-id order (the driver numbers chips in probe
  order). Where the two counts differ no core states a `PciBusId`. The core
  derives the `PCIe` links from it ("Links" below).
- Everything is read under `$KNEXUS_HOST_ROOT` (`include/knexus/pci.h`).

## Properties

Runtime:

| property | value |
| --- | --- |
| `Name` | `neuron` |
| `Type` | `npu` |
| `Vendor` | `AWS` |
| `Size` | visible logical NeuronCores |
| `Version` | libnrt's version, `2.35.30.0` |
| `LinkCount`, `LinkType`, ... | the NeuronLink table ("Links" below); listed in `Keys` only when it has a row |

Device (absent where the source does not state it):

| property | value |
| --- | --- |
| `Name` | the chip's `info/architecture/device_name` (`Trainium2`), else the architecture |
| `Vendor` | `aws` |
| `Type` | `npu` |
| `Architecture` | `nrt_get_instance_info`'s `arch_name` (`trn1`, `trn2`) |
| `Size` | physical NeuronCores in the logical core (the LNC degree) |
| `PciBusId` | the chip's PCI address |

`Vendor`-`Type`-`Architecture` is the device_lib tag, so a trn2 core loads
`device_lib/aws-npu-trn2.json` and a trn1 core `aws-npu-trn1.json`.

Buffer: `Size` (bytes) and `Value` (`nrt_tensor_get_va`).

## Links

The plugin states only what the core cannot see, its NeuronLink table, the
way the cuda (NVLink) and tenstorrent (Ethernet) plugins state theirs
(`docs/Plugin_API.md` "Fabric links", `plugins/include/rt_fabric.h`). It is
read from sysfs at discovery, without `nrt_init`, and does not change.

**Source.** `sys/class/neuron_device/neuron<N>/info/connected_devices`, the
driver's list of chip `N`'s NeuronLink neighbours as device indices
(`"12, 1, 3, 4"`, one entry per neighbour link; the same list `neuron-ls`
prints as `connected_to`). The driver (aws-neuronx-dkms 2.31,
`fw_io_topology_v2`/`_v3`) does not probe the links: it answers from a
table per instance type (trn1.32xlarge and trn2.48xlarge: a 4x4 torus of 16
chips, four neighbours each; inf2.24xlarge/48xlarge: a line, one or two) and
an empty list everywhere else, a one-chip trn2.3xlarge included.

**Rows.** One `NeuronLink` row per neighbour entry, each chip pair once
(from the lower-numbered chip's list; from the higher one's only where the
lower one does not list it). A link joins chips while a KNexus device is a
core, so a chip is stood for by its **first visible core** (its lowest
device index): `LinkDevice` and `LinkPeer` are those cores, `LinkPeerId` the
peer chip's PCI bus id. A chip with no visible core (outside
`NEURON_RT_VISIBLE_CORES`) is a peer `-1` with its bus id, which the core
resolves to an empty `Device`; a pair with no visible core on either side is
not stated. `LinkGen*` and `LinkWidth*` are `-1`: the driver states no
NeuronLink version or width, and none is derived from the architecture.

**Not stated.**

- Nothing joins the cores of one chip: neither libnrt nor sysfs reports an
  on-chip interconnect, so no row is invented for it.
- libnrt's own topology calls do not give adjacency. `nrt_get_device_id`
  maps a core to its chip (used as a cross-check after `nrt_init`);
  `nrt_platform_*` (`nrt/neuron_platform_sysfs.h`, the
  `sys/class/neuron_device/platform` files) give the rack shape for
  Trn3 UltraServers (`intra_rack_topology`, `num_chips`, ...) and on this
  trn2.3xlarge report `none` and one chip; `nrt_async_sendrecv` and `nec.h`
  are collective transports, not a link listing.
- EFA between instances: not a link between devices of this host.

**PCIe is the core's.** The plugin states `PciBusId` only. Every core of a
chip states the chip's bus id, and the core computes one `PCIe` host link per
device that states one, so a trn2.3xlarge at LNC=2 shows four identical
`PCIe` links, one per core, to the one PCI function. That is left as is: the
docs define `PciBusId` as the device's PCI address and give no way for a
plugin to say several devices share one function, and dropping the bus id
from all but one core would claim the other cores have no PCI node. Two core
behaviours follow, both outside the plugin:

- The core loads hwloc with `HWLOC_TYPE_FILTER_KEEP_IMPORTANT`, which drops
  a Neuron chip (PCI class `0x0880`, "System peripheral"): on this host
  `hwloc_get_pcidev_by_busidstring("0000:33:00.0")` is null with that filter
  and found with `KEEP_ALL`. So a Neuron device gets its host link (gen and
  width from sysfs) but no `PCIe` path link to any other PCI device, on a
  multi-chip host too.
- `test/python/test_links.py` assumes one device per bus id:
  `test_bus_ids_distinct` fails, and its endpoint lookup by bus id is
  ambiguous for a core; `test_pcie_paths` expects the path links the filter
  drops. Both fail on a Neuron host whatever this plugin states about links.

**Verified on one chip only.** This host is a trn2.3xlarge, one chip, so the
table is empty here (`connected_devices` is empty, `Keys` lists no `Link*`
name). The row building was exercised under `$KNEXUS_HOST_ROOT` with fake
sysfs trees on this host (real libnrt counts): two chips with every core
visible (one row, core 0 to core 2), the same with `NEURON_RT_VISIBLE_CORES`
selecting either chip's cores (one row, peer `-1` and the other chip's bus
id), and a four-chip ring (four rows, each pair once). It has not been run on
a multi-chip instance (trn1.32xlarge, trn2.48xlarge, inf2), so the real
`connected_devices` contents there, and that the driver's device indices
match the `neuron<N>` entries inside a container, are taken from the driver
source, not observed.

## Buffers

A buffer is a device tensor of the layout's byte count on its device's core.
`nxsCopyBuffer` is `nrt_tensor_write` / `nrt_tensor_read` over the whole
tensor; `nxsFillBuffer` is `nrt_tensor_memset` for a one-byte value and a
host-tiled write otherwise. A buffer created with host data (and without
`NXS_BufferSettings_OnDevice`) is written at creation. A zero-byte buffer is
refused (`NXS_InvalidBufferSize`) before libnrt sees it. A copy or fill first
waits for the executions queued on the buffer's core (a non-blocking run), as a
synchronous copy on a CUDA default stream does, so reading an output after
`run(non_blocking)` needs no explicit wait.

## Execution

A NEFF is what the Neuron compiler (`neuronx-cc`, or `nki.jit` for an NKI
kernel) produces: one compiled graph with named input and output tensors and
no launch geometry. The mapping:

- **Library = one NEFF.** `nxsCreateLibrary` takes its bytes,
  `nxsCreateLibraryFromFile` a path (read whole); both `nrt_load` it onto the
  device's core (never vnc -1, "first free core": the buffers of the device
  are on its core). libnrt keeps its own copy, so the host bytes can go at
  once. A NEFF libnrt refuses is `NXS_InvalidBinary`, a held core
  `NXS_DeviceNotAvailable`. Properties: `Type` (`neff`), `Size` (NEFF bytes),
  `Count` (kernels, 1), `Name` (the file, for one loaded from a path).
  `nxsReleaseLibrary` is `nrt_unload`; commands still using its kernel are
  detached and then fail to run with `NXS_InvalidKernel`.
- **Kernel = the loaded model.** A NEFF has one entry point, so
  `nxsGetKernel` accepts any name (a NKI kernel's NEFF does not carry its
  function's name) and fails only for a NEFF with no tensor. Its arguments
  are the model's tensors from `nrt_get_model_tensor_info`, **inputs first,
  then outputs**, each in libnrt's order. Properties: `Size` (argument
  count), `SubUnits` (tensor names), `DataTypes` (libnrt dtype names,
  `float32`), `Shape` (each argument's rank then its dimensions, laid end to
  end: `[2,128,512, 2,128,512, 2,128,512]`), `Description` (`a: INPUT
  float32[128,512] (262144 bytes); ...`, the usage of each), `Type` (`neff`),
  `Name` (the name first asked for).
- **Command = one input and one output tensor set**
  (`nrt_allocate_tensor_set`), owned by the command, so two commands of a
  kernel never share one. `nxsSetCommandArgument` binds a buffer to a tensor:
  by `name` where one is given (the index is then ignored), else argument
  `index` in the order above; the tensor's usage picks the set. The buffer
  must be on the model's core (`NXS_InvalidDevice`) and exactly the tensor's
  byte size (`NXS_InvalidArgSize`); an unknown name is `NXS_InvalidArgValue`,
  an index past the arguments `NXS_InvalidArgIndex`, and a buffer data type
  other than the tensor's a warning (bytes are passed as they are).
  `nxsSetCommandScalar` is refused (`NXS_InvalidArgValue`): a NEFF takes
  device tensors only; put the value in a buffer. `nxsFinalizeCommand`
  ignores grid, block and shared memory and checks that every tensor is
  bound, logging each missing one with its name, usage, dtype and shape, and
  returns `NXS_InvalidKernelArgs`; libnrt itself would only fail the execute
  with a bare `NRT_INVALID` (status 2, "Incorrect number of inputs"). A run
  makes the same check, so an unfinalized command works too. The sets are
  (re)built from the bindings on finalize or the next run.
- **Schedule = its commands in order** on its device's queue.
  - Blocking `nxsRunSchedule`: earlier non-blocking work on the core is
    waited for, then each command runs with `nrt_execute`.
  - `NXS_ExecutionSettings_NonBlocking`: each command is enqueued with
    `nrta_execute_schedule` (queue 0) and the run returns; the queue keeps
    each request's sequence number and the `NRT_STATUS` libnrt writes at
    completion. A full queue (`NRT_QUEUE_FULL`, status 7) is drained (the
    oldest request waited for) and the submit retried. A request that
    completes with an error is logged as it retires and returned by the next
    event wait on a signal after it, or by a timed non-blocking run; a
    blocking run only warns of it (it belongs to the earlier run).
  - `NXS_ExecutionSettings_Timing`: `ElapsedTime` is the host time of the
    run, completion included (a non-blocking run then waits).
  - Release: the queue is waited for, then the schedule's commands go with it
    (the API has no command release) and their tensor sets are destroyed.
- **Stream = the core's queue.** libnrt has one compute queue per logical
  core (`nrta_execute_schedule` takes no queue but 0), so every stream of a
  device is that queue and runs in order with the others. A stream of another
  device is refused (`NXS_InvalidStream`); a run with no stream uses the
  queue.
- **Event = a host-side value.** `nxsSignalEvent` sets it. A signal command
  in a non-blocking run sets it when the request before it on the queue
  completes (`nrta_is_completed`); in a blocking run, at once.
  `nxsWaitEvent(v)` returns once every pending signal has completed and the
  value is at least `v` (the latest signal counts, as with a CUDA event
  record). With nothing pending and the value short it waits for a signal
  from another thread or a later run, so waiting on an event nothing will
  signal does not return. A wait command in a non-blocking run holds the
  host only where the event is not already ordered on the same queue.
- **Buffers and commands.** A released buffer is unbound from every command
  that uses it (the next run of those reports it unbound), so no tensor set
  ever names a freed tensor; libnrt requires a set's tensors to outlive it.
- **Busy core.** `nrt_init` claims every visible core and fails while another
  process holds one: `createLibrary` (like `createBuffer`) returns
  `NXS_DeviceNotAvailable` and a later call retries. An execute that libnrt
  answers `NRT_EXEC_NC_BUSY` is `NXS_DeviceNotAvailable` too.

Measured here (trn2.3xlarge, one logical core, the NKI `vadd` below on
float32 `[128,512]`): `nrt_init` 6 s on the first library, `nrt_load` 3 ms,
a blocking run 0.3-0.5 ms host time, a non-blocking run returns in 0.04 ms,
3000 back-to-back non-blocking runs finish in 0.3 s (one queue-full drain).

The test NEFF is an NKI kernel compiled by `nki.jit` (its `kernel.neff` lands
in the artifact directory):

```python
@nki.jit
def vadd(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    nl.store(out, value=nl.add(nl.load(a), nl.load(b)))
    return out
```

## Building

The plugin is built where `nrt/nrt.h` and `libnrt.so` are found under
`NEURON_PATH` (CMake cache variable or environment, default
`/opt/aws/neuron`), and skipped with a status message otherwise. Pass
`-DKNEXUS_REQUIRE_NEURON=ON` to make a missing runtime a configure error. The
plugin links libnrt with an RPATH to its directory and lands in
`runtime_libs/libneuron_plugin.so`.

```bash
cmake .. -DKNEXUS_BUILD_PLUGINS=ON
make neuron_plugin test_neuron_exec
./test/cpp/test_buffers neuron   # from the build directory
NEURON_RT_VISIBLE_CORES=0 ./test/cpp/test_neuron_exec kernel.neff
```

`test_neuron_exec` takes a NEFF whose one float32 output is the elementwise
sum of its float32 inputs (the `vadd` above), as `argv[1]` or
`KNEXUS_NEURON_NEFF`, and skips without one or without a neuron device.

## Status

Done: discovery, runtime and device properties, `PciBusId`, the NeuronLink
table, buffers (create, copy both ways, fill, release), and execution
(libraries, kernels, commands, schedules blocking and non-blocking, streams,
events).

Open:

- Scalars: a NEFF takes tensors only; a scalar could become a small tensor
  bound by name, but nothing here does that yet.
- One queue per core: two streams of a device do not run concurrently.
  Blocking runs from several threads should (libnrt ran 4 threads x 200
  `nrt_execute`s of one model, each with its own tensor sets, correctly when
  called directly), but that is not exercised through KNexus.
- Only single-core NEFFs were run. A collectives NEFF (`nrt_load_collectives`)
  and a NEFF that spans cores are not handled.
- An execution error reported at completion of a non-blocking request is
  logged and returned by the next wait; no test here produces one.
- `GlobalMemorySize` from `nrt_get_vnc_memory_stats`, which needs `nrt_init`,
  is not stated yet.
