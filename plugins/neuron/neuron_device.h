#ifndef RT_NEURON_DEVICE_H
#define RT_NEURON_DEVICE_H

#include "neuron.h"

#include <string>

class NeuronQueue;

// A plugin device: one logical NeuronCore (an NRT "vnc") visible to this
// process. libnrt addresses every resource (tensor, model, execution) by the
// local vnc index 0..visible-1, so that index is the device index. The chip
// holding the core, its PCI node and its name are discovered once by the
// runtime (neuron_runtime.h); each is -1 or empty where it could not be
// resolved, and the matching property is then absent. `queue` is the core's
// execution queue (neuron_schedule.h), owned by the runtime.
struct NeuronDevice {
  nxs_int vnc = -1;          // local logical core index, nrt_tensor_allocate's vnc
  nxs_long global_vnc = -1;  // its index among the host's logical cores
  nxs_long chip = -1;        // its Neuron device, /dev/neuron<chip>
  nxs_long lnc = -1;         // physical NeuronCores fused into this logical core
  std::string name;          // the chip's name, "Trainium2"
  std::string bus_id;        // the chip's PCI node, "0000:33:00.0"
  NeuronQueue *queue = nullptr;  // the core's compute queue 0
};

// Wait for the executions queued on the device's core (one may read or write
// a tensor about to be touched). syncQueue returns, and clears, the failure of
// an earlier non-blocking execution (NeuronQueue::sync): use it where the
// status reaches the API caller. drainQueue leaves that failure for the next
// caller-visible synchronization: use it where the status would be dropped.
// Both are no-ops without a device or queue. Defined in neuron_schedule.cpp.
nxs_status syncQueue(NeuronDevice *device);
void drainQueue(NeuronDevice *device);

#endif  // RT_NEURON_DEVICE_H
