#ifndef RT_NEURON_RUNTIME_H
#define RT_NEURON_RUNTIME_H

#include "neuron.h"

#include <neuron_buffer.h>
#include <neuron_command.h>
#include <neuron_device.h>
#include <neuron_library.h>
#include <neuron_schedule.h>
#include <rt_fabric.h>
#include <rt_pool.h>
#include <rt_runtime.h>

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

using namespace nxs;

// The runtime: one plugin device per logical NeuronCore libnrt makes visible
// to this process (nrt_get_visible_vnc_count, which honours
// NEURON_RT_VISIBLE_CORES), device k being local vnc k.
//
// - Discovery does not initialize libnrt. The counts, the instance's
//   architecture and the library version are answered before nrt_init, and
//   the chip of each core, its name and its PCI node come from sysfs, so
//   loading the plugin neither claims a NeuronCore nor pays nrt_init's
//   15-20 s.
// - nrt_init runs on the first call that needs a core (a buffer), claiming
//   every visible core for the process. It fails while another process holds
//   one of them; the call then fails and the next one retries.
// - Links: the NeuronLink table (rt_fabric.h) is read from sysfs at
//   discovery too; PCIe links are the core's.
// - Execution (README.md "Execution"): a library is a NEFF loaded on the
//   device's core, its one kernel the model; a command owns its tensor sets;
//   every stream of a device is the core's one async queue (`queues`).
// - Teardown: the queues are drained, then the command tensor sets
//   destroyed, the models unloaded and the tensors freed, then nrt_close.
//   A released buffer is first unbound from every live command and a
//   released library detaches its kernel from them and drops its kernel ids,
//   so no set names a freed tensor and no command or kernel id an unloaded
//   model. A released event is detached from the signal and wait commands
//   naming it (they then fail with NXS_InvalidEvent).
// - `devices` is sized once, at construction, and never grows, so the
//   pointers handed to addObject stay valid; they take object ids 0..N-1.
class NeuronRuntime : public rt::Runtime {
  std::vector<NeuronDevice> devices;
  std::vector<std::unique_ptr<NeuronQueue>> queues;  // one per device
  rt::Pool<NeuronBuffer, 256> buffer_pool;
  rt::Pool<NeuronLibrary, 64> library_pool;
  rt::Pool<NeuronCommand, 256> command_pool;
  rt::Pool<NeuronSchedule, 256> schedule_pool;
  rt::Pool<NeuronEvent, 256> event_pool;
  rt::Pool<NeuronStream, 64> stream_pool;
  std::mutex command_mutex;
  std::set<NeuronCommand *> live_commands;
  // The event of each signal and wait command (under command_mutex), so a
  // released event is detached from the commands that name it.
  std::map<NeuronCommand *, NeuronEvent *> command_events;
  // The object ids nxsGetKernel handed out for each library's kernel (under
  // command_mutex): released with the library, so a stale kernel id never
  // resolves to the kernel of a later library in the same pool slot.
  std::map<NeuronLibrary *, std::vector<nxs_int>> kernel_ids;
  std::string arch;     // nrt_get_instance_info's arch_name, "trn2"
  std::string version;  // libnrt's version, "2.35.30.0"
  std::vector<rt::FabricLink> links;  // NeuronLink rows, empty on a one-chip host
  std::mutex init_mutex;
  bool initialized = false;

  void discover();
  // Clear the id's entry, so it resolves to no object, and free the id.
  bool forgetObject(nxs_int id);

 public:
  NeuronRuntime() : rt::Runtime() {
    discover();
    for (auto &d : devices) {
      queues.push_back(std::make_unique<NeuronQueue>(d.vnc));
      d.queue = queues.back().get();
    }
  }
  ~NeuronRuntime();

  nxs_int getNumDevices() const { return devices.size(); }
  NeuronDevice *getDevice(nxs_int device_id) {
    if (device_id < 0 || device_id >= getNumDevices()) return nullptr;
    return &devices[device_id];
  }
  const std::string &getArch() const { return arch; }
  const std::string &getVersion() const { return version; }
  const std::vector<rt::FabricLink> &getLinks() const { return links; }

  // Initialize libnrt if it is not yet; false if nrt_init fails.
  bool ensureInit();

  NeuronBuffer *getBuffer(NeuronDevice *device, nxs_buffer_layout shape, void *data_ptr,
                          nxs_uint settings, nxs_status &status);
  nxs_status releaseBuffer(nxs_int buffer_id);

  NeuronLibrary *getLibrary(NeuronDevice *device, const std::string &path, const void *data,
                            size_t size, nxs_status &status);
  nxs_status releaseLibrary(nxs_int library_id);

  // An id for the library's kernel (nxsGetKernel), and its release.
  nxs_int addKernel(NeuronKernel *kernel);
  nxs_status releaseKernel(nxs_int kernel_id);

  NeuronCommand *getCommand(NeuronKernel *kernel, nxs_uint settings);
  NeuronCommand *getCommand(NeuronEvent *event, nxs_command_type type, nxs_int value,
                            nxs_uint settings);
  nxs_int addCommand(NeuronSchedule *schedule, NeuronCommand *command);

  nxs_int getSchedule(NeuronDevice *device, nxs_uint settings);
  nxs_status releaseSchedule(nxs_int schedule_id);

  nxs_int getEvent(NeuronDevice *device, nxs_event_type type);
  nxs_status releaseEvent(nxs_int event_id);

  nxs_int getStream(NeuronDevice *device, nxs_uint settings);
  nxs_status releaseStream(nxs_int stream_id);
  bool isStream(void *obj) { return stream_pool.owns_object(static_cast<NeuronStream *>(obj)); }
};

NeuronRuntime *getRuntime();

#endif  // RT_NEURON_RUNTIME_H
