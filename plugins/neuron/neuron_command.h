#ifndef RT_NEURON_COMMAND_H
#define RT_NEURON_COMMAND_H

#include "neuron.h"

#include <neuron_buffer.h>
#include <neuron_library.h>
#include <rt_command.h>

#include <vector>

class NeuronEvent;
class NeuronQueue;

// A command: one execution of a kernel (a loaded NEFF) with its tensors
// bound, or a signal or wait on an event.
//
// - Bindings: argument k of the kernel is its k-th tensor, inputs first then
//   outputs (NeuronLibrary). A named argument binds the tensor of that name
//   wherever it sits; an unnamed one binds by index. The buffer must be on
//   the model's core and exactly the tensor's byte size.
// - Tensor sets: each command owns one input and one output nrt_tensor_set_t,
//   so commands never share them (a set is keyed by tensor name: two
//   commands of one kernel would otherwise rebind each other's tensors). They
//   are (re)built by prepare() from the bindings, after every tensor is
//   checked bound: libnrt refuses an incomplete set only at execute, with a
//   bare NRT_INVALID (status 2).
// - Grid, block and shared memory size are meaningless to a NEFF and are
//   ignored; finalize() is the binding check.
// - A released buffer is unbound (unbind) and a released library detaches its
//   kernel (detachKernel), so a set never names a freed tensor or a command an
//   unloaded model.
class NeuronCommand : public nxs::rt::Command<NeuronKernel *, NeuronEvent *, nxs_int> {
  nxs_int id = -1;  // its object id in the runtime
  std::vector<NeuronBuffer *> bound;  // per kernel argument
  nrt_tensor_set_t *inputs = nullptr;
  nrt_tensor_set_t *outputs = nullptr;
  bool dirty = true;  // the sets do not match `bound`

  nxs_status destroySets();

 public:
  NeuronCommand(NeuronKernel *kernel = nullptr, nxs_uint command_settings = 0)
      : Command(kernel, command_settings) {
    event = nullptr;
  }
  NeuronCommand(NeuronEvent *event, nxs_command_type type, nxs_int event_value = 1,
                nxs_uint command_settings = 0)
      : Command(event, type, event_value, command_settings) {
    kernel = nullptr;
  }
  ~NeuronCommand() { release(); }

  void setId(nxs_int i) { id = i; }
  nxs_int getId() const { return id; }
  NeuronKernel *getKernel() const { return kernel; }
  NeuronLibrary *getLibrary() const { return kernel ? kernel->getLibrary() : nullptr; }

  nxs_status bind(nxs_int index, NeuronBuffer *buffer, const char *name);
  nxs_status finalize(nxs_dim3 grid, nxs_dim3 block, nxs_uint shared_memory_size);
  nxs_status prepare();

  // Drop every binding of `buffer`; true if there was one.
  bool unbind(NeuronBuffer *buffer);
  // The kernel's library is unloading: forget the kernel and its bindings.
  void detachKernel();

  // Execute on `queue` (the device's): blocking with nrt_execute, otherwise
  // enqueued with nrta_execute_schedule. Signal and wait commands act on
  // their event.
  nxs_status run(NeuronQueue *queue, bool non_blocking, bool timing);
  nxs_status runCommand(nxs_int stream) override;

  void release() override;
};

#endif  // RT_NEURON_COMMAND_H
