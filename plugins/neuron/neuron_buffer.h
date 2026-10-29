#ifndef RT_NEURON_BUFFER_H
#define RT_NEURON_BUFFER_H

#include "neuron.h"

#include <rt_buffer.h>

struct NeuronDevice;

// A device buffer: one nrt tensor in the HBM of its device's logical core,
// sized to the layout's byte count. The tensor is allocated by allocate(),
// not the constructor, so a failed allocation is a status, and freed by
// freeBuffer() (or the destructor, for one still live at teardown): the pool
// reuses a released slot without destroying it. Copies and fills first wait
// for the core's queued executions (a non-blocking run), as a synchronous
// copy on a CUDA default stream does.
class NeuronBuffer : public nxs::rt::Buffer {
  NeuronDevice *device;
  nrt_tensor_t *tensor;

 public:
  NeuronBuffer(NeuronDevice *dev = nullptr,
               nxs_buffer_layout shape = nxs_buffer_layout{(nxs_uint)NXS_DataType_Undefined, 0, {0}, {0}},
               void *data_ptr = nullptr, nxs_uint settings = 0)
      : Buffer(shape, data_ptr, settings), device(dev), tensor(nullptr) {}
  ~NeuronBuffer() { freeBuffer(); }

  nxs_status allocate();

  NeuronDevice *getDevice() const { return device; }
  nrt_tensor_t *getTensor() const { return tensor; }
  void *getAddress() const;

  nxs_status copyToDevice(const void *host_buf);
  nxs_status copyToHost(void *host_buf) const;
  nxs_status fill(const void *value, size_t value_size);

  nxs_status freeBuffer();
};

#endif  // RT_NEURON_BUFFER_H
