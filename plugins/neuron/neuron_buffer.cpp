#include <neuron_buffer.h>
#include <neuron_device.h>
#include <neuron_schedule.h>

#include <algorithm>
#include <string>
#include <vector>

// The caller (NeuronRuntime::getBuffer) has initialized libnrt. A zero-byte
// tensor is refused here: nrt_tensor_allocate fails it with NRT_RESOURCE and a
// full runtime dump on stderr.
nxs_status NeuronBuffer::allocate() {
  if (!device) return NXS_InvalidDevice;
  size_t size = getSizeBytes();
  if (size == 0) {
    NXSLOG_ERROR("NeuronBuffer: zero-size buffer");
    return NXS_InvalidBufferSize;
  }
  std::string name = "knexus_buf_nc" + std::to_string(device->vnc);
  NRT_CHECK(NXS_MemObjectAllocationFailure, nrt_tensor_allocate, NRT_TENSOR_PLACEMENT_DEVICE,
            device->vnc, size, name.c_str(), &tensor);
  NXSLOG_INFO("NeuronBuffer: vnc={} size={}", device->vnc, size);

  if (!(getSettings() & NXS_BufferSettings_OnDevice) && getData() != nullptr) {
    setSettings(getSettings() | NXS_BufferSettings_OnDevice);
    return copyToDevice(getData());
  }
  return NXS_Success;
}

// Wait for the executions queued on the buffer's core: one may read or write
// this tensor. Their failure is logged as they retire and is the run's.
static void syncQueue(NeuronDevice *device) {
  if (device && device->queue) device->queue->sync();
}

// The tensor's virtual address as libnrt reports it; null where it has none.
void *NeuronBuffer::getAddress() const {
  return tensor ? nrt_tensor_get_va(tensor) : nullptr;
}

nxs_status NeuronBuffer::copyToDevice(const void *host_buf) {
  if (!tensor) return NXS_InvalidBuffer;
  if (!host_buf) return NXS_InvalidHostPtr;
  syncQueue(device);
  NRT_CHECK(NXS_InvalidBuffer, nrt_tensor_write, tensor, host_buf, 0, getSizeBytes());
  return NXS_Success;
}

nxs_status NeuronBuffer::copyToHost(void *host_buf) const {
  if (!tensor) return NXS_InvalidBuffer;
  if (!host_buf) return NXS_InvalidHostPtr;
  syncQueue(device);
  NRT_CHECK(NXS_InvalidBuffer, nrt_tensor_read, tensor, host_buf, 0, getSizeBytes());
  return NXS_Success;
}

// Repeat `value` (value_size bytes) over the buffer, the last copy truncated
// where the size is not a multiple. One byte is a device memset; a wider
// pattern is laid out on the host and written once.
nxs_status NeuronBuffer::fill(const void *value, size_t value_size) {
  if (!tensor) return NXS_InvalidBuffer;
  if (!value || value_size == 0) return NXS_InvalidValue;
  size_t size = getSizeBytes();
  if (value_size == 1) {
    syncQueue(device);
    NRT_CHECK(NXS_InvalidBuffer, nrt_tensor_memset, tensor, 0,
              *static_cast<const unsigned char *>(value), size);
    return NXS_Success;
  }
  std::vector<char> host(size);
  const char *src = static_cast<const char *>(value);
  for (size_t off = 0; off < size; off += value_size)
    std::copy_n(src, std::min(value_size, size - off), host.data() + off);
  return copyToDevice(host.data());
}

nxs_status NeuronBuffer::freeBuffer() {
  if (tensor) {
    NXSLOG_INFO("NeuronBuffer: freeBuffer: size={}", getSizeBytes());
    nrt_tensor_free(&tensor);
    tensor = nullptr;
  }
  return NXS_Success;
}
