#include <neuron_command.h>
#include <neuron_device.h>
#include <neuron_schedule.h>

#include <chrono>
#include <string>

// The sets may be in use by a request still on the queue: wait for the core's
// queue before freeing them. A failure of that request is left for the next
// caller-visible synchronization (drainQueue).
nxs_status NeuronCommand::destroySets() {
  if (!inputs && !outputs) return NXS_Success;
  if (auto *lib = getLibrary()) drainQueue(lib->getDevice());
  if (inputs) nrt_destroy_tensor_set(&inputs);
  if (outputs) nrt_destroy_tensor_set(&outputs);
  inputs = outputs = nullptr;
  dirty = true;
  return NXS_Success;
}

nxs_status NeuronCommand::bind(nxs_int index, NeuronBuffer *buffer, const char *name) {
  NeuronLibrary *lib = getLibrary();
  if (!lib || !lib->getModel()) {
    NXSLOG_ERROR("command {}: no loaded kernel to bind to", id);
    return NXS_InvalidKernel;
  }
  const auto &tensors = lib->getTensors();
  int t = -1;
  if (name && *name) {
    t = lib->findTensor(name);
    if (t < 0) {
      std::string names;
      for (const auto &ti : tensors) names += (names.empty() ? "" : ", ") + ti.name;
      NXSLOG_ERROR("command {}: the kernel has no tensor '{}' (it takes: {})", id, name, names);
      return NXS_InvalidArgValue;
    }
  } else {
    if (index < 0 || index >= (nxs_int)tensors.size()) {
      NXSLOG_ERROR("command {}: argument {} out of range, the kernel takes {} tensors", id, index,
                   tensors.size());
      return NXS_InvalidArgIndex;
    }
    t = index;
  }
  const NeuronTensorInfo &ti = tensors[t];
  if (!buffer || !buffer->getTensor()) return NXS_InvalidBuffer;
  if (buffer->getDevice() != lib->getDevice()) {
    NXSLOG_ERROR("command {}: tensor '{}': the buffer is on another core than the model (vnc {})",
                 id, ti.name, lib->getDevice()->vnc);
    return NXS_InvalidDevice;
  }
  if (buffer->getSizeBytes() != ti.size) {
    NXSLOG_ERROR("command {}: a {}-byte buffer for tensor {}", id, buffer->getSizeBytes(),
                 ti.describe());
    return NXS_InvalidArgSize;
  }
  nxs_data_type want = nrtDtypeToNxs(ti.dtype);
  if (buffer->getDataType() != NXS_DataType_Undefined && want != NXS_DataType_Undefined &&
      buffer->getDataType() != want)
    NXSLOG_WARN("command {}: tensor {} bound to a buffer of data type {}: bytes are passed as is",
                id, ti.describe(), (int)buffer->getDataType());

  if (bound.size() != tensors.size()) bound.resize(tensors.size(), nullptr);
  bound[t] = buffer;
  dirty = true;
  NXSLOG_INFO("command {}: argument {} -> tensor {}", id, index, ti.describe());
  return NXS_Success;
}

nxs_status NeuronCommand::finalize(nxs_dim3 grid, nxs_dim3 block, nxs_uint shared_memory_size) {
  Command::finalize(grid, block, shared_memory_size);  // kept, but a NEFF has no launch geometry
  if (getType() != NXS_CommandType_Dispatch) return NXS_Success;
  return prepare();
}

// Check that every tensor of the kernel is bound, then (re)build the sets.
nxs_status NeuronCommand::prepare() {
  NeuronLibrary *lib = getLibrary();
  if (!lib || !lib->getModel()) {
    NXSLOG_ERROR("command {}: its kernel's library was released", id);
    return NXS_InvalidKernel;
  }
  const auto &tensors = lib->getTensors();
  if (bound.size() != tensors.size()) bound.resize(tensors.size(), nullptr);
  std::string missing;
  size_t num_missing = 0;
  for (size_t t = 0; t < tensors.size(); ++t) {
    if (bound[t]) continue;
    missing += "\n    argument " + std::to_string(t) + " = " + tensors[t].describe();
    ++num_missing;
  }
  if (num_missing) {
    NXSLOG_ERROR("command {}: {} of {} kernel tensors not bound (bind by name or argument "
                 "index):{}",
                 id, num_missing, tensors.size(), missing);
    return NXS_InvalidKernelArgs;
  }
  if (!dirty) return NXS_Success;

  destroySets();
  NRT_CHECK(NXS_OutOfHostMemory, nrt_allocate_tensor_set, &inputs);
  NRT_CHECK(NXS_OutOfHostMemory, nrt_allocate_tensor_set, &outputs);
  for (size_t t = 0; t < tensors.size(); ++t)
    NRT_CHECK(NXS_InvalidKernelArgs, nrt_add_tensor_to_tensor_set,
              tensors[t].isInput() ? inputs : outputs, tensors[t].name.c_str(),
              bound[t]->getTensor());
  dirty = false;
  return NXS_Success;
}

bool NeuronCommand::unbind(NeuronBuffer *buffer) {
  bool found = false;
  for (auto &b : bound)
    if (b == buffer) {
      b = nullptr;
      found = true;
    }
  if (found) destroySets();
  return found;
}

void NeuronCommand::detachKernel() {
  destroySets();
  bound.clear();
  kernel = nullptr;
}

nxs_status NeuronCommand::run(NeuronQueue *queue, bool non_blocking, bool timing) {
  switch (getType()) {
    case NXS_CommandType_Dispatch: {
      nxs_status status = prepare();
      if (nxs_failed(status)) return status;
      nrt_model_t *model = getLibrary()->getModel();
      if (non_blocking) {
        if (!queue) return NXS_InvalidStream;
        nrta_seq_t seq = 0;
        return queue->submit(model, inputs, outputs, seq);
      }
      auto start = std::chrono::steady_clock::now();
      NRT_STATUS st = nrt_execute(model, inputs, outputs);
      if (timing)
        time_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start)
                      .count();
      if (st == NRT_EXEC_COMPLETED_WITH_NUM_ERR)
        NXSLOG_WARN("command {}: nrt_execute: {}", id, nrt_get_status_as_str(st));
      else if (st != NRT_SUCCESS)
        NXSLOG_ERROR("command {}: nrt_execute failed: {} ({})", id, nrt_get_status_as_str(st),
                     (int)st);
      return nrtExecStatus(st);
    }
    case NXS_CommandType_Signal: {
      if (!event) return NXS_InvalidEvent;
      nrta_seq_t seq = 0;
      if (non_blocking && queue && queue->lastPending(seq)) {
        event->signalAfter(queue, seq, event_value);
        return NXS_Success;
      }
      return event->signal(event_value);
    }
    case NXS_CommandType_Wait: {
      if (!event) return NXS_InvalidEvent;
      if (non_blocking && queue && event->orderedOn(queue, event_value)) return NXS_Success;
      return event->wait(event_value);
    }
    default:
      return NXS_InvalidCommand;
  }
}

nxs_status NeuronCommand::runCommand(nxs_int stream) {
  NeuronLibrary *lib = getLibrary();
  return run(lib && lib->getDevice() ? lib->getDevice()->queue : nullptr, false, false);
}

void NeuronCommand::release() {
  destroySets();
  bound.clear();
  kernel = nullptr;
  event = nullptr;
}
