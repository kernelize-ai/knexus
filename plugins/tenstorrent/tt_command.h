#ifndef RT_TT_COMMAND_H
#define RT_TT_COMMAND_H

#include "tenstorrent.h"

#include <rt_command.h>
#include <tt_library.h>
#include <tt_device.h>
#include <tt_buffer.h>

class TTRuntime;

class TTCommand : public nxs::rt::Command<TTKernel *, nxs_int, nxs_int> {
  TTRuntime *rt;
  std::array<TTBuffer*, NXS_KERNEL_MAX_ARGS> buffers;

 public:
  TTCommand(TTRuntime *rt = nullptr, TTKernel *kernel = nullptr,
             nxs_uint command_settings = 0)
      : Command(kernel, command_settings), rt(rt) {}

  TTCommand(TTRuntime *rt, nxs_int event, nxs_command_type type,
             nxs_int event_value = 1, nxs_uint command_settings = 0)
      : Command(event, type, event_value, command_settings), rt(rt) {}

  ~TTCommand() = default;

  void setBufferArgument(nxs_int argument_index, TTBuffer *buffer) {
    if (argument_index < 0 || argument_index >= NXS_KERNEL_MAX_ARGS) return;
    NXSLOG_INFO("Set buffer argument: {} -> {}", argument_index, (intptr_t)buffer);
    buffers[argument_index] = buffer;
  }

  nxs_status runCommand(nxs_int stream) override { assert(0); return NXS_Success; }
  nxs_status runCommand(TTDevice *device, nxs_int stream, ttmd::MeshWorkload &workload,
                        ttmd::MeshCoordinateRange &dev_range,
                        ttm::CoreRange &core_range);

  void release() override {}
};

#endif  // RT_TT_COMMAND_H