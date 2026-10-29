#ifndef RT_NEURON_H
#define RT_NEURON_H

#define NXSAPI_LOG_MODULE "neuron_rt"
#define NXSAPI_LOG_MODULE_COLOR "\033[93m"
#include <knexus-api.h>
#include <knexus-api/nxs_log.h>

#include <rt_utilities.h>

#include <nrt/nrt.h>

// Log a failed libnrt call and return `err` from the enclosing function.
// libnrt prints its own diagnostics to stderr; this adds the call site.
#define NRT_CHECK(err, call, ...)                                          \
  do {                                                                     \
    NXSLOG_TRACE("NRT_CHECK {}", #call);                                   \
    NRT_STATUS _nrt_status = call(__VA_ARGS__);                            \
    if (_nrt_status != NRT_SUCCESS) {                                      \
      NXSLOG_ERROR("{} failed: {}", #call, nrt_get_status_as_str(_nrt_status)); \
      return err;                                                          \
    }                                                                      \
  } while (0)

// The KNexus status for a failed (or finished) model execution: a NEFF
// execution reports what went wrong through its NRT_STATUS, synchronously from
// nrt_execute or at completion of an nrta_execute_schedule request.
// NRT_EXEC_COMPLETED_WITH_NUM_ERR (the outputs hold a NaN) is a completed run
// and is the caller's to warn about.
static inline nxs_status nrtExecStatus(NRT_STATUS st) {
  switch (st) {
    case NRT_SUCCESS:
    case NRT_EXEC_COMPLETED_WITH_NUM_ERR:
      return NXS_Success;
    case NRT_INVALID:          // a tensor name or size the model does not take
    case NRT_EXEC_BAD_INPUT:
      return NXS_InvalidKernelArgs;
    case NRT_EXEC_NC_BUSY:     // the core is held by another model or process
      return NXS_DeviceNotAvailable;
    case NRT_QUEUE_FULL:
    case NRT_RESOURCE:
      return NXS_OutOfResources;
    case NRT_INVALID_HANDLE:
      return NXS_InvalidKernel;
    default:
      return NXS_InvalidCommand;
  }
}

#endif  // RT_NEURON_H
