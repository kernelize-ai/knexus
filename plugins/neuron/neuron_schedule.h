#ifndef RT_NEURON_SCHEDULE_H
#define RT_NEURON_SCHEDULE_H

#include "neuron.h"

#include <neuron_command.h>
#include <neuron_device.h>
#include <rt_schedule.h>

#include <nrt/nrt_async.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

// A logical core's compute queue: libnrt's async execution queue 0 of the
// core (nrta_execute_schedule takes no other), in order. One per device,
// owned by the runtime; every stream of the device maps onto it.
//
// - submit() enqueues a model execution and returns its sequence number. The
//   NRT_STATUS libnrt writes at completion lives in `inflight` (a deque keeps
//   its address while newer requests are pushed and older ones popped). A
//   full queue (NRT_QUEUE_FULL) is drained (the oldest request waited for)
//   and the submit retried.
// - Completion is polled (nrta_is_completed); completed requests are retired
//   oldest first and the first failure is kept and logged.
// - Error reporting: the kept failure is returned, and cleared, by the next
//   wait() or sync(), the synchronization points whose status reaches the
//   API caller (buffer copies and fills, blocking runs, event waits). drain()
//   waits the same way but leaves the failure for that next caller: it is
//   for internal waits whose status nobody sees (releases, tensor set
//   teardown). An execution failure is thus reported once, at the first
//   caller-visible synchronization after it retires.
// - A hard error polling completion (anything but NRT_SUCCESS, or NRT_INVALID
//   for an unknown sequence: NRT_CLOSED, NRT_UNINITIALIZED, NRT_TIMEOUT, ...)
//   marks the queue dead: libnrt can no longer tell when a request finishes,
//   so its slot is never retired (libnrt may still write its status), every
//   wait returns NXS_DeviceNotAvailable at once instead of spinning, and
//   submit() is refused. The state is sticky; it is not consumed.
class NeuronQueue {
  struct InFlight {
    nrta_seq_t seq;
    NRT_STATUS ret;
    bool submitted;
  };
  int vnc;
  std::mutex mutex;
  std::deque<InFlight> inflight;
  nxs_status error = NXS_Success;
  std::atomic<bool> dead{false};

  void retireLocked();
  bool completed(nrta_seq_t seq);
  bool waitFor(nrta_seq_t seq);

 public:
  explicit NeuronQueue(int vnc) : vnc(vnc) {}

  int getVnc() const { return vnc; }

  nxs_status submit(nrt_model_t *model, const nrt_tensor_set_t *inputs,
                    nrt_tensor_set_t *outputs, nrta_seq_t &seq);
  bool isComplete(nrta_seq_t seq);
  // The newest request still in flight; false when the queue is idle.
  bool lastPending(nrta_seq_t &seq);
  // libnrt can no longer report completion on this queue (see above).
  bool isDead() const { return dead; }
  nxs_status wait(nrta_seq_t seq);
  nxs_status sync();
  void drain();
};

// An event: a value the host or a schedule signals and the host or a schedule
// waits on. A signal command of a non-blocking run is a pending signal: the
// value is set when the queue request before it completes. wait(v) returns
// once the value is at least v and no pending signal is outstanding (the
// latest signal counts, as with a CUDA event record). With nothing pending and
// the value short, it waits for a host signal (nxsSignalEvent from another
// thread) or for a run to post one.
//
// - A direct signal (signal(): nxsSignalEvent or a blocking run's signal
//   command) is the latest and supersedes every pending signal posted
//   before it.
// - Pending signals apply in the order they were posted: when one completes,
//   the older ones still outstanding are superseded and dropped, so a
//   request completing late on another queue never moves the value back.
// - wait() returns the failure of an execution it waited for. Where a
//   pending signal's queue is dead it is dropped, and with the value then
//   short wait() returns the failure rather than waiting for a signal that
//   cannot come.
class NeuronEvent {
  struct Pending {
    NeuronQueue *queue;
    nrta_seq_t seq;
    nxs_int value;
  };
  NeuronDevice *device;
  nxs_event_type type;
  std::mutex mutex;
  std::condition_variable cv;
  nxs_int value = 0;
  std::vector<Pending> pending;

  nxs_status pollLocked();

 public:
  NeuronEvent(NeuronDevice *dev = nullptr, nxs_event_type type = NXS_EventType_Shared)
      : device(dev), type(type) {}

  NeuronDevice *getDevice() const { return device; }
  nxs_event_type getType() const { return type; }
  nxs_int getValue();

  nxs_status signal(nxs_int v);
  void signalAfter(NeuronQueue *queue, nrta_seq_t seq, nxs_int v);
  // True where waiting for `v` needs nothing from the host before more work
  // goes on `queue`: reached already, or signalled by a request on that same
  // in-order queue.
  bool orderedOn(NeuronQueue *queue, nxs_int v);
  nxs_status wait(nxs_int v);
  void reset();
};

// A stream: a handle on its device's queue. libnrt has one compute queue per
// core, so every stream of a device is that queue and orders with the others.
struct NeuronStream {
  NeuronDevice *device = nullptr;
  nxs_uint settings = 0;
  NeuronStream(NeuronDevice *dev = nullptr, nxs_uint s = 0) : device(dev), settings(s) {}
};

// A schedule: its commands, in order, on its device's queue.
//
// - Blocking run: the queue is synchronized first (earlier non-blocking
//   work); a failure of that work is returned and the run does not start.
//   Then each command runs with nrt_execute in turn.
// - Non-blocking run: each command is enqueued (nrta_execute_schedule) and
//   the run returns; a signal command signals its event when the requests
//   before it complete, and a wait command holds the host only where the
//   event is not already ordered on this queue.
// - NXS_ExecutionSettings_Timing: host time of the run, completion included
//   (a non-blocking run waits for its requests then).
class NeuronSchedule : public nxs::rt::Schedule<NeuronCommand, NeuronDevice *, nxs_int> {
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point end_time;

 public:
  NeuronSchedule(NeuronDevice *device = nullptr, nxs_uint settings = 0)
      : Schedule(device, settings) {}
  virtual ~NeuronSchedule() = default;

  float getTime() const;

  nxs_status run(nxs_int stream, nxs_uint run_settings) override;

  nxs_status release() override;
};

#endif  // RT_NEURON_SCHEDULE_H
