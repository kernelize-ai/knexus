#include "neuron_schedule.h"

#include <algorithm>
#include <thread>

/************************************************************************
 * NeuronQueue
 ***********************************************************************/

// nrta_is_completed answers NRT_INVALID for a sequence number it does not
// know; that request is not waited for. Any other failure (NRT_CLOSED,
// NRT_UNINITIALIZED, NRT_TIMEOUT, ...) says nothing about the request, which
// may still be running and write its status: it is not done, and the queue is
// marked dead (neuron_schedule.h) so no wait spins on it.
bool NeuronQueue::completed(nrta_seq_t seq) {
  if (dead) return false;
  bool done = false;
  NRT_STATUS st = nrta_is_completed(seq, &done);
  if (st == NRT_SUCCESS) return done;
  if (st == NRT_INVALID) {
    NXSLOG_WARN("nrta_is_completed(0x{:x}) on vnc {}: {}", seq, vnc, nrt_get_status_as_str(st));
    return true;
  }
  if (!dead.exchange(true))
    NXSLOG_ERROR("nrta_is_completed(0x{:x}) on vnc {} failed: {} ({}); the queue is dead", seq,
                 vnc, nrt_get_status_as_str(st), (int)st);
  return false;
}

// Spin until `seq` completes; false once the queue is dead.
bool NeuronQueue::waitFor(nrta_seq_t seq) {
  while (!completed(seq)) {
    if (dead) return false;
    std::this_thread::yield();
  }
  return true;
}

// Pop the completed requests, oldest first (the queue is in order), keeping
// the first failure.
void NeuronQueue::retireLocked() {
  while (!inflight.empty() && inflight.front().submitted &&
         completed(inflight.front().seq)) {
    const InFlight &f = inflight.front();
    if (f.ret == NRT_EXEC_COMPLETED_WITH_NUM_ERR) {
      NXSLOG_WARN("vnc {}: execution 0x{:x}: {}", vnc, f.seq, nrt_get_status_as_str(f.ret));
    } else if (f.ret != NRT_SUCCESS) {
      NXSLOG_ERROR("vnc {}: execution 0x{:x} failed: {} ({})", vnc, f.seq,
                   nrt_get_status_as_str(f.ret), (int)f.ret);
      if (nxs_success(error)) error = nrtExecStatus(f.ret);
    }
    inflight.pop_front();
  }
}

nxs_status NeuronQueue::submit(nrt_model_t *model, const nrt_tensor_set_t *inputs,
                               nrt_tensor_set_t *outputs, nrta_seq_t &seq) {
  std::lock_guard<std::mutex> lock(mutex);
  if (dead) return NXS_DeviceNotAvailable;
  retireLocked();
  // libnrt writes the request's status at completion, so its slot is taken
  // in the deque first (push_back keeps the other elements in place) and
  // marked submitted once queued; retireLocked stops at an unsubmitted one.
  inflight.push_back(InFlight{0, NRT_SUCCESS, false});
  InFlight &slot = inflight.back();
  for (int spins = 0;;) {
    NRT_STATUS st = nrta_execute_schedule(model, inputs, outputs, 0, &slot.ret, &seq);
    if (st == NRT_SUCCESS) break;
    if (st != NRT_QUEUE_FULL || spins > 100000) {
      NXSLOG_ERROR("nrta_execute_schedule on vnc {} failed: {} ({})", vnc,
                   nrt_get_status_as_str(st), (int)st);
      inflight.pop_back();
      return nrtExecStatus(st);
    }
    // Full: drain the oldest request of ours and retry.
    if (inflight.size() > 1) {
      nrta_seq_t oldest = inflight.front().seq;
      NXSLOG_INFO("vnc {}: queue full with {} requests, draining 0x{:x}", vnc,
                  inflight.size() - 1, oldest);
      if (!waitFor(oldest)) {
        inflight.pop_back();  // ours, never submitted
        return NXS_DeviceNotAvailable;
      }
      retireLocked();
    } else {
      ++spins;  // full with nothing of ours on it: another thread's or library's
      std::this_thread::yield();
    }
  }
  slot.seq = seq;
  slot.submitted = true;
  return NXS_Success;
}

bool NeuronQueue::isComplete(nrta_seq_t seq) {
  bool done = completed(seq);
  std::lock_guard<std::mutex> lock(mutex);
  retireLocked();
  return done;
}

bool NeuronQueue::lastPending(nrta_seq_t &seq) {
  std::lock_guard<std::mutex> lock(mutex);
  retireLocked();
  if (inflight.empty()) return false;
  seq = inflight.back().seq;
  return true;
}

// Wait for `seq` and report (and clear) the kept failure.
nxs_status NeuronQueue::wait(nrta_seq_t seq) {
  waitFor(seq);
  std::lock_guard<std::mutex> lock(mutex);
  retireLocked();
  if (dead) return NXS_DeviceNotAvailable;
  nxs_status status = error;
  error = NXS_Success;
  return status;
}

// Wait for every request in flight; the kept failure stays for the next
// wait() or sync().
void NeuronQueue::drain() {
  nrta_seq_t last = 0;
  {
    std::lock_guard<std::mutex> lock(mutex);
    retireLocked();
    if (inflight.empty()) return;
    last = inflight.back().seq;
  }
  if (!waitFor(last)) return;
  std::lock_guard<std::mutex> lock(mutex);
  retireLocked();
}

// Wait for every request in flight and report (and clear) the kept failure.
nxs_status NeuronQueue::sync() {
  drain();
  std::lock_guard<std::mutex> lock(mutex);
  if (dead) return NXS_DeviceNotAvailable;
  nxs_status status = error;
  error = NXS_Success;
  return status;
}

nxs_status syncQueue(NeuronDevice *device) {
  return device && device->queue ? device->queue->sync() : NXS_Success;
}

void drainQueue(NeuronDevice *device) {
  if (device && device->queue) device->queue->drain();
}

/************************************************************************
 * NeuronEvent
 ***********************************************************************/

// Apply the newest pending signal whose request completed; the ones posted
// before it are superseded (they may sit on another queue and complete later:
// applying them then would move the value back).
nxs_status NeuronEvent::pollLocked() {
  for (size_t i = pending.size(); i-- > 0;) {
    if (pending[i].queue->isComplete(pending[i].seq)) {
      value = pending[i].value;
      pending.erase(pending.begin(), pending.begin() + i + 1);
      break;
    }
  }
  return NXS_Success;
}

nxs_int NeuronEvent::getValue() {
  std::lock_guard<std::mutex> lock(mutex);
  pollLocked();
  return value;
}

nxs_status NeuronEvent::signal(nxs_int v) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    value = v;
    pending.clear();  // the latest signal: older pending ones must not overwrite it
  }
  cv.notify_all();
  return NXS_Success;
}

void NeuronEvent::signalAfter(NeuronQueue *queue, nrta_seq_t seq, nxs_int v) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    pending.push_back({queue, seq, v});
  }
  cv.notify_all();
}

bool NeuronEvent::orderedOn(NeuronQueue *queue, nxs_int v) {
  std::lock_guard<std::mutex> lock(mutex);
  pollLocked();
  if (pending.empty()) return value >= v;
  for (const auto &p : pending)
    if (p.queue != queue) return false;
  return pending.back().value >= v;
}

nxs_status NeuronEvent::wait(nxs_int v) {
  nxs_status status = NXS_Success;
  std::unique_lock<std::mutex> lock(mutex);
  for (;;) {
    pollLocked();
    if (!pending.empty()) {
      Pending p = pending.front();
      lock.unlock();
      nxs_status st = p.queue->wait(p.seq);  // a failed request is reported here
      if (nxs_failed(st) && nxs_success(status)) status = st;
      lock.lock();
      if (p.queue->isDead())  // its signals cannot complete: drop them
        pending.erase(std::remove_if(pending.begin(), pending.end(),
                                     [&](const Pending &q) { return q.queue == p.queue; }),
                      pending.end());
      continue;
    }
    // A failure waited for above is returned rather than waiting on for a
    // signal that may never come.
    if (value >= v || nxs_failed(status)) return status;
    NXSLOG_TRACE("event: waiting for a signal of {} (value {}, nothing pending)", v, value);
    cv.wait(lock);
  }
}

void NeuronEvent::reset() {
  std::lock_guard<std::mutex> lock(mutex);
  value = 0;
  pending.clear();
}

/************************************************************************
 * NeuronSchedule
 ***********************************************************************/

float NeuronSchedule::getTime() const {
  return std::chrono::duration<float, std::milli>(end_time - start_time).count();
}

nxs_status NeuronSchedule::run(nxs_int stream, nxs_uint run_settings) {
  nxs_uint settings = getSettings() | run_settings;
  bool non_blocking = settings & NXS_ExecutionSettings_NonBlocking;
  bool timing = settings & NXS_ExecutionSettings_Timing;
  NeuronDevice *device = getDevice();
  if (!device || !device->queue) return NXS_InvalidSchedule;
  NeuronQueue *queue = device->queue;
  NXSLOG_INFO("Schedule::run on vnc {}: {} commands, {}", device->vnc, getCommands().size(),
              non_blocking ? "non-blocking" : "blocking");

  start_time = std::chrono::steady_clock::now();
  end_time = start_time;  // an untimed run reports 0
  if (!non_blocking) {
    // Earlier non-blocking work goes first. Its failure is reported here, the
    // first synchronization the caller sees, and this run does not start.
    nxs_status st = queue->sync();
    if (nxs_failed(st)) {
      NXSLOG_ERROR("an earlier non-blocking run on vnc {} failed; not running", device->vnc);
      return st;
    }
  }
  for (auto *cmd : getCommands()) {
    nxs_status status = cmd->run(queue, non_blocking, timing);
    if (nxs_failed(status)) return status;
  }
  if (timing) {
    nxs_status status = non_blocking ? queue->sync() : NXS_Success;
    end_time = std::chrono::steady_clock::now();
    return status;
  }
  return NXS_Success;
}

nxs_status NeuronSchedule::release() {
  // The runtime has waited for the queue and released the commands.
  return Schedule::release();
}
