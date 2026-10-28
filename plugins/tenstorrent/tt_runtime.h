#ifndef RT_TT_RUNTIME_H
#define RT_TT_RUNTIME_H

#include "tenstorrent.h"

#include <tt_command.h>
#include <tt_device.h>
#include <tt_schedule.h>
#include <tt_library.h>
#include <tt_buffer.h>
#include <rt_runtime.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace nxs;

// The runtime owns one parent MeshDevice, the system mesh tt-metal's control
// plane maps (MeshDevice::create(MeshDeviceConfig{})), and one plugin device
// per 1x1 submesh of it (create_submeshes(MeshShape(1,1))).
//
// - Chip-id order: device k is the submesh whose chip id is k-th ascending,
//   not the k-th row-major coordinate (the 2x2 here maps chips 2 0 1 3), so
//   device k stays the k-th chip in bus-id order and every property holds.
// - The parent is for lifetime and whole-mesh queries only: nothing is
//   allocated on or enqueued to it (see tt_device.h).
// - No unit-mesh fallback: the devices are the chips the mesh maps. A mesh
//   tt-metal cannot map throws, and unit meshes then fail the same way (the
//   control plane is process-wide), so a throw is logged and the runtime
//   states 0 devices; a partial map yields fewer devices and one WARN.
// - Teardown order: every pooled object (library, schedule, command, buffer)
//   is released before the submeshes, and the submeshes before the parent:
//   `mesh` is declared before `devices`, and ~TTRuntime clears in that order.
// - `devices` is reserved to its final size before any emplace and never
//   grows afterwards, so the pointers handed to addObject stay valid.
class TTRuntime : public rt::Runtime {
  std::shared_ptr<ttmd::MeshDevice> mesh;
  std::vector<TTDevice> devices;
  rt::Pool<TTBuffer, 256> buffer_pool;
  rt::Pool<TTCommand> command_pool;
  rt::Pool<TTSchedule, 256> schedule_pool;
  rt::Pool<TTLibrary, 64> library_pool;

  template <typename T>  // a MeshCoordinate or MeshShape as "(a,b)"
  static std::string dimStr(const T &c) {
    std::string s = "(";
    for (size_t i = 0; i < c.dims(); ++i) s += (i ? "," : "") + std::to_string(c[i]);
    return s + ")";
  }

  void openMesh() {
    struct Sub {
      std::shared_ptr<ttmd::MeshDevice> mesh;
      ttmd::MeshCoordinate coord;
      int chip;
    };
    std::vector<Sub> subs;
    try {
      mesh = ttmd::MeshDevice::create(ttmd::MeshDeviceConfig(std::nullopt));
      NXSLOG_INFO("Open TT mesh: shape {}, {} chips", dimStr(mesh->shape()), mesh->num_devices());
      for (auto &sub : mesh->create_submeshes(ttmd::MeshShape(1, 1))) {
        auto chips = sub->get_devices();
        if (chips.empty()) {
          NXSLOG_WARN("TT submesh {} has no local device: skipped", sub->id());
          continue;
        }
        int chip = chips[0]->id();
        try {  // the chip's coordinate in the parent's shape; throws if absent
          subs.push_back({sub, mesh->get_view().find_device(chip), chip});
        } catch (const std::exception &e) {
          NXSLOG_WARN("TT chip {} has no coordinate in the parent mesh, skipped: {}", chip, e.what());
        }
      }
    } catch (const std::exception &e) {
      NXSLOG_ERROR("TT mesh open failed, 0 devices: {}", e.what());
      subs.clear();
      mesh.reset();
    }
    std::sort(subs.begin(), subs.end(), [](const Sub &a, const Sub &b) { return a.chip < b.chip; });
    devices.reserve(subs.size());
    for (auto &s : subs) {
      NXSLOG_INFO("Create TTDevice {}: chip {} at {}", devices.size(), s.chip, dimStr(s.coord));
      devices.emplace_back(s.mesh, s.coord, s.chip);
    }
    for (auto &d : devices) addObject(&d);
    size_t avail = 0;
    try {
      avail = ttm::GetNumAvailableDevices();
    } catch (const std::exception &e) {
      NXSLOG_WARN("TT GetNumAvailableDevices failed: {}", e.what());
    }
    if (devices.size() < avail)
      NXSLOG_WARN("TT mesh maps {} of {} available chips: the rest are not devices",
                  devices.size(), avail);
  }

 public:
  TTRuntime() : rt::Runtime() { openMesh(); }
  ~TTRuntime() {
    NXSLOG_INFO("Close TT Devices");
    // Pooled objects first, then the submeshes, then the parent: a schedule's
    // MeshWorkload frees its programs' circular buffers through its device and
    // a buffer deallocates from its submesh's allocator. Released objects hold
    // no tt-metal state by now (TTSchedule::release, TTBuffer::freeBuffer); one
    // still live here is freed after tt-metal's thread_local GraphTracker
    // state is gone, which crashes the process at exit.
    library_pool.clear();
    schedule_pool.clear();
    command_pool.clear();
    buffer_pool.clear();
    devices.clear();
    mesh.reset();
  }

  nxs_int getNumDevices() const { return devices.size(); }

  TTDevice *getDevice(nxs_int device_id) {
    return &devices[device_id];
  }

  template <typename T>
  T getPtr(nxs_int id) {
    return static_cast<T>(get(id));
  }

  TTBuffer *getBuffer(TTDevice *device, nxs_buffer_layout shape, void *data_ptr = nullptr,
                        nxs_uint settings = 0) {
    return buffer_pool.get_new(device, shape, data_ptr, settings);
  }
  nxs_status releaseBuffer(nxs_int buffer_id) {
    auto buf = get<TTBuffer>(buffer_id);
    if (!buf) return NXS_InvalidBuffer;
    buf->freeBuffer();
    buffer_pool.release(buf);
    if (!dropObject(buffer_id)) return NXS_InvalidBuffer;
    return NXS_Success;
  }

  nxs_int getSchedule(TTDevice *device, nxs_uint settings = 0) {
    auto schedule = schedule_pool.get_new(device, settings);
    if (!schedule) return NXS_InvalidSchedule;
    return addObject(schedule);
  }

  TTCommand *getCommand(TTKernel *kernel, nxs_uint settings = 0) {
    return command_pool.get_new(this, kernel, settings);
  }
#if 0
  TTCommand *getCommand(nxs_int event, nxs_command_type type,
                         nxs_int event_value = 0, nxs_uint settings = 0) {
    return command_pool.get_new(this, event, type, event_value, settings);
  }
#endif

  nxs_status releaseCommand(nxs_int command_id) {
    auto cmd = get<TTCommand>(command_id);
    if (!cmd) return NXS_InvalidCommand;
    // cmd->release();
    command_pool.release(cmd);
    if (!dropObject(command_id)) return NXS_InvalidCommand;
    return NXS_Success;
  }

  TTLibrary *getLibrary(const std::string &filename, nxs_uint settings = 0) {
    return library_pool.get_new(this, filename, settings);
  }
  TTLibrary *getLibrary(void *library_data, nxs_uint data_size, nxs_uint settings = 0) {
    return library_pool.get_new(this, library_data, data_size, settings);
  }
  nxs_status releaseLibrary(nxs_int library_id) {
    auto lib = get<TTLibrary>(library_id);
    if (!lib) return NXS_InvalidLibrary;
    // cmd->release();
    library_pool.release(lib);
    if (!dropObject(library_id)) return NXS_InvalidCommand;
    return NXS_Success;
  }

  nxs_status releaseSchedule(nxs_int schedule_id) {
    auto sched = get<TTSchedule>(schedule_id);
    if (!sched) return NXS_InvalidSchedule;
    for (auto cmd : sched->getCommands()) {
      // releaseCommand(cmd->getId());
    }
    sched->release();
    schedule_pool.release(sched);
    if (!dropObject(schedule_id)) return NXS_InvalidSchedule;
    return NXS_Success;
  }
};

#endif  // RT_TT_RUNTIME_H
