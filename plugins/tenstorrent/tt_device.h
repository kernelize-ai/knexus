#ifndef RT_TT_DEVICE_H
#define RT_TT_DEVICE_H

#include "tenstorrent.h"

#include <mutex>

// A plugin device: a 1x1 submesh of the runtime's mesh (TTRuntime::mesh, see
// tt_runtime.h). The device holds the submesh, its coordinate in the parent
// mesh's shape, and its tt-metal chip id. Every buffer, program and workload
// of this device is created on and enqueued to the submesh, never the parent:
// each mesh has its own allocator over the same DRAM, so parent and submesh
// allocations alias, and using both CQs aborts the process at release.
class TTDevice {
  std::shared_ptr<ttmd::MeshDevice> device;  // the 1x1 submesh
  ttmd::MeshCoordinate coord;                // its position in the parent mesh
  int chip_id;                               // its one chip's tt-metal id
 public:
  TTDevice(std::shared_ptr<ttmd::MeshDevice> submesh, ttmd::MeshCoordinate coord, int chip_id)
      : device(std::move(submesh)), coord(coord), chip_id(chip_id) {}
  virtual ~TTDevice() { release(); }

  nxs_status release() {
    device.reset();
    return NXS_Success;
  }

  std::shared_ptr<ttmd::MeshDevice> get() { return device; }

  ttmd::MeshCommandQueue& getCQ() { return device->mesh_command_queue(); }

  ttmd::MeshCoordinateRange getRange() {
    TT_NOBJ_CHECK(devRange, ttmd::MeshCoordinateRange, device->shape());
    return devRange;
  }

  const ttmd::MeshCoordinate &getCoord() const { return coord; }
  int getChipId() const { return chip_id; }
};

#endif  // RT_TT_DEVICE_H
