#ifndef RT_TOPOLOGY_H
#define RT_TOPOLOGY_H

// The host's hardware topology, from hwloc (>= 2.0, system library found by
// pkg-config): one hwloc_topology_t per plugin, loaded on first use and kept
// for the life of the plugin. Nothing here opens a device or measures
// anything; hwloc reads the kernel's view (sysfs/procfs on Linux).
//
// - Whole machine, not the cgroup: loaded with
//   HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED, so PU and NUMA counts match the
//   kernel's sysfs even inside a restricted cpuset.
// - I/O objects only when asked for: TopologyIo::None drops every PCI, bridge
//   and OS-device object (HWLOC_TYPE_FILTER_KEEP_NONE); TopologyIo::Pci keeps
//   them with HWLOC_TYPE_FILTER_KEEP_IMPORTANT. Measured on hwloc 2.10.0 with
//   four Tenstorrent Blackhole p150b: KEEP_IMPORTANT keeps every class 0x1200
//   (ProcessingAccelerator) device and its full bridge chain (downstream port,
//   host bridge, Machine), the same chain as KEEP_ALL and as sysfs; it drops
//   only unimportant devices (14 PCI devices and 15 bridges kept vs 60 and 22).
//   So KEEP_ALL is not needed.
// - Fixtures are hwloc's own: HWLOC_XMLFILE and HWLOC_SYNTHETIC are read by
//   hwloc at load time, so a test sets one before the first call; nothing
//   here handles them specially.
//
// The first call fixes the I/O filter for the plugin: a plugin asks with one
// constant. topology() returns nullptr only when hwloc itself fails to load;
// callers then state no topology-derived values. A malformed HWLOC_SYNTHETIC
// or HWLOC_XMLFILE is not such a failure: hwloc warns on stderr and falls back
// to this host (hwloc_topology_is_thissystem() then returns 1), so a fixture
// test checks that it got the fixture.

#include <hwloc.h>

#include <string>

#if defined(__GNUC__) || defined(__clang__)
#define NXS_RT_HIDDEN __attribute__((visibility("hidden")))
#else
#define NXS_RT_HIDDEN
#endif

namespace nxs {
namespace rt {

enum class TopologyIo { None, Pci };

// A freshly loaded topology the caller owns (hwloc_topology_destroy it), or
// nullptr on failure. topology() below is the shared, lazily loaded one.
inline hwloc_topology_t loadTopology(TopologyIo io) {
  hwloc_topology_t topo = nullptr;
  if (hwloc_topology_init(&topo) != 0) return nullptr;
  hwloc_topology_set_flags(topo, HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED);
  hwloc_topology_set_io_types_filter(topo, io == TopologyIo::Pci
                                               ? HWLOC_TYPE_FILTER_KEEP_IMPORTANT
                                               : HWLOC_TYPE_FILTER_KEEP_NONE);
  if (hwloc_topology_load(topo) != 0) {
    hwloc_topology_destroy(topo);
    return nullptr;
  }
  return topo;
}

namespace topology_detail {

struct Holder {
  hwloc_topology_t topo;
  explicit Holder(TopologyIo io) : topo(loadTopology(io)) {}
  ~Holder() {
    if (topo) hwloc_topology_destroy(topo);
  }
  Holder(const Holder &) = delete;
  Holder &operator=(const Holder &) = delete;
};

}  // namespace topology_detail

// Hidden so each plugin .so keeps its own instance: a default-visibility
// inline static would be unified across every plugin loaded in the process.
NXS_RT_HIDDEN inline hwloc_topology_t topology(
    TopologyIo io = TopologyIo::None) {
  static topology_detail::Holder holder(io);  // thread-safe one-time load
  return holder.topo;
}

// PCI objects, on a topology loaded with TopologyIo::Pci. A bus id hwloc
// states no PCI device for (XML from another machine, a filtered I/O type)
// resolves to nullptr, and a nullptr object states no NUMA node and no path
// level: never a guess.

// The PCI device at `bus_id` ("0000:01:00.0", domain:bus:device.function).
inline hwloc_obj_t pciObject(hwloc_topology_t topo, const std::string &bus_id) {
  if (!topo || bus_id.empty()) return nullptr;
  return hwloc_get_pcidev_by_busidstring(topo, bus_id.c_str());
}

// The NUMA node of an I/O object: the os_index of the one node in its first
// non-I/O ancestor's nodeset (the object hwloc attaches the host bridge to),
// -1 when that nodeset is empty or holds more than one node.
inline int pciNumaNode(hwloc_topology_t topo, hwloc_obj_t obj) {
  if (!topo || !obj) return -1;
  hwloc_obj_t anc = hwloc_get_non_io_ancestor_obj(topo, obj);
  if (!anc || !anc->nodeset || hwloc_bitmap_weight(anc->nodeset) != 1) return -1;
  return hwloc_bitmap_first(anc->nodeset);
}

// The path between two PCI objects in NVML's nvmlGpuTopologyLevel_t values,
// from their common ancestor:
//   10  a PCI bridge with at most one more bridge between it and each device
//       (one switch: its upstream port, or a downstream port both sit below);
//   20  a PCI bridge with more bridges than that between it and a device;
//   30  a host bridge (the same root complex);
//   40  a non-I/O object whose nodeset is one NUMA node (on a one-node host,
//       the Machine);
//   50  otherwise.
// 0 is never stated; -1 when either object is nullptr or they are the same.
//
// hwloc_get_common_ancestor_obj() compares depths, and every I/O object
// shares a virtual (negative) depth per type, so two I/O chains of different
// lengths would walk it past the root. The common ancestor is therefore found
// by walking the I/O chains, and only the non-I/O ancestors go to hwloc's
// helper.
inline int pciTopologyLevel(hwloc_topology_t topo, hwloc_obj_t a, hwloc_obj_t b) {
  if (!topo || !a || !b || a == b) return -1;
  hwloc_obj_t common = nullptr;
  for (hwloc_obj_t x = a->parent; x && !common && hwloc_obj_type_is_io(x->type); x = x->parent)
    for (hwloc_obj_t y = b->parent; y && hwloc_obj_type_is_io(y->type); y = y->parent)
      if (x == y) {
        common = x;
        break;
      }
  if (common) {
    if (common->type != HWLOC_OBJ_BRIDGE) return -1;  // not reached: only bridges have PCI children
    if (common->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_HOST) return 30;
    auto bridgesBelow = [common](hwloc_obj_t o) {
      int n = 0;
      for (o = o->parent; o && o != common; o = o->parent)
        if (o->type == HWLOC_OBJ_BRIDGE) ++n;
      return n;
    };
    return bridgesBelow(a) <= 1 && bridgesBelow(b) <= 1 ? 10 : 20;
  }
  hwloc_obj_t na = hwloc_get_non_io_ancestor_obj(topo, a);
  hwloc_obj_t nb = hwloc_get_non_io_ancestor_obj(topo, b);
  if (!na || !nb) return -1;
  hwloc_obj_t shared = hwloc_get_common_ancestor_obj(topo, na, nb);
  return shared && shared->nodeset && hwloc_bitmap_weight(shared->nodeset) == 1 ? 40 : 50;
}

}  // namespace rt
}  // namespace nxs

#endif  // RT_TOPOLOGY_H
