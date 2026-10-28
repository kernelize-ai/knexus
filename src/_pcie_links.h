#ifndef _KNEXUS_PCIE_LINKS_H
#define _KNEXUS_PCIE_LINKS_H

// The core's PCIe link builder, on plain inputs so a test drives it with an
// hwloc XML fixture and a temp sysfs tree, no device needed. The System feeds
// it every runtime's devices in getDevices() order and wraps each result in a
// Link (src/system.cpp).

#include <knexus-api.h>

#include <hwloc.h>

#include <filesystem>
#include <string>
#include <vector>

namespace knexus {
namespace detail {

// One PCIe link. `a` and `b` index the bus-id list the builder was given;
// b == kPcieHost marks a link to the host (the cpu device). -1 where the kind
// states no such figure: a host link has no level, a path link no gen/width.
struct PcieLink {
  static constexpr nxs_int kPcieHost = -1;
  nxs_int a = -1, b = kPcieHost;
  nxs_long gen_max = -1, gen_cur = -1;
  nxs_long width_max = -1, width_cur = -1;
  nxs_long level = -1;
  bool isHost() const { return b == kPcieHost; }
};

// `busIds[i]` is device i's PciBusId, empty for a device that states none.
// Returns one host link per non-empty bus id, in list order, with gen/width
// from <hostRoot>/sys/bus/pci/devices/<bus id> (-1 where sysfs has no such
// file), then one path link per unordered pair i < j whose two bus ids `topo`
// both resolves, with the level from nxs::rt::pciTopologyLevel. A null `topo`
// gives host links only.
std::vector<PcieLink> buildPcieLinks(const std::vector<std::string> &busIds,
                                     hwloc_topology_t topo,
                                     const std::filesystem::path &hostRoot);

}  // namespace detail
}  // namespace knexus

#endif  // _KNEXUS_PCIE_LINKS_H
