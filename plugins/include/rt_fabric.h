#ifndef RT_FABRIC_H
#define RT_FABRIC_H

// A plugin's fabric links (NVLink, Ethernet, a kind of its own) as the
// runtime-level table of docs/Plugin_API.md "Fabric links": one row per link
// (a lane, a socket), each unordered pair in one direction (`device <
// peer` where both are this runtime's devices), answered column by column on
// nxsGetRuntimeProperty. The core resolves the endpoints to Devices and does
// not symmetrize. No PCIe row: the core computes every PCIe link itself.

#include <rt_utilities.h>

#include <string>
#include <vector>

namespace nxs {
namespace rt {

struct FabricLink {
  std::string type;         // LinkType: "NVLink", "Ethernet", ...
  nxs_long device = -1;     // LinkDevice: the first endpoint, a device of this runtime
  nxs_long peer = -1;       // LinkPeer: the second, -1 when not a device of this runtime
  std::string peer_id;      // LinkPeerId: the second's PCI bus id where known, else ""
  nxs_long gen_max = -1, gen_cur = -1;
  nxs_long width_max = -1, width_cur = -1;
};

// The column names a runtime's Keys lists when its table has a row.
inline const std::vector<nxs_long> &fabricKeys() {
  static const std::vector<nxs_long> keys = {
      NP_LinkCount,  NP_LinkType,   NP_LinkDevice,      NP_LinkPeer,     NP_LinkPeerId,
      NP_LinkGenMax, NP_LinkGenCurrent, NP_LinkWidthMax, NP_LinkWidthCurrent};
  return keys;
}

// True when `prop` is one of the table's columns.
inline bool isFabricProperty(nxs_uint prop) {
  for (nxs_long k : fabricKeys())
    if ((nxs_uint)k == prop) return true;
  return false;
}

// One column of `links` (or LinkCount). An empty table states nothing.
inline nxs_status getFabricProperty(nxs_uint prop, const std::vector<FabricLink> &links,
                                    void *property_value, size_t *property_value_size) {
  if (links.empty()) return NXS_InvalidProperty;
  if (prop == NP_LinkCount)
    return getPropertyInt(property_value, property_value_size, links.size());
  if (prop == NP_LinkType || prop == NP_LinkPeerId) {
    std::vector<std::string> strs;
    for (const auto &l : links) strs.push_back(prop == NP_LinkType ? l.type : l.peer_id);
    return getPropertyStrVec(property_value, property_value_size, strs);
  }
  std::vector<nxs_long> ints;
  for (const auto &l : links) {
    switch (prop) {
      case NP_LinkDevice:       ints.push_back(l.device);    break;
      case NP_LinkPeer:         ints.push_back(l.peer);      break;
      case NP_LinkGenMax:       ints.push_back(l.gen_max);   break;
      case NP_LinkGenCurrent:   ints.push_back(l.gen_cur);   break;
      case NP_LinkWidthMax:     ints.push_back(l.width_max); break;
      case NP_LinkWidthCurrent: ints.push_back(l.width_cur); break;
      default:                  return NXS_InvalidProperty;
    }
  }
  return getPropertyVec(property_value, property_value_size, ints.data(), ints.size());
}

}  // namespace rt
}  // namespace nxs

#endif  // RT_FABRIC_H
