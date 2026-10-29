#ifndef _KNEXUS_FABRIC_LINKS_H
#define _KNEXUS_FABRIC_LINKS_H

// The core's fabric link builder: one runtime's fabric table (the Link*
// columns a plugin states on nxsGetRuntimeProperty, docs/Plugin_API.md
// "Fabric links") resolved onto the System's device list, on plain inputs so
// a test drives it without a device. The System feeds it each runtime's table
// in turn and wraps each result in a Link (src/system.cpp).

#include <knexus-api.h>

#include <string>
#include <vector>

#include "_link_impl.h"

namespace knexus {
namespace detail {

// One runtime's table as read off the wire, a vector per column. `type` and
// `device` give the rows; a shorter optional column reads as unstated.
struct FabricTable {
  std::vector<std::string> type;
  std::vector<nxs_long> device;
  std::vector<nxs_long> peer;
  std::vector<std::string> peer_id;
  std::vector<nxs_long> gen_max, gen_cur, width_max, width_cur;
};

// One fabric link. `a` and `b` index the System's device list; b == kNoDevice
// marks a second endpoint that resolves to no device (an empty Device).
struct FabricLink {
  static constexpr nxs_int kNoDevice = -1;
  nxs_int a = -1, b = kNoDevice;
  LinkFacts facts;
};

// `busIds` is the System's device list as bus ids (empty where a device states
// none); the runtime's devices are busIds[base .. base + size). Each row's
// first endpoint is its LinkDevice in the runtime (a row whose LinkDevice is
// out of range is dropped). The second is resolved by LinkPeer inside the
// runtime, else by LinkPeerId against every runtime's bus ids, else to no
// device. Rows keep their order; the builder never adds the reverse direction.
// A fabric link states no topology level.
std::vector<FabricLink> buildFabricLinks(const FabricTable &table, nxs_int base, nxs_int size,
                                         const std::vector<std::string> &busIds);

}  // namespace detail
}  // namespace knexus

#endif  // _KNEXUS_FABRIC_LINKS_H
