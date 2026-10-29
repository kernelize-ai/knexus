#include "_fabric_links.h"

#include <algorithm>

using namespace knexus::detail;

namespace {
template <typename T>
T at(const std::vector<T> &col, size_t k, T unstated) {
  return k < col.size() ? col[k] : unstated;
}
}  // namespace

std::vector<FabricLink> knexus::detail::buildFabricLinks(
    const FabricTable &table, nxs_int base, nxs_int size,
    const std::vector<std::string> &busIds) {
  std::vector<FabricLink> links;
  size_t rows = std::min(table.type.size(), table.device.size());
  for (size_t k = 0; k < rows; ++k) {
    nxs_long dev = table.device[k];
    if (dev < 0 || dev >= size) continue;
    FabricLink l;
    l.a = base + (nxs_int)dev;
    nxs_long peer = at<nxs_long>(table.peer, k, -1);
    std::string peer_id = at<std::string>(table.peer_id, k, "");
    if (peer >= 0 && peer < size && peer != dev) {
      l.b = base + (nxs_int)peer;
    } else if (!peer_id.empty()) {
      for (size_t j = 0; j < busIds.size(); ++j)
        if ((nxs_int)j != l.a && busIds[j] == peer_id) {
          l.b = (nxs_int)j;
          break;
        }
    }
    l.facts.type = table.type[k];
    l.facts.gen_max = at<nxs_long>(table.gen_max, k, -1);
    l.facts.gen_cur = at<nxs_long>(table.gen_cur, k, -1);
    l.facts.width_max = at<nxs_long>(table.width_max, k, -1);
    l.facts.width_cur = at<nxs_long>(table.width_cur, k, -1);
    links.push_back(l);
  }
  return links;
}
