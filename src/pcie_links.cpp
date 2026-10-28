#include <knexus/pci.h>
#include <knexus/topology.h>

#include "_pcie_links.h"

using namespace knexus::detail;

std::vector<PcieLink> knexus::detail::buildPcieLinks(
    const std::vector<std::string> &busIds, hwloc_topology_t topo,
    const std::filesystem::path &hostRoot) {
  std::vector<PcieLink> links;
  std::vector<hwloc_obj_t> objs(busIds.size(), nullptr);
  for (size_t i = 0; i < busIds.size(); ++i) {
    if (busIds[i].empty()) continue;
    PcieLink host;
    host.a = (nxs_int)i;
    nxs::rt::PciNode node;
    if (nxs::rt::readPciNode(hostRoot / "sys" / "bus" / "pci" / "devices" / busIds[i], node)) {
      host.gen_max = node.gen_max;
      host.gen_cur = node.gen_cur;
      host.width_max = node.width_max;
      host.width_cur = node.width_cur;
    }
    links.push_back(host);
    objs[i] = nxs::rt::pciObject(topo, busIds[i]);
  }
  for (size_t i = 0; i < busIds.size(); ++i) {
    if (!objs[i]) continue;
    for (size_t j = i + 1; j < busIds.size(); ++j) {
      if (!objs[j]) continue;
      PcieLink path;
      path.a = (nxs_int)i;
      path.b = (nxs_int)j;
      path.level = nxs::rt::pciTopologyLevel(topo, objs[i], objs[j]);
      links.push_back(path);
    }
  }
  return links;
}
