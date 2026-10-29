#ifndef KNEXUS_PCI_H
#define KNEXUS_PCI_H

// What the kernel states about a device's PCI node that hwloc does not: the
// sysfs class listing (sys/class/<cls>/<cls>!N, so N to bus id) and the
// link's PCIe generation and width, current and max (the four *_link_* files).
// hwloc states a link as one current GB/s figure (63.02 for gen5 x16), with
// no generation, lane count or maximum, and that figure is not decoded here
// (gen5 x8 and gen4 x16 are the same number); it has no OS-device object for
// a vendor's char device either. Everything else about the node (its NUMA
// node, its bridge ancestry, the path level between two nodes) comes from the
// hwloc topology: include/knexus/topology.h.
//
// Read once from sysfs under $KNEXUS_HOST_ROOT (default "/") so a test can
// point the reader at a fixture tree. Linux only: elsewhere the class lists
// nothing. Nothing here is measured; the names and units are those tabled in
// docs/Python_API.md under "Host and link properties". A vendor's char devices
// are listed by their sysfs class (e.g. "tenstorrent"), so the plugin names
// only the class. -1 where the kernel writes no such file or an unreadable
// value.

#include <knexus-api.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace nxs {
namespace rt {

struct PciNode {
  std::string bus_id;              // "0000:01:00.0", domain:bus:device.function
  nxs_long class_index = -1;       // N of the class entry (tenstorrent!N, /dev/tenstorrent/N)
  nxs_long gen_max = -1, gen_cur = -1;
  nxs_long width_max = -1, width_cur = -1;
};

namespace pci_detail {

inline std::string readLine(const std::filesystem::path &p) {
  std::ifstream in(p);
  std::string line;
  if (!in || !std::getline(in, line)) return std::string();
  line.erase(line.find_last_not_of(" \t\r\n") + 1);
  return line;
}

inline nxs_long readLong(const std::filesystem::path &p) {
  auto text = readLine(p);
  if (text.empty()) return -1;
  try {
    return std::stoll(text);
  } catch (...) {
    return -1;
  }
}

// "0000:01:00.0": four hex, colon, two hex, colon, two hex, dot, one hex
inline bool isBusId(const std::string &s) {
  if (s.size() != 12 || s[4] != ':' || s[7] != ':' || s[10] != '.') return false;
  for (size_t i = 0; i < s.size(); ++i)
    if (i != 4 && i != 7 && i != 10 && !std::isxdigit((unsigned char)s[i])) return false;
  return true;
}

// The trailing decimal of a class entry's name ("tenstorrent!3" -> 3), -1 if none.
inline nxs_long trailingNumber(const std::string &name) {
  size_t i = name.size();
  while (i > 0 && std::isdigit((unsigned char)name[i - 1])) --i;
  if (i == name.size()) return -1;
  try {
    return std::stoll(name.substr(i));
  } catch (...) {
    return -1;
  }
}

}  // namespace pci_detail

// sysfs's link speed ("32.0 GT/s PCIe") as a PCIe generation: 2.5 GT/s 1,
// 5 2, 8 3, 16 4, 32 5, 64 6; -1 for anything else ("Unknown").
inline nxs_long pciGeneration(const std::string &speed) {
  double gts = 0;
  try {
    gts = std::stod(speed);
  } catch (...) {
    return -1;
  }
  const double rates[] = {2.5, 5, 8, 16, 32, 64};
  for (int g = 0; g < 6; ++g)
    if (gts > rates[g] - 0.05 && gts < rates[g] + 0.05) return g + 1;
  return -1;
}

// The PCI node `device` resolves to (a class entry's `device` link or the
// node's own directory). False when it does not resolve to a PCI node.
inline bool readPciNode(const std::filesystem::path &device, PciNode &out) {
  using namespace pci_detail;
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path node = fs::canonical(device, ec);
  if (ec || !isBusId(node.filename().string())) return false;
  out.bus_id = node.filename().string();
  out.gen_max = pciGeneration(readLine(node / "max_link_speed"));
  out.gen_cur = pciGeneration(readLine(node / "current_link_speed"));
  out.width_max = readLong(node / "max_link_width");
  out.width_cur = readLong(node / "current_link_width");
  return true;
}

// Every entry of sys/class/<cls> under `root` that resolves to a PCI node,
// in ascending bus-id order. An entry without a PCI node is left out.
inline std::vector<PciNode> readPciClass(const std::filesystem::path &root,
                                         const std::string &cls) {
  namespace fs = std::filesystem;
  std::vector<PciNode> nodes;
  std::error_code ec;
  fs::path dir = root / "sys" / "class" / cls;
  if (!fs::is_directory(dir, ec)) return nodes;
  for (auto &entry : fs::directory_iterator(dir, ec)) {
    PciNode n;
    if (!readPciNode(entry.path() / "device", n)) continue;
    n.class_index = pci_detail::trailingNumber(entry.path().filename().string());
    nodes.push_back(n);
  }
  std::sort(nodes.begin(), nodes.end(),
            [](const PciNode &a, const PciNode &b) { return a.bus_id < b.bus_id; });
  return nodes;
}

// $KNEXUS_HOST_ROOT, or "/".
inline std::filesystem::path pciHostRoot() {
  const char *env = std::getenv("KNEXUS_HOST_ROOT");
  return env && *env ? env : "/";
}

}  // namespace rt
}  // namespace nxs

#endif  // KNEXUS_PCI_H
