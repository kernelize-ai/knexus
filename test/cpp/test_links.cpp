// The core's PCIe link builder (src/_pcie_links.h), device-free, so it runs as
// ctest registers it: test/fixtures/tt-quietbox.xml (the four-card host's hwloc
// export) plus a temp sysfs tree under a fake $KNEXUS_HOST_ROOT. The bus-id
// list stands for System::getDevices(): runtime A's four Tenstorrent chips,
// the cpu runtime's device (no PciBusId), then runtime B's two devices, one
// the fixture holds (the NIC 0000:0a:00.0, below the same host bridge as
// 0000:01:00.0) and one it lacks (0000:99:00.0). Also the fabric builder
// (src/_fabric_links.h) on a fake runtime table over the same device list, and
// the Link object itself.

#include <gtest/gtest.h>

#include <knexus/link.h>
#include <knexus/pci.h>
#include <knexus/topology.h>

#include "_fabric_links.h"
#include "_link_impl.h"
#include "_pcie_links.h"

#include <stdlib.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using knexus::detail::PcieLink;

namespace {

struct SysfsTree {
  fs::path root;
  SysfsTree() {
    std::string tmpl = (fs::temp_directory_path() / "knexus-links-XXXXXX").string();
    root = mkdtemp(tmpl.data());
  }
  ~SysfsTree() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
  static void put(const fs::path &p, const std::string &text) { std::ofstream(p) << text; }

  // sys/devices/<path...> with the four link files, and
  // sys/bus/pci/devices/<bus id> linking to it (relative, as sysfs does).
  void node(const std::vector<std::string> &path, const std::string &maxSpeed,
            const std::string &curSpeed, const std::string &maxWidth,
            const std::string &curWidth) {
    fs::path dev = root / "sys" / "devices";
    for (auto &a : path) dev /= a;
    fs::create_directories(dev);
    put(dev / "max_link_speed", maxSpeed + "\n");
    put(dev / "current_link_speed", curSpeed + "\n");
    put(dev / "max_link_width", maxWidth + "\n");
    put(dev / "current_link_width", curWidth + "\n");
    fs::path bus = root / "sys" / "bus" / "pci" / "devices";
    fs::create_directories(bus);
    fs::create_symlink(fs::relative(dev, bus), bus / path.back());
  }
};

hwloc_topology_t loadXml(const std::string &path) {
  setenv("HWLOC_XMLFILE", path.c_str(), 1);
  hwloc_topology_t topo = nxs::rt::loadTopology(nxs::rt::TopologyIo::Pci);
  unsetenv("HWLOC_XMLFILE");
  return topo;
}

struct Topo {
  hwloc_topology_t t;
  explicit Topo(const std::string &path) : t(loadXml(path)) {}
  ~Topo() {
    if (t) hwloc_topology_destroy(t);
  }
};

const std::string kQuietbox = std::string(KNEXUS_FIXTURE_DIR) + "/tt-quietbox.xml";

// runtime A (0-3), the cpu device (4), runtime B (5-6)
const std::vector<std::string> kBusIds = {"0000:01:00.0", "0000:41:00.0", "0000:42:00.0",
                                          "0000:c1:00.0", "",             "0000:0a:00.0",
                                          "0000:99:00.0"};

void liveHostTree(SysfsTree &s) {
  const std::string g5 = "32.0 GT/s PCIe";
  s.node({"pci0000:00", "0000:00:01.1", "0000:01:00.0"}, g5, g5, "16", "16");
  s.node({"pci0000:40", "0000:40:01.1", "0000:41:00.0"}, g5, g5, "16", "16");
  s.node({"pci0000:40", "0000:40:03.1", "0000:42:00.0"}, g5, g5, "16", "16");
  s.node({"pci0000:c0", "0000:c0:01.1", "0000:c1:00.0"}, g5, g5, "16", "16");
  // runtime B's devices: a gen4 x8 slot running at gen3 x4, and one hwloc lacks
  s.node({"pci0000:00", "0000:00:05.1", "0000:0a:00.0"}, "16.0 GT/s PCIe", "8.0 GT/s PCIe",
         "8", "4");
  s.node({"pci0000:80", "0000:80:01.1", "0000:99:00.0"}, "8.0 GT/s PCIe", "8.0 GT/s PCIe",
         "4", "4");
}

}  // namespace

TEST(Links, QuietboxTwoRuntimes) {
  Topo q(kQuietbox);
  ASSERT_NE(q.t, nullptr);
  ASSERT_EQ(hwloc_topology_is_thissystem(q.t), 0) << "the XML fixture, not this host";
  SysfsTree s;
  liveHostTree(s);
  auto links = knexus::detail::buildPcieLinks(kBusIds, q.t, s.root);

  std::vector<PcieLink> host, path;
  for (auto &l : links) (l.isHost() ? host : path).push_back(l);
  ASSERT_EQ(links.size(), 6u + 10u);
  ASSERT_EQ(host.size(), 6u) << "one per bus id; the cpu device has none";
  EXPECT_TRUE(std::equal(host.begin(), host.end(), links.begin(),
                         [](const PcieLink &x, const PcieLink &y) { return x.a == y.a; }))
      << "host links first, in device order";

  const nxs_int hostIdx[] = {0, 1, 2, 3, 5, 6};
  for (size_t k = 0; k < host.size(); ++k) {
    EXPECT_EQ(host[k].a, hostIdx[k]);
    EXPECT_EQ(host[k].level, -1) << "a host link has no level";
  }
  for (size_t k = 0; k < 4; ++k) {
    EXPECT_EQ(host[k].gen_max, 5);
    EXPECT_EQ(host[k].gen_cur, 5);
    EXPECT_EQ(host[k].width_max, 16);
    EXPECT_EQ(host[k].width_cur, 16);
  }
  EXPECT_EQ(host[4].gen_max, 4);
  EXPECT_EQ(host[4].gen_cur, 3);
  EXPECT_EQ(host[4].width_max, 8);
  EXPECT_EQ(host[4].width_cur, 4);
  EXPECT_EQ(host[5].gen_max, 3) << "sysfs states it though hwloc does not";
  EXPECT_EQ(host[5].width_cur, 4);

  // pairs once, i < j, none touching the cpu device or the bus id hwloc lacks
  std::map<std::pair<nxs_int, nxs_int>, nxs_long> level;
  for (auto &l : path) {
    EXPECT_LT(l.a, l.b);
    EXPECT_NE(l.a, 4);
    EXPECT_NE(l.b, 4);
    EXPECT_NE(l.b, 6) << "0000:99:00.0 keeps its host link and has no path links";
    EXPECT_EQ(l.gen_max, -1);
    EXPECT_EQ(l.gen_cur, -1);
    EXPECT_EQ(l.width_max, -1);
    EXPECT_EQ(l.width_cur, -1);
    EXPECT_TRUE(level.emplace(std::make_pair(l.a, l.b), l.level).second) << "a pair repeats";
  }
  const std::map<std::pair<nxs_int, nxs_int>, nxs_long> want = {
      {{0, 1}, 40}, {{0, 2}, 40}, {{0, 3}, 40}, {{1, 2}, 30}, {{1, 3}, 40}, {{2, 3}, 40},
      // across runtimes: the NIC sits below host bridge 0000:[00-0c] with chip 0
      {{0, 5}, 30}, {{1, 5}, 40}, {{2, 5}, 40}, {{3, 5}, 40}};
  EXPECT_EQ(level, want);
}

TEST(Links, NoSysfsNoTopology) {
  SysfsTree empty;
  // no topology: host links only, every figure -1 without sysfs files
  auto links = knexus::detail::buildPcieLinks(kBusIds, nullptr, empty.root);
  ASSERT_EQ(links.size(), 6u);
  for (auto &l : links) {
    EXPECT_TRUE(l.isHost());
    EXPECT_EQ(l.gen_max, -1);
    EXPECT_EQ(l.gen_cur, -1);
    EXPECT_EQ(l.width_max, -1);
    EXPECT_EQ(l.width_cur, -1);
    EXPECT_EQ(l.level, -1);
  }
  EXPECT_TRUE(knexus::detail::buildPcieLinks({}, nullptr, empty.root).empty());
  EXPECT_TRUE(knexus::detail::buildPcieLinks({"", ""}, nullptr, empty.root).empty());
}

TEST(Links, LinkObject) {
  knexus::detail::LinkFacts facts;
  facts.type = "PCIe";
  facts.level = 30;
  knexus::Link link(knexus::detail::Impl(nullptr, 7), knexus::Device(), knexus::Device(), facts);
  ASSERT_TRUE(link);
  EXPECT_EQ(link.getId(), 7);
  EXPECT_EQ(link.getProp<std::string>(NP_Type), "PCIe");
  EXPECT_EQ(link.getProp<nxs_long>(NP_LinkTopologyLevel), 30);
  EXPECT_EQ(link.getProp<nxs_long>(NP_LinkGenMax), -1);
  EXPECT_EQ(link.getProp<nxs_long>(NP_LinkGenCurrent), -1);
  EXPECT_EQ(link.getProp<nxs_long>(NP_LinkWidthMax), -1);
  EXPECT_EQ(link.getProp<nxs_long>(NP_LinkWidthCurrent), -1);
  EXPECT_FALSE(link.getProperty(NP_Name).has_value());
  EXPECT_FALSE(link.getProperty(NP_LinkPeer).has_value()) << "no per-device row name";
  auto keys = link.getProperty(NP_Keys);
  ASSERT_TRUE(keys.has_value());
  const std::vector<nxs_long> want = {NP_Type, NP_LinkGenMax, NP_LinkGenCurrent, NP_LinkWidthMax,
                                      NP_LinkWidthCurrent, NP_LinkTopologyLevel};
  EXPECT_EQ(keys->getValueVec<nxs_long>(), want);
  auto ends = link.getDevices();
  ASSERT_EQ(ends.size(), 2);
  EXPECT_FALSE(ends[0]);
  EXPECT_FALSE(ends[1]);
}

// Runtime A's fabric table over kBusIds (A is devices 0-3, base 0): pairs
// resolved by LinkPeer inside the runtime, one by LinkPeerId to runtime B's
// NIC, one whose peer is nothing the System lists, one whose LinkDevice is out
// of range (dropped). Then runtime B's table (base 5, size 2) naming chip 0 of
// runtime A by bus id. The builder keeps the plugin's direction and order.
TEST(Links, FabricTable) {
  knexus::detail::FabricTable a;
  a.type = {"Ethernet", "Ethernet", "Ethernet", "NVLink", "Ethernet", "Ethernet"};
  a.device = {0, 0, 1, 2, 3, 4};
  a.peer = {2, 2, 3, -1, -1, 0};
  a.peer_id = {"0000:42:00.0", "", "0000:c1:00.0", "0000:0a:00.0", "0000:ff:00.0", ""};
  a.gen_max = {-1, -1, -1, 4, -1, -1};
  a.gen_cur = {-1, -1, -1, 4};  // shorter: the rest read as unstated
  auto links = knexus::detail::buildFabricLinks(a, 0, 4, kBusIds);
  ASSERT_EQ(links.size(), 5u) << "LinkDevice 4 is outside a 4-device runtime";
  const std::vector<std::pair<nxs_int, nxs_int>> want = {
      {0, 2}, {0, 2}, {1, 3}, {2, 5}, {3, knexus::detail::FabricLink::kNoDevice}};
  for (size_t k = 0; k < links.size(); ++k) {
    EXPECT_EQ(std::make_pair(links[k].a, links[k].b), want[k]) << "row " << k;
    EXPECT_EQ(links[k].facts.level, -1) << "a fabric link has no level";
    EXPECT_EQ(links[k].facts.width_max, -1);
  }
  EXPECT_EQ(links[0].facts.type, "Ethernet");
  EXPECT_EQ(links[3].facts.type, "NVLink");
  EXPECT_EQ(links[3].facts.gen_max, 4);
  EXPECT_EQ(links[3].facts.gen_cur, 4);
  EXPECT_EQ(links[4].facts.gen_cur, -1);

  knexus::detail::FabricTable b;
  b.type = {"NVLink", "NVLink"};
  b.device = {0, 1};
  b.peer = {-1, 0};
  b.peer_id = {"0000:01:00.0", ""};
  auto bl = knexus::detail::buildFabricLinks(b, 5, 2, kBusIds);
  ASSERT_EQ(bl.size(), 2u);
  EXPECT_EQ(std::make_pair(bl[0].a, bl[0].b), std::make_pair(5, 0)) << "across runtimes by bus id";
  EXPECT_EQ(std::make_pair(bl[1].a, bl[1].b), std::make_pair(6, 5)) << "by index, as stated";

  EXPECT_TRUE(knexus::detail::buildFabricLinks({}, 0, 4, kBusIds).empty()) << "no table";
}
