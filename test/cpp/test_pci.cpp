// The tenstorrent plugin's PCI node, device-free, so it runs as ctest
// registers it. Two halves:
// - the sysfs sliver (plugins/include/rt_pci.h: the class listing and the
//   link's generation and width) over a tiny fixture tree;
// - hwloc's PCI objects (plugins/include/rt_topology.h: bus id to object,
//   NUMA node, path level) over hwloc XML: test/fixtures/tt-quietbox.xml,
//   exported from the four-card host this was written on, and a hand-written
//   topology for the levels that host cannot show (10, 20, 50).
// hwloc reads HWLOC_XMLFILE at load, so each case sets it just before a fresh
// loadTopology(); a malformed file would silently load this host instead, so
// every XML case checks hwloc_topology_is_thissystem() is 0.

#include <gtest/gtest.h>

#include "../../plugins/include/rt_pci.h"
#include "../../plugins/include/rt_topology.h"

#include <stdlib.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using nxs::rt::TopologyIo;

namespace {

struct Fixture {
  fs::path root;
  Fixture() {
    std::string tmpl = (fs::temp_directory_path() / "knexus-pci-XXXXXX").string();
    root = mkdtemp(tmpl.data());
  }
  ~Fixture() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
  static void put(const fs::path &p, const std::string &text) { std::ofstream(p) << text; }

  // sys/devices/<path...> with the four link files, and
  // sys/class/<cls>/<cls>!<n>/device linking to it (relative, as sysfs does).
  fs::path node(const std::string &cls, int n, const std::vector<std::string> &path,
                bool files = true, const std::string &speed = "32.0 GT/s PCIe",
                const std::string &width = "16") {
    fs::path dev = root / "sys" / "devices";
    for (auto &a : path) dev /= a;
    fs::create_directories(dev);
    if (files) {
      put(dev / "current_link_speed", speed + "\n");
      put(dev / "max_link_speed", "32.0 GT/s PCIe\n");
      put(dev / "current_link_width", width + "\n");
      put(dev / "max_link_width", "16\n");
    }
    fs::path entry = root / "sys" / "class" / cls / (cls + "!" + std::to_string(n));
    fs::create_directories(entry);
    fs::create_symlink(fs::relative(dev, entry), entry / "device");
    return dev;
  }

  // The four Blackhole p150b of tt-quietbox: /dev N order is not bus-id order.
  void liveHost() {
    node("tenstorrent", 0, {"pci0000:c0", "0000:c0:01.1", "0000:c1:00.0"});
    node("tenstorrent", 1, {"pci0000:00", "0000:00:01.1", "0000:01:00.0"});
    node("tenstorrent", 2, {"pci0000:40", "0000:40:01.1", "0000:41:00.0"});
    node("tenstorrent", 3, {"pci0000:40", "0000:40:03.1", "0000:42:00.0"});
  }
};

// A fresh TopologyIo::Pci topology from the hwloc XML at `path`.
hwloc_topology_t loadXml(const std::string &path) {
  setenv("HWLOC_XMLFILE", path.c_str(), 1);
  hwloc_topology_t topo = nxs::rt::loadTopology(TopologyIo::Pci);
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
const char *kQuietboxBus[] = {"0000:01:00.0", "0000:41:00.0", "0000:42:00.0", "0000:c1:00.0"};

// Two packages, one NUMA node each. Package 0 has a two-level switch below
// host bridge [00-0f] (upstream port 00:01.1, downstream ports 01:00.0,
// 01:01.0, 01:02.0, the last one leading to a second switch) and a second host
// bridge [40-4f] with one root port; package 1 has one host bridge [80-8f]
// with one root port. Every endpoint is a class 1200 accelerator.
const char *kHandXml = R"(<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE topology SYSTEM "hwloc2.dtd">
<topology version="2.0">
  <object type="Machine" os_index="0" cpuset="0x00000003" complete_cpuset="0x00000003" allowed_cpuset="0x00000003" nodeset="0x00000003" complete_nodeset="0x00000003" allowed_nodeset="0x00000003" gp_index="1">
    <object type="Package" os_index="0" cpuset="0x00000001" complete_cpuset="0x00000001" nodeset="0x00000001" complete_nodeset="0x00000001" gp_index="2">
      <object type="NUMANode" os_index="0" cpuset="0x00000001" complete_cpuset="0x00000001" nodeset="0x00000001" complete_nodeset="0x00000001" gp_index="3" local_memory="1073741824"/>
      <object type="Core" os_index="0" cpuset="0x00000001" complete_cpuset="0x00000001" nodeset="0x00000001" complete_nodeset="0x00000001" gp_index="4">
        <object type="PU" os_index="0" cpuset="0x00000001" complete_cpuset="0x00000001" nodeset="0x00000001" complete_nodeset="0x00000001" gp_index="5"/>
      </object>
      <object type="Bridge" gp_index="20" bridge_type="0-1" depth="0" bridge_pci="0000:[00-0f]">
        <object type="Bridge" gp_index="21" bridge_type="1-1" depth="1" bridge_pci="0000:[01-0f]" pci_busid="0000:00:01.1" pci_type="0604 [1022:14ab] [1022:14a5] 01" pci_link_speed="0.000000">
          <object type="Bridge" gp_index="22" bridge_type="1-1" depth="2" bridge_pci="0000:[02-02]" pci_busid="0000:01:00.0" pci_type="0604 [10b5:8747] [10b5:8747] 01" pci_link_speed="0.000000">
            <object type="PCIDev" gp_index="23" pci_busid="0000:02:00.0" pci_type="1200 [1e52:b140] [1e52:0041] 00" pci_link_speed="0.000000"/>
          </object>
          <object type="Bridge" gp_index="24" bridge_type="1-1" depth="2" bridge_pci="0000:[03-03]" pci_busid="0000:01:01.0" pci_type="0604 [10b5:8747] [10b5:8747] 01" pci_link_speed="0.000000">
            <object type="PCIDev" gp_index="25" pci_busid="0000:03:00.0" pci_type="1200 [1e52:b140] [1e52:0041] 00" pci_link_speed="0.000000"/>
          </object>
          <object type="Bridge" gp_index="26" bridge_type="1-1" depth="2" bridge_pci="0000:[04-06]" pci_busid="0000:01:02.0" pci_type="0604 [10b5:8747] [10b5:8747] 01" pci_link_speed="0.000000">
            <object type="Bridge" gp_index="27" bridge_type="1-1" depth="3" bridge_pci="0000:[05-06]" pci_busid="0000:04:00.0" pci_type="0604 [10b5:8747] [10b5:8747] 01" pci_link_speed="0.000000">
              <object type="Bridge" gp_index="28" bridge_type="1-1" depth="4" bridge_pci="0000:[06-06]" pci_busid="0000:05:00.0" pci_type="0604 [10b5:8747] [10b5:8747] 01" pci_link_speed="0.000000">
                <object type="PCIDev" gp_index="29" pci_busid="0000:06:00.0" pci_type="1200 [1e52:b140] [1e52:0041] 00" pci_link_speed="0.000000"/>
              </object>
            </object>
          </object>
        </object>
      </object>
      <object type="Bridge" gp_index="30" bridge_type="0-1" depth="0" bridge_pci="0000:[40-4f]">
        <object type="Bridge" gp_index="31" bridge_type="1-1" depth="1" bridge_pci="0000:[41-41]" pci_busid="0000:40:01.1" pci_type="0604 [1022:14ab] [1022:14a5] 01" pci_link_speed="0.000000">
          <object type="PCIDev" gp_index="32" pci_busid="0000:41:00.0" pci_type="1200 [1e52:b140] [1e52:0041] 00" pci_link_speed="0.000000"/>
        </object>
      </object>
    </object>
    <object type="Package" os_index="1" cpuset="0x00000002" complete_cpuset="0x00000002" nodeset="0x00000002" complete_nodeset="0x00000002" gp_index="6">
      <object type="NUMANode" os_index="1" cpuset="0x00000002" complete_cpuset="0x00000002" nodeset="0x00000002" complete_nodeset="0x00000002" gp_index="7" local_memory="1073741824"/>
      <object type="Core" os_index="1" cpuset="0x00000002" complete_cpuset="0x00000002" nodeset="0x00000002" complete_nodeset="0x00000002" gp_index="8">
        <object type="PU" os_index="1" cpuset="0x00000002" complete_cpuset="0x00000002" nodeset="0x00000002" complete_nodeset="0x00000002" gp_index="9"/>
      </object>
      <object type="Bridge" gp_index="40" bridge_type="0-1" depth="0" bridge_pci="0000:[80-8f]">
        <object type="Bridge" gp_index="41" bridge_type="1-1" depth="1" bridge_pci="0000:[81-81]" pci_busid="0000:80:01.1" pci_type="0604 [1022:14ab] [1022:14a5] 01" pci_link_speed="0.000000">
          <object type="PCIDev" gp_index="42" pci_busid="0000:81:00.0" pci_type="1200 [1e52:b140] [1e52:0041] 00" pci_link_speed="0.000000"/>
        </object>
      </object>
    </object>
  </object>
</topology>
)";

std::string upstreamBusId(hwloc_obj_t bridge) {
  char s[16];
  const auto &u = bridge->attr->bridge.upstream.pci;
  snprintf(s, sizeof s, "%04x:%02x:%02x.%01x", u.domain, u.bus, u.dev, u.func);
  return s;
}

}  // namespace

// --- the sysfs sliver ---------------------------------------------------

TEST(Pci, Generation) {
  EXPECT_EQ(nxs::rt::pciGeneration("2.5 GT/s PCIe"), 1);
  EXPECT_EQ(nxs::rt::pciGeneration("5.0 GT/s PCIe"), 2);
  EXPECT_EQ(nxs::rt::pciGeneration("8.0 GT/s PCIe"), 3);
  EXPECT_EQ(nxs::rt::pciGeneration("16.0 GT/s PCIe"), 4);
  EXPECT_EQ(nxs::rt::pciGeneration("32.0 GT/s PCIe"), 5);
  EXPECT_EQ(nxs::rt::pciGeneration("64.0 GT/s PCIe"), 6);
  EXPECT_EQ(nxs::rt::pciGeneration("Unknown"), -1);
  EXPECT_EQ(nxs::rt::pciGeneration(""), -1);
  EXPECT_EQ(nxs::rt::pciGeneration("4.0 GT/s"), -1);
}

TEST(Pci, ClassListing) {
  Fixture f;
  f.liveHost();
  auto nodes = nxs::rt::readPciClass(f.root, "tenstorrent");
  ASSERT_EQ(nodes.size(), 4u);
  const nxs_long index[] = {1, 2, 3, 0};
  for (size_t k = 0; k < 4; ++k) {
    EXPECT_EQ(nodes[k].bus_id, kQuietboxBus[k]) << "chip " << k << " is the k-th in bus-id order";
    EXPECT_EQ(nodes[k].class_index, index[k]);
    EXPECT_EQ(nodes[k].gen_cur, 5);
    EXPECT_EQ(nodes[k].gen_max, 5);
    EXPECT_EQ(nodes[k].width_cur, 16);
    EXPECT_EQ(nodes[k].width_max, 16);
  }
}

TEST(Pci, MissingFiles) {
  Fixture f;
  f.node("tenstorrent", 0, {"pci0000:00", "0000:00:01.1", "0000:01:00.0"}, false);
  auto nodes = nxs::rt::readPciClass(f.root, "tenstorrent");
  ASSERT_EQ(nodes.size(), 1u);
  EXPECT_EQ(nodes[0].bus_id, "0000:01:00.0");
  EXPECT_EQ(nodes[0].class_index, 0);
  EXPECT_EQ(nodes[0].gen_cur, -1);
  EXPECT_EQ(nodes[0].gen_max, -1);
  EXPECT_EQ(nodes[0].width_cur, -1);
  EXPECT_EQ(nodes[0].width_max, -1);
}

TEST(Pci, UnreadableValues) {
  Fixture f;
  f.node("tenstorrent", 0, {"pci0000:00", "0000:00:01.1", "0000:01:00.0"}, true, "Unknown", "x");
  auto nodes = nxs::rt::readPciClass(f.root, "tenstorrent");
  ASSERT_EQ(nodes.size(), 1u);
  EXPECT_EQ(nodes[0].gen_cur, -1);
  EXPECT_EQ(nodes[0].gen_max, 5);
  EXPECT_EQ(nodes[0].width_cur, -1);
  EXPECT_EQ(nodes[0].width_max, 16);
}

TEST(Pci, NoPciNode) {
  Fixture f;
  // an entry whose device link dangles, one with no device link, one that
  // resolves to a directory not named like a bus id, and an empty root: none
  // is a PCI node
  fs::path entry = f.root / "sys" / "class" / "tenstorrent" / "tenstorrent!0";
  fs::create_directories(entry);
  fs::create_symlink("../../../devices/pci0000:00/0000:00:01.1/0000:01:00.0", entry / "device");
  fs::create_directories(f.root / "sys" / "class" / "tenstorrent" / "tenstorrent!1");
  f.node("tenstorrent", 2, {"platform", "tt-fake"});
  EXPECT_TRUE(nxs::rt::readPciClass(f.root, "tenstorrent").empty());
  EXPECT_TRUE(nxs::rt::readPciClass(f.root, "absent").empty());
  Fixture empty;
  EXPECT_TRUE(nxs::rt::readPciClass(empty.root, "tenstorrent").empty());
}

// --- hwloc's PCI objects --------------------------------------------------

TEST(Pci, QuietboxFixtureLoads) {
  Topo q(kQuietbox);
  ASSERT_NE(q.t, nullptr);
  EXPECT_EQ(hwloc_topology_is_thissystem(q.t), 0) << "the XML fixture, not this host";
  const char *host = hwloc_obj_get_info_by_name(hwloc_get_root_obj(q.t), "HostName");
  ASSERT_NE(host, nullptr);
  EXPECT_STREQ(host, "tt-quietbox");
  int cards = 0;
  for (hwloc_obj_t o = nullptr; (o = hwloc_get_next_pcidev(q.t, o));)
    if (o->attr->pcidev.vendor_id == 0x1e52 && o->attr->pcidev.device_id == 0xb140) ++cards;
  EXPECT_EQ(cards, 4) << "KEEP_IMPORTANT keeps every class 1200 card";
}

TEST(Pci, QuietboxAncestry) {
  Topo q(kQuietbox);
  ASSERT_NE(q.t, nullptr);
  ASSERT_EQ(hwloc_topology_is_thissystem(q.t), 0);
  const char *port[] = {"0000:00:01.1", "0000:40:01.1", "0000:40:03.1", "0000:c0:01.1"};
  hwloc_obj_t host_bridge[4] = {};
  for (size_t k = 0; k < 4; ++k) {
    hwloc_obj_t o = nxs::rt::pciObject(q.t, kQuietboxBus[k]);
    ASSERT_NE(o, nullptr) << kQuietboxBus[k];
    EXPECT_EQ(o->attr->pcidev.class_id, 0x1200);
    hwloc_obj_t rp = o->parent;  // the root port
    ASSERT_EQ(rp->type, HWLOC_OBJ_BRIDGE);
    ASSERT_EQ(rp->attr->bridge.upstream_type, HWLOC_OBJ_BRIDGE_PCI);
    EXPECT_EQ(upstreamBusId(rp), port[k]);
    host_bridge[k] = rp->parent;
    ASSERT_EQ(host_bridge[k]->type, HWLOC_OBJ_BRIDGE);
    EXPECT_EQ(host_bridge[k]->attr->bridge.upstream_type, HWLOC_OBJ_BRIDGE_HOST);
    EXPECT_EQ(host_bridge[k]->parent->type, HWLOC_OBJ_MACHINE);
    EXPECT_EQ(hwloc_get_non_io_ancestor_obj(q.t, o), hwloc_get_root_obj(q.t));
    EXPECT_EQ(nxs::rt::pciNumaNode(q.t, o), 0);
  }
  EXPECT_EQ(host_bridge[1], host_bridge[2]) << "41 and 42 share root complex 0000:[40-43]";
  EXPECT_NE(host_bridge[0], host_bridge[1]);
  EXPECT_NE(host_bridge[0], host_bridge[3]);
  EXPECT_NE(host_bridge[1], host_bridge[3]);
}

TEST(Pci, QuietboxLevels) {
  Topo q(kQuietbox);
  ASSERT_NE(q.t, nullptr);
  ASSERT_EQ(hwloc_topology_is_thissystem(q.t), 0);
  hwloc_obj_t o[4];
  for (size_t k = 0; k < 4; ++k) ASSERT_NE(o[k] = nxs::rt::pciObject(q.t, kQuietboxBus[k]), nullptr);
  // 1-2 share a host bridge (30); every other pair the one-node Machine (40)
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 4; ++j) {
      nxs_long want = i == j ? -1 : ((i == 1 && j == 2) || (i == 2 && j == 1)) ? 30 : 40;
      EXPECT_EQ(nxs::rt::pciTopologyLevel(q.t, o[i], o[j]), want) << i << "-" << j;
    }
}

TEST(Pci, UnknownBusIdStatesNothing) {
  Topo q(kQuietbox);
  ASSERT_NE(q.t, nullptr);
  ASSERT_EQ(hwloc_topology_is_thissystem(q.t), 0);
  hwloc_obj_t known = nxs::rt::pciObject(q.t, kQuietboxBus[0]);
  ASSERT_NE(known, nullptr);
  EXPECT_EQ(nxs::rt::pciObject(q.t, "0000:99:00.0"), nullptr) << "no such device";
  EXPECT_EQ(nxs::rt::pciObject(q.t, ""), nullptr);
  EXPECT_EQ(nxs::rt::pciObject(nullptr, kQuietboxBus[0]), nullptr);
  EXPECT_EQ(nxs::rt::pciNumaNode(q.t, nullptr), -1);
  EXPECT_EQ(nxs::rt::pciTopologyLevel(q.t, known, nullptr), -1);
  EXPECT_EQ(nxs::rt::pciTopologyLevel(q.t, nullptr, known), -1);
  EXPECT_EQ(nxs::rt::pciTopologyLevel(nullptr, known, known), -1);
}

TEST(Pci, HandWrittenLevels) {
  Fixture f;
  fs::path xml = f.root / "levels.xml";
  Fixture::put(xml, kHandXml);
  Topo h(xml.string());
  ASSERT_NE(h.t, nullptr);
  ASSERT_EQ(hwloc_topology_is_thissystem(h.t), 0);
  ASSERT_EQ(hwloc_get_nbobjs_by_type(h.t, HWLOC_OBJ_NUMANODE), 2) << "the fixture, not this host";
  auto obj = [&](const char *bus) {
    hwloc_obj_t o = nxs::rt::pciObject(h.t, bus);
    EXPECT_NE(o, nullptr) << bus;
    return o;
  };
  hwloc_obj_t a = obj("0000:02:00.0"), b = obj("0000:03:00.0"), c = obj("0000:06:00.0");
  hwloc_obj_t e = obj("0000:41:00.0"), d = obj("0000:81:00.0");
  ASSERT_TRUE(a && b && c && d && e);
  auto level = [&](hwloc_obj_t x, hwloc_obj_t y) {
    nxs_long l = nxs::rt::pciTopologyLevel(h.t, x, y);
    EXPECT_EQ(l, nxs::rt::pciTopologyLevel(h.t, y, x)) << "symmetric";
    return l;
  };
  EXPECT_EQ(level(a, b), 10) << "one switch: the upstream port, one downstream port each";
  EXPECT_EQ(level(a, c), 20) << "c sits behind a second switch";
  EXPECT_EQ(level(b, c), 20);
  EXPECT_EQ(level(a, e), 40) << "two host bridges on package 0, one NUMA node";
  EXPECT_EQ(level(c, e), 40) << "I/O chains of different lengths";
  EXPECT_EQ(level(a, d), 50) << "packages 0 and 1, two NUMA nodes";
  EXPECT_EQ(level(c, d), 50);
  EXPECT_EQ(level(e, d), 50);
  EXPECT_EQ(nxs::rt::pciNumaNode(h.t, a), 0);
  EXPECT_EQ(nxs::rt::pciNumaNode(h.t, c), 0);
  EXPECT_EQ(nxs::rt::pciNumaNode(h.t, e), 0);
  EXPECT_EQ(nxs::rt::pciNumaNode(h.t, d), 1);
}
