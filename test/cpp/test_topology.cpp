// The hwloc wrapper (include/knexus/topology.h) over a synthetic
// topology: no device and no arguments needed, so it runs as ctest registers
// it. hwloc reads HWLOC_SYNTHETIC itself at load, so each test sets it before
// the first load in its process.

#include <gtest/gtest.h>

#include <knexus/topology.h>

#include <stdlib.h>

using nxs::rt::TopologyIo;

namespace {

const char *kSynthetic = "pack:2 numa:1 core:4 pu:2";

void useSynthetic() { setenv("HWLOC_SYNTHETIC", kSynthetic, 1); }

int count(hwloc_topology_t topo, hwloc_obj_type_t type) {
  return hwloc_get_nbobjs_by_type(topo, type);
}

}  // namespace

TEST(Topology, SyntheticCounts) {
  useSynthetic();
  hwloc_topology_t topo = nxs::rt::topology();
  ASSERT_NE(topo, nullptr);
  EXPECT_EQ(count(topo, HWLOC_OBJ_PACKAGE), 2);
  EXPECT_EQ(count(topo, HWLOC_OBJ_NUMANODE), 2);
  EXPECT_EQ(count(topo, HWLOC_OBJ_CORE), 8);
  EXPECT_EQ(count(topo, HWLOC_OBJ_PU), 16);
  EXPECT_EQ(hwloc_topology_is_thissystem(topo), 0) << "synthetic, not this host";
}

TEST(Topology, LoadedOncePerPlugin) {
  useSynthetic();
  hwloc_topology_t first = nxs::rt::topology();
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(nxs::rt::topology(), first);
  EXPECT_EQ(nxs::rt::topology(TopologyIo::Pci), first) << "the first call fixes the filter";
}

TEST(Topology, FreshLoadEitherFilter) {
  useSynthetic();
  for (TopologyIo io : {TopologyIo::None, TopologyIo::Pci}) {
    hwloc_topology_t topo = nxs::rt::loadTopology(io);
    ASSERT_NE(topo, nullptr);
    EXPECT_EQ(count(topo, HWLOC_OBJ_PU), 16);
    EXPECT_EQ(count(topo, HWLOC_OBJ_PCI_DEVICE), 0) << "a synthetic topology has no I/O";
    EXPECT_NE(topo, nxs::rt::topology());
    hwloc_topology_destroy(topo);
  }
}

TEST(Topology, IncludesDisallowed) {
  useSynthetic();
  hwloc_topology_t topo = nxs::rt::topology();
  ASSERT_NE(topo, nullptr);
  EXPECT_TRUE(hwloc_topology_get_flags(topo) & HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED);
}

// hwloc does not fail on a malformed fixture: it falls back to this host.
TEST(Topology, MalformedSyntheticFallsBackToHost) {
  setenv("HWLOC_SYNTHETIC", "pack:x", 1);
  hwloc_topology_t topo = nxs::rt::loadTopology(TopologyIo::None);
  ASSERT_NE(topo, nullptr);
  EXPECT_EQ(hwloc_topology_is_thissystem(topo), 1);
  hwloc_topology_destroy(topo);
  useSynthetic();
}
