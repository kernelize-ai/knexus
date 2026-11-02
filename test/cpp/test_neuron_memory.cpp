// The neuron plugin's memory sizes (plugins/neuron/README.md "Properties"):
// GlobalMemorySize per logical core, CoreMemorySize (SBUF) and
// CoreRegisterSize (PSUM) per physical core. Skipped where there is no
// neuron runtime or device.
//
//   ./test/cpp/test_neuron_memory
//   KNEXUS_NEURON_LIVE=1 ./test/cpp/test_neuron_memory   # also the live leg
//
// The live leg creates a buffer on device 0, which runs nrt_init (15-20 s)
// and claims every visible core, then checks GlobalMemorySize against
// libnrt's own nrt_get_vnc_memory_stats; it is skipped unless
// KNEXUS_NEURON_LIVE is set.
#include <gtest/gtest.h>
#include <knexus.h>

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr nxs_long KiB = 1024, MiB = KiB * KiB, GiB = MiB * KiB;

// The figures AWS's sources declare, restated here so the test does not read
// them from the plugin: the chip's HBM (aws-neuronx-dkms address maps) and
// one physical core's SBUF (nki's _SBUF_SIZE_BYTES).
struct Expected {
  nxs_long chip_hbm, sbuf;
};
const std::map<std::string, Expected> kExpected = {
    {"trn1", {32 * GiB, 24 * MiB}},
    {"inf2", {32 * GiB, 24 * MiB}},
    {"trn2", {96 * GiB, 28 * MiB}},
    {"trn3", {144 * GiB, 32 * MiB}},
};

class NeuronMemory : public ::testing::Test {
 protected:
  knexus::Runtime rt;
  std::vector<knexus::Device> devs;

  void SetUp() override {
    rt = knexus::getSystem().getRuntime("neuron");
    if (!rt || rt.getDevices().empty()) GTEST_SKIP() << "no neuron runtime or device";
    for (auto d : rt.getDevices()) devs.push_back(d);
  }

  static bool listed(knexus::Device &d, nxs_int prop) {
    for (nxs_long k : d.getProp<std::vector<nxs_long>>(NP_Keys))
      if (k == prop) return true;
    return false;
  }
};

}  // namespace

TEST_F(NeuronMemory, SizesStated) {
  for (auto &d : devs) {
    auto arch = d.getProp<std::string>(NP_Architecture);
    auto it = kExpected.find(arch);
    if (it == kExpected.end()) {
      // An architecture the table does not hold states no SBUF or PSUM.
      EXPECT_FALSE(d.getProperty(NP_CoreMemorySize)) << arch;
      EXPECT_FALSE(d.getProperty(NP_CoreRegisterSize)) << arch;
      continue;
    }
    for (nxs_int p : {NP_GlobalMemorySize, NP_CoreMemorySize, NP_CoreRegisterSize}) {
      ASSERT_TRUE(d.getProperty(p)) << nxsGetPropName(p);
      EXPECT_TRUE(listed(d, p)) << nxsGetPropName(p) << " not in Keys";
      EXPECT_GT(d.getProp<nxs_long>(p), 0) << nxsGetPropName(p);
    }
    EXPECT_EQ(d.getProp<nxs_long>(NP_CoreRegisterSize), 2 * MiB);
    EXPECT_EQ(d.getProp<nxs_long>(NP_CoreMemorySize), it->second.sbuf) << arch;
  }
}

// The logical cores of a chip (the devices sharing a PciBusId) split its HBM
// evenly. Before nrt_init this is the table's figure; after it, libnrt's.
TEST_F(NeuronMemory, CoresShareChipHbm) {
  std::map<std::string, std::vector<nxs_long>> per_chip;
  std::string arch;
  for (auto &d : devs) {
    if (!d.getProperty(NP_PciBusId) || !d.getProperty(NP_GlobalMemorySize)) continue;
    arch = d.getProp<std::string>(NP_Architecture);
    per_chip[d.getProp<std::string>(NP_PciBusId)].push_back(
        d.getProp<nxs_long>(NP_GlobalMemorySize));
  }
  if (per_chip.empty() || !kExpected.count(arch))
    GTEST_SKIP() << "no device states PciBusId and GlobalMemorySize";
  // With NEURON_RT_VISIBLE_CORES a chip may show only some of its cores.
  if (std::getenv("NEURON_RT_VISIBLE_CORES")) GTEST_SKIP() << "NEURON_RT_VISIBLE_CORES is set";
  for (auto &[bus, mems] : per_chip) {
    nxs_long total = 0;
    for (nxs_long m : mems) {
      EXPECT_EQ(m, mems[0]) << bus;
      total += m;
    }
    std::cout << bus << ": " << mems.size() << " cores x " << mems[0] << " bytes" << std::endl;
    EXPECT_EQ(total, kExpected.at(arch).chip_hbm) << bus;
  }
}

// libnrt's nrt_vnc_memory_stats_t (nrt/nrt.h), restated so the test needs no
// Neuron headers.
struct VncMemoryStats {
  size_t bytes_used;
  size_t bytes_limit;
};
using GetVncMemoryStats = int (*)(uint32_t, VncMemoryStats *, size_t, size_t *);

TEST_F(NeuronMemory, LiveMatchesLibnrt) {
  if (!std::getenv("KNEXUS_NEURON_LIVE"))
    GTEST_SKIP() << "set KNEXUS_NEURON_LIVE=1 (runs nrt_init, claims the cores)";
  auto dev = devs[0];
  nxs_long before = dev.getProp<nxs_long>(NP_GlobalMemorySize);
  auto buf = dev.createBuffer(4096);
  ASSERT_TRUE(buf) << "createBuffer failed (cores held by another process?)";
  nxs_long after = dev.getProp<nxs_long>(NP_GlobalMemorySize);

  // The plugin's libnrt, already loaded and initialized in this process.
  void *nrt = dlopen("libnrt.so.1", RTLD_NOW | RTLD_NOLOAD);
  ASSERT_NE(nrt, nullptr) << "libnrt.so.1 is not loaded";
  auto fn = reinterpret_cast<GetVncMemoryStats>(dlsym(nrt, "nrt_get_vnc_memory_stats"));
  ASSERT_NE(fn, nullptr);
  VncMemoryStats stats{};
  size_t out = 0;
  ASSERT_EQ(fn(0, &stats, sizeof(stats), &out), 0);
  std::cout << "device 0: table " << before << ", after nrt_init " << after << ", libnrt limit "
            << stats.bytes_limit << ", used " << stats.bytes_used << std::endl;
  EXPECT_EQ(after, (nxs_long)stats.bytes_limit);
  EXPECT_EQ(before, after) << "the arch table disagrees with libnrt";
  dlclose(nrt);
}
