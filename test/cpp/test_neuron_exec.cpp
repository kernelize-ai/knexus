// Execution through the neuron plugin (plugins/neuron/README.md "Execution"):
// load a NEFF, bind buffers, run blocking and non-blocking, and the failure
// paths. The NEFF must compute one float32 output as the elementwise sum of
// its float32 inputs, as the NKI kernel in the README does:
//
//   NEURON_RT_VISIBLE_CORES=0 ./test/cpp/test_neuron_exec kernel.neff
//
// (or KNEXUS_NEURON_NEFF=kernel.neff). Skipped where there is no neuron
// runtime, no device, or no NEFF.
#include <gtest/gtest.h>
#include <knexus.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

static std::string g_neff;

namespace {

struct Arg {
  std::string name;
  std::vector<nxs_long> shape;
  size_t count = 1;  // elements
};

class NeuronExec : public ::testing::Test {
 protected:
  knexus::Device dev;
  knexus::Library lib;
  knexus::Kernel kern;
  std::vector<Arg> args;  // inputs, then the output
  size_t num_inputs = 0;

  void SetUp() override {
    if (g_neff.empty()) GTEST_SKIP() << "no NEFF (argv[1] or KNEXUS_NEURON_NEFF)";
    auto rt = knexus::getSystem().getRuntime("neuron");
    if (!rt || rt.getDevices().empty()) GTEST_SKIP() << "no neuron runtime or device";
    dev = rt.getDevice(0);
    lib = dev.createLibrary(g_neff);
    ASSERT_TRUE(lib) << "cannot load " << g_neff << " (cores held by another process?)";
    kern = lib.getKernel("kernel");  // a NEFF has one kernel, whatever the name
    ASSERT_TRUE(kern);

    auto names = kern.getProp<std::vector<std::string>>(NP_SubUnits);
    auto types = kern.getProp<std::vector<std::string>>(NP_DataTypes);
    auto shapes = kern.getProp<std::vector<nxs_long>>(NP_Shape);  // rank, dims, rank, ...
    ASSERT_EQ(kern.getProp<nxs_long>(NP_Size), (nxs_long)names.size());
    size_t pos = 0;
    for (size_t i = 0; i < names.size(); ++i) {
      ASSERT_EQ(types[i], "float32") << names[i];
      Arg a;
      a.name = names[i];
      nxs_long rank = shapes.at(pos++);
      for (nxs_long d = 0; d < rank; ++d) {
        a.shape.push_back(shapes.at(pos++));
        a.count *= a.shape.back();
      }
      args.push_back(a);
    }
    ASSERT_GE(args.size(), 2u);
    num_inputs = args.size() - 1;  // the test kernel has one output, last
    std::cout << "kernel: " << kern.getProp<std::string>(NP_Description) << std::endl;
  }

  knexus::Buffer makeBuffer(const std::vector<float> &host) {
    return dev.createBuffer(host.size() * sizeof(float), host.data());
  }

  std::vector<float> input(size_t i) {
    std::vector<float> v(args[i].count);
    for (size_t j = 0; j < v.size(); ++j) v[j] = 0.25f * (float)(j % 1000) + (float)i;
    return v;
  }

  std::vector<float> sum(const std::vector<std::vector<float>> &ins) {
    std::vector<float> out(ins[0].size(), 0.f);
    for (auto &in : ins)
      for (size_t j = 0; j < out.size(); ++j) out[j] += in[j];
    return out;
  }

  static size_t mismatches(const std::vector<float> &got, const std::vector<float> &want) {
    size_t bad = 0;
    for (size_t j = 0; j < got.size(); ++j) bad += got[j] != want[j];
    return bad;
  }
};

}  // namespace

TEST_F(NeuronExec, BlockingByIndex) {
  std::vector<std::vector<float>> ins;
  std::vector<knexus::Buffer> bufs;
  for (size_t i = 0; i < num_inputs; ++i) {
    ins.push_back(input(i));
    bufs.push_back(makeBuffer(ins.back()));
  }
  std::vector<float> out(args.back().count, 0.f);
  auto bout = makeBuffer(out);

  auto sched = dev.createSchedule();
  auto cmd = sched.createCommand(kern);
  for (size_t i = 0; i < num_inputs; ++i) ASSERT_EQ(cmd.setArgument(i, bufs[i]), NXS_Success);
  ASSERT_EQ(cmd.setArgument(num_inputs, bout), NXS_Success);
  ASSERT_EQ(cmd.finalize({1, 1, 1}, {1, 1, 1}, 0), NXS_Success);
  ASSERT_EQ(sched.run(knexus::Stream(), NXS_ExecutionSettings_Timing), NXS_Success);
  std::cout << "blocking run: " << sched.getProp<nxs_double>(NP_ElapsedTime) << " ms" << std::endl;

  ASSERT_EQ(bout.copy(out.data()), NXS_Success);
  EXPECT_EQ(mismatches(out, sum(ins)), 0u);
}

// Two schedules on two streams of the device, ordered by events, both
// non-blocking, bound by tensor name: o1 = sum(ins), o2 = o1 + ins[1..].
TEST_F(NeuronExec, NonBlockingNamedWithEvents) {
  std::vector<std::vector<float>> ins;
  std::vector<knexus::Buffer> bufs;
  for (size_t i = 0; i < num_inputs; ++i) {
    ins.push_back(input(i));
    bufs.push_back(makeBuffer(ins.back()));
  }
  std::vector<float> zero(args.back().count, 0.f);
  auto o1 = makeBuffer(zero);
  auto o2 = makeBuffer(zero);
  auto s0 = dev.createStream();
  auto s1 = dev.createStream();
  auto ev0 = dev.createEvent();
  auto evFinal = dev.createEvent();

  auto sched0 = dev.createSchedule();
  auto cmd0 = sched0.createCommand(kern);
  // Named, in reverse order: the index is ignored where a name is given.
  ASSERT_EQ(cmd0.setArgument(0, o1, args.back().name.c_str()), NXS_Success);
  for (size_t i = 0; i < num_inputs; ++i)
    ASSERT_EQ(cmd0.setArgument(i + 1, bufs[i], args[i].name.c_str()), NXS_Success);
  ASSERT_EQ(cmd0.finalize({1, 1, 1}, {1, 1, 1}, 0), NXS_Success);
  sched0.createSignalCommand(ev0);

  auto sched1 = dev.createSchedule();
  sched1.createWaitCommand(ev0);
  auto cmd1 = sched1.createCommand(kern);
  ASSERT_EQ(cmd1.setArgument(0, o1), NXS_Success);
  for (size_t i = 1; i < num_inputs; ++i) ASSERT_EQ(cmd1.setArgument(i, bufs[i]), NXS_Success);
  ASSERT_EQ(cmd1.setArgument(num_inputs, o2), NXS_Success);
  ASSERT_EQ(cmd1.finalize({1, 1, 1}, {1, 1, 1}, 0), NXS_Success);
  sched1.createSignalCommand(evFinal);

  ASSERT_EQ(sched0.run(s0, NXS_ExecutionSettings_NonBlocking), NXS_Success);
  ASSERT_EQ(sched1.run(s1, NXS_ExecutionSettings_NonBlocking), NXS_Success);
  ASSERT_EQ(evFinal.wait(1), NXS_Success);
  EXPECT_EQ(evFinal.getProp<nxs_long>(NP_Value), 1);

  std::vector<float> got(zero.size());
  ASSERT_EQ(o2.copy(got.data()), NXS_Success);
  auto want = sum(ins);
  for (size_t i = 1; i < num_inputs; ++i)
    for (size_t j = 0; j < want.size(); ++j) want[j] += ins[i][j];
  EXPECT_EQ(mismatches(got, want), 0u);
}

// More non-blocking runs than the core's queue holds: a full queue is
// drained and the submit retried.
TEST_F(NeuronExec, NonBlockingQueueFull) {
  std::vector<std::vector<float>> ins;
  std::vector<knexus::Buffer> bufs;
  for (size_t i = 0; i < num_inputs; ++i) {
    ins.push_back(input(i));
    bufs.push_back(makeBuffer(ins.back()));
  }
  std::vector<float> out(args.back().count, 0.f);
  auto bout = makeBuffer(out);
  auto sched = dev.createSchedule();
  auto cmd = sched.createCommand(kern);
  for (size_t i = 0; i < num_inputs; ++i) cmd.setArgument(i, bufs[i]);
  cmd.setArgument(num_inputs, bout);
  ASSERT_EQ(cmd.finalize({1, 1, 1}, {1, 1, 1}, 0), NXS_Success);
  for (int r = 0; r < 500; ++r)
    ASSERT_EQ(sched.run(knexus::Stream(), NXS_ExecutionSettings_NonBlocking), NXS_Success) << r;
  ASSERT_EQ(bout.copy(out.data()), NXS_Success);  // waits for the queue
  EXPECT_EQ(mismatches(out, sum(ins)), 0u);
}

TEST_F(NeuronExec, BadBindings) {
  auto a = makeBuffer(input(0));
  auto sched = dev.createSchedule();
  auto cmd = sched.createCommand(kern);
  ASSERT_EQ(cmd.setArgument(0, a), NXS_Success);
  // The other tensors are unbound: refused before libnrt's bare NRT_INVALID.
  EXPECT_EQ(cmd.finalize({1, 1, 1}, {1, 1, 1}, 0), NXS_InvalidKernelArgs);
  EXPECT_EQ(sched.run(knexus::Stream(), 0), NXS_InvalidKernelArgs);

  std::vector<float> small(16, 0.f);
  EXPECT_EQ(cmd.setArgument(1, makeBuffer(small)), NXS_InvalidArgSize);
  EXPECT_EQ(cmd.setArgument(1, a, "no_such_tensor"), NXS_InvalidArgValue);
  EXPECT_EQ(cmd.setArgument((nxs_uint)args.size(), a), NXS_InvalidArgIndex);
  EXPECT_EQ(cmd.setArgument(1, (nxs_int)3), NXS_InvalidArgValue);  // scalars unsupported
}

// A released buffer is unbound from the commands using it: the next run is
// refused, not executed against a freed tensor.
TEST_F(NeuronExec, ReleasedBufferUnbinds) {
  std::vector<knexus::Buffer> bufs;
  for (size_t i = 0; i < num_inputs; ++i) bufs.push_back(makeBuffer(input(i)));
  std::vector<float> out(args.back().count, 0.f);
  auto bout = makeBuffer(out);
  auto sched = dev.createSchedule();
  auto cmd = sched.createCommand(kern);
  for (size_t i = 0; i < num_inputs; ++i) cmd.setArgument(i, bufs[i]);
  cmd.setArgument(num_inputs, bout);
  ASSERT_EQ(cmd.finalize({1, 1, 1}, {1, 1, 1}, 0), NXS_Success);
  ASSERT_EQ(sched.run(knexus::Stream(), NXS_ExecutionSettings_NonBlocking), NXS_Success);
  bufs[0].release();
  EXPECT_EQ(sched.run(knexus::Stream(), 0), NXS_InvalidKernelArgs);
}

// A library from the NEFF's bytes: libnrt keeps its own copy, so the host
// bytes may go at once.
TEST_F(NeuronExec, LibraryFromBytes) {
  std::ifstream in(g_neff, std::ios::binary);
  auto *bytes = new std::vector<char>((std::istreambuf_iterator<char>(in)),
                                      std::istreambuf_iterator<char>());
  auto blib = dev.createLibrary(bytes->data(), bytes->size());
  size_t size = bytes->size();
  delete bytes;
  ASSERT_TRUE(blib);
  EXPECT_EQ(blib.getProp<nxs_long>(NP_Size), (nxs_long)size);
  EXPECT_EQ(blib.getProp<std::string>(NP_Type), "neff");
  auto bkern = blib.getKernel("any_name");
  ASSERT_TRUE(bkern);

  std::vector<std::vector<float>> ins;
  std::vector<knexus::Buffer> bufs;
  for (size_t i = 0; i < num_inputs; ++i) {
    ins.push_back(input(i));
    bufs.push_back(makeBuffer(ins.back()));
  }
  std::vector<float> out(args.back().count, 0.f);
  auto bout = makeBuffer(out);
  auto sched = dev.createSchedule();
  auto cmd = sched.createCommand(bkern);
  for (size_t i = 0; i < num_inputs; ++i) cmd.setArgument(i, bufs[i]);
  cmd.setArgument(num_inputs, bout);
  ASSERT_EQ(sched.run(knexus::Stream(), 0), NXS_Success);  // not finalized: checked at run
  ASSERT_EQ(bout.copy(out.data()), NXS_Success);
  EXPECT_EQ(mismatches(out, sum(ins)), 0u);

  std::vector<char> junk(64, 'x');
  EXPECT_FALSE(dev.createLibrary(junk.data(), junk.size()));
}

// A released event is detached from the signal and wait commands naming it:
// running them fails, and the event a later createEvent places in the same
// slot is neither signalled nor waited on.
TEST_F(NeuronExec, ReleasedEventDetaches) {
  auto ev = dev.createEvent();
  auto sched = dev.createSchedule();
  ASSERT_TRUE(sched.createSignalCommand(ev, 5));
  auto wsched = dev.createSchedule();
  ASSERT_TRUE(wsched.createWaitCommand(ev, 7));
  ev.release();
  auto ev2 = dev.createEvent();  // likely the released slot
  ASSERT_TRUE(ev2);
  EXPECT_EQ(sched.run(knexus::Stream(), 0), NXS_InvalidEvent);
  EXPECT_EQ(ev2.getProp<nxs_long>(NP_Value), 0);
  EXPECT_EQ(wsched.run(knexus::Stream(), 0), NXS_InvalidEvent);  // does not block on ev2
}

// A released library takes its kernel ids with it: a kernel id from before
// the release resolves to no kernel, not to the kernel of a library loaded
// into the same slot later.
TEST_F(NeuronExec, ReleasedLibraryDropsKernelIds) {
  auto lib1 = dev.createLibrary(g_neff);
  ASSERT_TRUE(lib1);
  auto k1 = lib1.getKernel("kernel");
  ASSERT_TRUE(k1);
  lib1.release();
  auto lib2 = dev.createLibrary(g_neff);  // likely the released slot
  ASSERT_TRUE(lib2);
  auto sched = dev.createSchedule();
  EXPECT_FALSE(sched.createCommand(k1));
  // lib2's own kernel is good.
  auto k2 = lib2.getKernel("kernel");
  ASSERT_TRUE(k2);
  EXPECT_TRUE(sched.createCommand(k2));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc > 1) g_neff = argv[1];
  else if (const char *env = std::getenv("KNEXUS_NEURON_NEFF")) g_neff = env;
  return RUN_ALL_TESTS();
}
