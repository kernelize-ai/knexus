#!/usr/bin/env python3
"""
Unit tests for the host properties the cpu plugin states: packages, cores,
threads and NUMA nodes from the hwloc topology (docs/Python_API.md, "Host and
link properties").

The plugin loads its topology once per process at first use, so each fixture
case drives a child interpreter with hwloc's own fixture variable set
(HWLOC_SYNTHETIC or HWLOC_XMLFILE) and reads its answer back as JSON.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

try:
    import knexus
except ImportError:
    print("Warning: knexus module not found. Tests will be skipped.")
    knexus = None

HOST_INTS = ("SocketCount", "CoreCount", "ThreadCount", "NumaNodeCount", "GlobalMemorySize")
HOST_VECS = ("ThreadSocket", "ThreadCore", "ThreadNumaNode", "NumaNodeMemorySize", "NumaDistance")
HOST_STR_VECS = ("NumaNodeCpuList",)

# Run in a child interpreter: the cpu device's host properties as JSON.
# The child loads the cpu plugin alone: hwloc's fixture variables are read by
# every hwloc user in the process, and tt-metal's UMD, finding no Tenstorrent
# PCI device in a fixture, fails its eager device open. knexus/__init__.py sets
# KNEXUS_RUNTIME_PATH at import, so the child points it at a directory holding
# only the cpu plugin before its first get_runtime().
READ = r"""
import json, os, sys
import knexus
os.environ["KNEXUS_RUNTIME_PATH"] = os.environ["KNEXUS_TEST_CPU_ONLY"]
dev = knexus.get_runtime("cpu").get_devices()[0]
out = {"Name": dev.get_property_str("Name")}
for name in %r:
    try:
        out[name] = dev.get_property_int(name)
    except RuntimeError:
        out[name] = None
for name in %r:
    try:
        out[name] = dev.get_property_int_vec(name)
    except RuntimeError:
        out[name] = None
for name in %r:
    try:
        out[name] = dev.get_property_str_vec(name)
    except RuntimeError:
        out[name] = None
out["Keys"] = [str(k).rsplit(".", 1)[-1] for k in dev.get_property_keys()]
print(json.dumps(out))
""" % (HOST_INTS, HOST_VECS, HOST_STR_VECS)


FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fixtures')
GIB = 1 << 30


def read_host(**hwloc_env):
    """The host properties a child interpreter states with the given hwloc
    fixture variables (HWLOC_SYNTHETIC or HWLOC_XMLFILE) set. hwloc reads them
    at the plugin's first topology load, once per process, hence the child."""
    # the child finds knexus where this file does and runs from the repo root
    # (tt-metal writes generated/ into the cwd); other plugins may log to
    # stdout around the one JSON line
    top = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    path = os.path.join(top, 'python')
    env = {k: v for k, v in os.environ.items() if not k.startswith("HWLOC_")}
    env.update(hwloc_env)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (path, os.environ.get("PYTHONPATH")) if p)
    plugin = os.path.join(os.path.dirname(os.path.abspath(knexus.__file__)),
                          'runtime_libs', 'libcpu_plugin.so')
    with tempfile.TemporaryDirectory() as only:
        os.symlink(plugin, os.path.join(only, 'libcpu_plugin.so'))
        env["KNEXUS_TEST_CPU_ONLY"] = only
        out = subprocess.run([sys.executable, "-c", READ], env=env, capture_output=True,
                             text=True, timeout=120, cwd=top)
    if out.returncode != 0:
        raise AssertionError(f"child interpreter failed: {out.stderr[-2000:]}")
    lines = [l for l in out.stdout.splitlines() if l.startswith('{"Name"')]
    if not lines:
        raise AssertionError(f"child interpreter printed no JSON: {out.stdout[-2000:]}")
    return json.loads(lines[-1])


@unittest.skipIf(knexus is None, "knexus module not available")
@unittest.skipUnless(sys.platform.startswith("linux"), "host properties are verified on Linux")
class TestHostFixture(unittest.TestCase):
    """The plugin over hwloc fixture topologies: every figure is the fixture's.

    A malformed fixture does not fail: hwloc warns and loads this host. Each
    case therefore pins figures (memory sizes, Name, counts) that this host
    does not state, so a fallback fails the case instead of passing it."""

    def test_synthetic_two_packages(self):
        # synthetic: 1 GiB per NUMA node, core os_index global, no CPUModel,
        # no distance matrix
        h = read_host(HWLOC_SYNTHETIC="pack:2 numa:1 core:4 pu:2")
        self.assertEqual(h["Name"], "cpu", "no CPUModel: the plugin's fallback name")
        self.assertEqual(h["GlobalMemorySize"], 2 * GIB)
        self.assertEqual((h["SocketCount"], h["CoreCount"], h["ThreadCount"], h["NumaNodeCount"]),
                         (2, 8, 16, 2))
        self.assertEqual(h["ThreadSocket"], [0] * 8 + [1] * 8)
        self.assertEqual(h["ThreadCore"], [c for c in range(8) for _ in range(2)])
        self.assertEqual(h["ThreadNumaNode"], [0] * 8 + [1] * 8)
        self.assertEqual(h["NumaNodeMemorySize"], [GIB] * 2)
        self.assertIsNone(h["NumaDistance"], "a synthetic topology carries no matrix")
        self.assertEqual(h["NumaNodeCpuList"], ["0-7", "8-15"])
        for name in HOST_INTS + HOST_VECS + HOST_STR_VECS:
            self.assertIn(name, h["Keys"], f"{name} is among the device's Keys")

    def test_synthetic_single_package_one_node(self):
        h = read_host(HWLOC_SYNTHETIC="pack:1 numa:1(memory=3221225472) core:4 pu:1")
        self.assertEqual(h["GlobalMemorySize"], 3 * GIB)
        self.assertEqual((h["SocketCount"], h["CoreCount"], h["ThreadCount"], h["NumaNodeCount"]),
                         (1, 4, 4, 1))
        self.assertEqual(h["NumaDistance"], [10], "one node states the kernel's [10]")
        self.assertEqual(h["ThreadNumaNode"], [0, 0, 0, 0])
        self.assertEqual(h["NumaNodeCpuList"], ["0-3"])

    def test_xml_two_nodes_with_distances(self):
        # test/fixtures/hwloc-two-node.xml: CPUModel, 16 + 8 GiB, a 10/21
        # FROM_OS matrix, core os_index local to each package
        h = read_host(HWLOC_XMLFILE=os.path.join(FIXTURES, "hwloc-two-node.xml"))
        self.assertEqual(h["Name"], "Fixture CPU 9000", "the package's CPUModel, trimmed")
        self.assertEqual(h["GlobalMemorySize"], 24 * GIB)
        self.assertEqual((h["SocketCount"], h["CoreCount"], h["ThreadCount"], h["NumaNodeCount"]),
                         (2, 8, 16, 2))
        self.assertEqual(h["ThreadSocket"], [0] * 8 + [1] * 8)
        self.assertEqual(h["ThreadCore"], [c for c in range(4) for _ in range(2)] * 2)
        self.assertEqual(h["ThreadNumaNode"], [0] * 8 + [1] * 8)
        self.assertEqual(h["NumaNodeMemorySize"], [16 * GIB, 8 * GIB])
        self.assertEqual(h["NumaDistance"], [10, 21, 21, 10])
        self.assertEqual(h["NumaNodeCpuList"], ["0-7", "8-15"])
        for name in HOST_INTS + HOST_VECS + HOST_STR_VECS:
            self.assertIn(name, h["Keys"], f"{name} is among the device's Keys")


@unittest.skipIf(knexus is None, "knexus module not available")
@unittest.skipUnless(sys.platform.startswith("linux"), "host properties are verified on Linux")
class TestHostLive(unittest.TestCase):
    """The plugin over this machine: the shape holds, whatever the figures."""

    def test_live_host(self):
        dev = knexus.get_runtime("cpu").get_devices()[0]
        sockets = dev.get_property_int("SocketCount")
        cores = dev.get_property_int("CoreCount")
        threads = dev.get_property_int("ThreadCount")
        nodes = dev.get_property_int("NumaNodeCount")
        self.assertGreaterEqual(sockets, 1)
        self.assertGreaterEqual(cores, sockets)
        self.assertGreaterEqual(threads, cores)
        self.assertGreaterEqual(nodes, 1)
        ts = dev.get_property_int_vec("ThreadSocket")
        tc = dev.get_property_int_vec("ThreadCore")
        tn = dev.get_property_int_vec("ThreadNumaNode")
        self.assertEqual(len(ts), len(tc))
        self.assertEqual(len(ts), len(tn))
        self.assertEqual(sum(1 for s in ts if s >= 0), threads)
        self.assertEqual(len(set(s for s in ts if s >= 0)), sockets)
        self.assertEqual(len(dev.get_property_int_vec("NumaNodeMemorySize")), nodes)
        self.assertEqual(len(dev.get_property_int_vec("NumaDistance")), nodes * nodes)
        lists = dev.get_property_str_vec("NumaNodeCpuList")
        self.assertEqual(len(lists), nodes)
        self.assertTrue(all(isinstance(l, str) and l for l in lists))
        self.assertGreater(dev.get_property_int("GlobalMemorySize"), 0)
        self.assertTrue(dev.get_property_str("Name"))


if __name__ == '__main__':
    unittest.main()
