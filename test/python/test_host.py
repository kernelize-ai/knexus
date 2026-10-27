#!/usr/bin/env python3
"""
Unit tests for the host properties the cpu plugin states — sockets, cores,
threads and NUMA nodes read from sysfs and procfs under $KNEXUS_HOST_ROOT
(docs/Python_API.md, "Host and link properties").

The reader runs once per process at first use, so the fixture case drives a
child interpreter with the variable set and reads its answer back as JSON.
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
READ = r"""
import json, sys
import knexus
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


def put(path, text):
    with open(path, "w") as f:
        f.write(text)


def write_fixture(root, sockets=2, cores=4, threads=2, nodes=2):
    """A fixture kernel tree: sockets x cores x threads cpus, one NUMA node
    per socket, a distance table, cpuinfo and meminfo."""
    cpu = os.path.join(root, "sys/devices/system/cpu")
    node = os.path.join(root, "sys/devices/system/node")
    n = 0
    per_node = {}
    for s in range(sockets):
        for c in range(cores):
            for t in range(threads):
                d = os.path.join(cpu, f"cpu{n}", "topology")
                os.makedirs(d)
                put(os.path.join(d, "physical_package_id"), f"{s}\n")
                put(os.path.join(d, "core_id"), f"{c}\n")
                per_node.setdefault(s % nodes, []).append(n)
                n += 1
    for i in range(nodes):
        d = os.path.join(node, f"node{i}")
        os.makedirs(d)
        cpus = per_node.get(i, [])
        put(os.path.join(d, "cpulist"), f"{cpus[0]}-{cpus[-1]}\n" if cpus else "\n")
        put(os.path.join(d, "meminfo"), 
            f"Node {i} MemTotal:       16000000 kB\nNode {i} MemFree:         1000000 kB\n")
        put(os.path.join(d, "distance"), 
            " ".join("10" if j == i else "21" for j in range(nodes)) + "\n")
    os.makedirs(os.path.join(root, "proc"))
    put(os.path.join(root, "proc/cpuinfo"), 
        "processor\t: 0\nmodel name\t: Fixture CPU 9000\nprocessor\t: 1\n")
    put(os.path.join(root, "proc/meminfo"), 
        "MemTotal:       32000000 kB\nMemFree:         1000000 kB\n")


def read_host(root):
    env = dict(os.environ, KNEXUS_HOST_ROOT=root)
    out = subprocess.run([sys.executable, "-c", READ], env=env, capture_output=True,
                         text=True, timeout=120, cwd=os.path.dirname(__file__))
    if out.returncode != 0:
        raise AssertionError(f"child interpreter failed: {out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


@unittest.skipIf(knexus is None, "knexus module not available")
@unittest.skipUnless(sys.platform.startswith("linux"), "the host reader is Linux sysfs")
class TestHostFixture(unittest.TestCase):
    """The reader over a fixture tree: every figure is the tree's."""

    def test_fixture_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp)
            h = read_host(tmp)
        self.assertEqual(h["Name"], "Fixture CPU 9000")
        self.assertEqual(h["GlobalMemorySize"], 32000000 * 1024)
        self.assertEqual((h["SocketCount"], h["CoreCount"], h["ThreadCount"], h["NumaNodeCount"]),
                         (2, 8, 16, 2))
        self.assertEqual(h["ThreadSocket"], [0] * 8 + [1] * 8)
        self.assertEqual(h["ThreadCore"], [c for c in range(4) for _ in range(2)] * 2)
        self.assertEqual(h["ThreadNumaNode"], [0] * 8 + [1] * 8)
        self.assertEqual(h["NumaNodeMemorySize"], [16000000 * 1024] * 2)
        self.assertEqual(h["NumaDistance"], [10, 21, 21, 10])
        self.assertEqual(h["NumaNodeCpuList"], ["0-7", "8-15"])
        for name in HOST_INTS + HOST_VECS + HOST_STR_VECS:
            self.assertIn(name, h["Keys"], f"{name} is among the device's Keys")

    def test_single_socket_no_numa(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp, sockets=1, cores=4, threads=1, nodes=1)
            h = read_host(tmp)
        self.assertEqual((h["SocketCount"], h["CoreCount"], h["ThreadCount"], h["NumaNodeCount"]),
                         (1, 4, 4, 1))
        self.assertEqual(h["NumaDistance"], [10])
        self.assertEqual(h["ThreadNumaNode"], [0, 0, 0, 0])
        self.assertEqual(h["NumaNodeCpuList"], ["0-3"])

    def test_empty_root_states_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = read_host(tmp)
        self.assertEqual(h["Name"], "cpu")
        for name in HOST_INTS + HOST_VECS + HOST_STR_VECS:
            self.assertIsNone(h[name], f"{name} is absent when the root holds no cpu topology")
        self.assertNotIn("SocketCount", h["Keys"])


@unittest.skipIf(knexus is None, "knexus module not available")
@unittest.skipUnless(sys.platform.startswith("linux"), "the host reader is Linux sysfs")
class TestHostLive(unittest.TestCase):
    """The reader over this machine: the shape holds, whatever the figures."""

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
