#!/usr/bin/env python3
"""
Unit tests for the link properties a runtime states on its devices — one row
per link, every Link* property a vector over the rows: the device's PCIe link
to the host, a row per fabric link to a peer (NVLink on cuda, Ethernet on
tenstorrent), and a PCIe path row per PCIe-attached peer (docs/Python_API.md,
"Host and link properties"). The runtime-agnostic checks run on every runtime
whose devices state LinkCount, the NVML-specific ones on cuda only. Skips
only when no runtime states LinkCount.

The sysfs PCI reader behind the tenstorrent rows is checked over fixture
trees by the device-free gtest test/cpp/test_pci.cpp.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

try:
    import knexus
except ImportError:
    print("Warning: knexus module not found. Tests will be skipped.")
    knexus = None

# domain:bus:device.function; NVML writes an 8-hex domain, sysfs a 4-hex one
BUS_ID = re.compile(r"^[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]$")
LEVELS = {0, 10, 20, 30, 40, 50}
TYPES = {"PCIe", "NVLink", "Ethernet"}
INT_COLUMNS = ("LinkPeer", "LinkGenMax", "LinkGenCurrent", "LinkWidthMax", "LinkWidthCurrent",
               "LinkTopologyLevel")
STR_COLUMNS = ("LinkType", "LinkPeerId")
LINK_NAMES = ("LinkCount",) + STR_COLUMNS + INT_COLUMNS


def rows(dev):
    """The device's links as a list of dicts, one per row."""
    n = dev.get_property_int("LinkCount")
    cols = {c: dev.get_property_str_vec(c) for c in STR_COLUMNS}
    cols.update({c: dev.get_property_int_vec(c) for c in INT_COLUMNS})
    for c, v in cols.items():
        assert len(v) == n, f"{c} has {len(v)} entries for LinkCount {n}"
    return [{c: cols[c][i] for c in cols} for i in range(n)]


def bus_id(dev):
    """The device's PciBusId, or None where it states none."""
    try:
        return dev.get_property_str("PciBusId")
    except RuntimeError:
        return None


def states_links(dev):
    try:
        dev.get_property_int("LinkCount")
        return True
    except RuntimeError:
        return False


def link_runtimes():
    """(name, devices) for every runtime with a device that states LinkCount."""
    if knexus is None:
        return []
    found = []
    for rt in knexus.get_runtimes():
        devices = list(rt.get_devices())
        if any(states_links(d) for d in devices):
            found.append((rt.get_property_str("Name"), devices))
    return found


RUNTIMES = link_runtimes()


@unittest.skipIf(not RUNTIMES, "no runtime states LinkCount on a device")
class TestLinks(unittest.TestCase):

    def test_host_row(self):
        """A device with a bus id states its PCIe host row first; one without states none."""
        for name, devices in RUNTIMES:
            for i, dev in enumerate(devices):
                with self.subTest(runtime=name, device=i):
                    table = rows(dev)
                    hosts = [r for r in table if r["LinkType"] == "PCIe" and r["LinkPeer"] == -1]
                    bus = bus_id(dev)
                    if name == "cuda":
                        self.assertIsNotNone(bus, "NVML states every device's bus id")
                    if bus is None:
                        self.assertEqual(hosts, [], "no PciBusId, no host row")
                        continue
                    self.assertRegex(bus, BUS_ID)
                    self.assertEqual(len(hosts), 1)
                    host = table[0]
                    self.assertEqual(host["LinkType"], "PCIe")
                    self.assertEqual(host["LinkPeer"], -1)
                    self.assertGreaterEqual(host["LinkGenMax"], host["LinkGenCurrent"])
                    self.assertGreaterEqual(host["LinkGenCurrent"], 1)
                    self.assertGreaterEqual(host["LinkWidthMax"], host["LinkWidthCurrent"])
                    self.assertGreaterEqual(host["LinkWidthCurrent"], 1)
                    self.assertEqual(host["LinkTopologyLevel"], -1)

    def test_rows(self):
        """Kinds, peers, one PCIe path row per PCIe peer, symmetric both ways."""
        for name, devices in RUNTIMES:
            n = len(devices)
            table = [rows(d) for d in devices]
            buses = [bus_id(d) for d in devices]
            for i in range(n):
                with self.subTest(runtime=name, device=i):
                    for r in table[i]:
                        self.assertIn(r["LinkType"], TYPES)
                        self.assertNotEqual(r["LinkPeer"], i, "a device states no link to itself")
                        self.assertIn(r["LinkPeer"], range(-1, n))
                        if name == "cuda" and r["LinkType"] == "NVLink":
                            self.assertRegex(r["LinkPeerId"], BUS_ID)
                for j in range(n):
                    if j == i:
                        continue
                    with self.subTest(runtime=name, pair=(i, j)):
                        paths = [r for r in table[i] if r["LinkType"] == "PCIe" and r["LinkPeer"] == j]
                        back = [r for r in table[j] if r["LinkType"] == "PCIe" and r["LinkPeer"] == i]
                        if buses[i] is None or buses[j] is None:
                            self.assertEqual(paths, [], "no path row without both bus ids")
                        else:
                            self.assertEqual(len(paths), 1, "one PCIe path row per peer")
                            self.assertEqual(paths[0]["LinkPeerId"], buses[j])
                            self.assertIn(paths[0]["LinkTopologyLevel"], LEVELS)
                            self.assertEqual(paths[0]["LinkTopologyLevel"], back[0]["LinkTopologyLevel"])
                        for kind in TYPES - {"PCIe"}:
                            there = [r for r in table[i] if r["LinkType"] == kind and r["LinkPeer"] == j]
                            again = [r for r in table[j] if r["LinkType"] == kind and r["LinkPeer"] == i]
                            self.assertEqual(len(there), len(again), f"{kind} rows are symmetric")
                            for r in there:
                                self.assertEqual(r["LinkPeerId"], buses[j] or "")

    def test_keys(self):
        for name, devices in RUNTIMES:
            for i, dev in enumerate(devices):
                with self.subTest(runtime=name, device=i):
                    keys = [str(k).rsplit(".", 1)[-1] for k in dev.get_property_keys()]
                    for key in LINK_NAMES:
                        self.assertIn(key, keys)
                    self.assertEqual("PciBusId" in keys, bus_id(dev) is not None)

    def test_bus_ids_distinct(self):
        for name, devices in RUNTIMES:
            with self.subTest(runtime=name):
                ids = [b for b in (bus_id(d) for d in devices) if b is not None]
                self.assertEqual(len(ids), len(set(ids)))


if __name__ == '__main__':
    unittest.main()
