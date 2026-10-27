#!/usr/bin/env python3
"""
Unit tests for the link properties the cuda plugin states from NVML — one
row per link, every Link* property a vector over the rows: the device's PCIe
link to the host, an NVLink row per active link, and a PCIe path row per
peer (docs/Python_API.md, "Host and link properties"). Skips without a cuda
runtime holding a device.
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

BUS_ID = re.compile(r"^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]$")
LEVELS = {0, 10, 20, 30, 40, 50}
TYPES = {"PCIe", "NVLink"}
INT_COLUMNS = ("LinkPeer", "LinkGenMax", "LinkGenCurrent", "LinkWidthMax", "LinkWidthCurrent",
               "LinkTopologyLevel")
STR_COLUMNS = ("LinkType", "LinkPeerId")
LINK_NAMES = ("PciBusId", "LinkCount") + STR_COLUMNS + INT_COLUMNS


def rows(dev):
    """The device's links as a list of dicts, one per row."""
    n = dev.get_property_int("LinkCount")
    cols = {c: dev.get_property_str_vec(c) for c in STR_COLUMNS}
    cols.update({c: dev.get_property_int_vec(c) for c in INT_COLUMNS})
    for c, v in cols.items():
        assert len(v) == n, f"{c} has {len(v)} entries for LinkCount {n}"
    return [{c: cols[c][i] for c in cols} for i in range(n)]


def cuda_devices():
    if knexus is None:
        return []
    rt = knexus.get_runtime("cuda")
    if not rt:
        return []
    return list(rt.get_devices())


@unittest.skipIf(not cuda_devices(), "no cuda runtime with a device")
class TestLinks(unittest.TestCase):

    def setUp(self):
        self.devices = cuda_devices()

    def test_host_row(self):
        for dev in self.devices:
            self.assertRegex(dev.get_property_str("PciBusId"), BUS_ID)
            host = rows(dev)[0]
            self.assertEqual(host["LinkType"], "PCIe")
            self.assertEqual(host["LinkPeer"], -1)
            self.assertGreaterEqual(host["LinkGenMax"], host["LinkGenCurrent"])
            self.assertGreaterEqual(host["LinkGenCurrent"], 1)
            self.assertGreaterEqual(host["LinkWidthMax"], host["LinkWidthCurrent"])
            self.assertGreaterEqual(host["LinkWidthCurrent"], 1)
            self.assertEqual(host["LinkTopologyLevel"], -1)

    def test_rows(self):
        n = len(self.devices)
        table = [rows(d) for d in self.devices]
        for i in range(n):
            for r in table[i]:
                self.assertIn(r["LinkType"], TYPES)
                self.assertNotEqual(r["LinkPeer"], i, "a device states no link to itself")
                self.assertIn(r["LinkPeer"], range(-1, n))
            for j in range(n):
                if j == i:
                    continue
                paths = [r for r in table[i] if r["LinkType"] == "PCIe" and r["LinkPeer"] == j]
                self.assertEqual(len(paths), 1, "one PCIe path row per peer")
                self.assertIn(paths[0]["LinkTopologyLevel"], LEVELS)
                back = [r for r in table[j] if r["LinkType"] == "PCIe" and r["LinkPeer"] == i]
                self.assertEqual(paths[0]["LinkTopologyLevel"], back[0]["LinkTopologyLevel"])
                nv = [r for r in table[i] if r["LinkType"] == "NVLink" and r["LinkPeer"] == j]
                nv_back = [r for r in table[j] if r["LinkType"] == "NVLink" and r["LinkPeer"] == i]
                self.assertEqual(len(nv), len(nv_back), "NVLinks are symmetric")
                for r in nv:
                    self.assertRegex(r["LinkPeerId"], BUS_ID)

    def test_keys(self):
        keys = [str(k).rsplit(".", 1)[-1] for k in self.devices[0].get_property_keys()]
        for name in LINK_NAMES:
            self.assertIn(name, keys)

    def test_bus_ids_distinct(self):
        ids = [d.get_property_str("PciBusId") for d in self.devices]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == '__main__':
    unittest.main()
