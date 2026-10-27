#!/usr/bin/env python3
"""
Unit tests for the link properties the cuda plugin states from NVML — each
device's PCIe link to the host and, per device pair, its NVLink count or
PCIe common-ancestor level (docs/Python_API.md, "Host and link properties").
Skips without a cuda runtime holding a device.
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
LINK_NAMES = ("PciBusId", "PcieLinkGenMax", "PcieLinkGenCurrent", "PcieLinkWidthMax",
              "PcieLinkWidthCurrent", "NvLinkCount", "PcieTopologyLevel")


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

    def test_pcie_link(self):
        for dev in self.devices:
            self.assertRegex(dev.get_property_str("PciBusId"), BUS_ID)
            gen_max = dev.get_property_int("PcieLinkGenMax")
            gen_cur = dev.get_property_int("PcieLinkGenCurrent")
            w_max = dev.get_property_int("PcieLinkWidthMax")
            w_cur = dev.get_property_int("PcieLinkWidthCurrent")
            self.assertGreaterEqual(gen_max, gen_cur)
            self.assertGreaterEqual(gen_cur, 1)
            self.assertGreaterEqual(w_max, w_cur)
            self.assertGreaterEqual(w_cur, 1)

    def test_pairs(self):
        n = len(self.devices)
        nvlinks = [d.get_property_int_vec("NvLinkCount") for d in self.devices]
        levels = [d.get_property_int_vec("PcieTopologyLevel") for d in self.devices]
        for i in range(n):
            self.assertEqual(len(nvlinks[i]), n)
            self.assertEqual(len(levels[i]), n)
            self.assertEqual(nvlinks[i][i], 0)
            self.assertEqual(levels[i][i], -1)
            for j in range(n):
                if j == i:
                    continue
                self.assertGreaterEqual(nvlinks[i][j], 0)
                self.assertEqual(nvlinks[i][j], nvlinks[j][i])
                self.assertIn(levels[i][j], LEVELS)
                self.assertEqual(levels[i][j], levels[j][i])

    def test_keys(self):
        keys = [str(k).rsplit(".", 1)[-1] for k in self.devices[0].get_property_keys()]
        for name in LINK_NAMES:
            self.assertIn(name, keys)

    def test_bus_ids_distinct(self):
        ids = [d.get_property_str("PciBusId") for d in self.devices]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == '__main__':
    unittest.main()
