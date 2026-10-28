#!/usr/bin/env python3
"""
Unit tests for the links the System states: knexus.get_links() returns every
link of the machine once, as Link objects whose two endpoints (get_devices())
are Devices that may belong to different runtimes (docs/Python_API.md, "Host
and link properties"). Three kinds: a device's PCIe link to the host, whose
second end is the cpu device; the PCIe path between two PCI-attached devices,
one per pair; and a fabric link a plugin states (NVLink on cuda, Ethernet on
tenstorrent), each pair once. No Device lists a Link* key. Runtime-agnostic;
the NVML-specific checks run on cuda only. Skips only when no link exists.

The core's PCIe and fabric builders are checked over fixtures by the
device-free gtest test/cpp/test_links.cpp, the sysfs reader by test_pci.cpp.
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
INT_COLUMNS = ("LinkGenMax", "LinkGenCurrent", "LinkWidthMax", "LinkWidthCurrent",
               "LinkTopologyLevel")
LINK_KEYS = ("Type",) + INT_COLUMNS
# the per-device row names a Device no longer lists
DEVICE_LINK_NAMES = ("LinkCount", "LinkType", "LinkPeerId", "LinkPeer", "LinkDevice") + INT_COLUMNS


def key_names(obj):
    return [str(k).rsplit(".", 1)[-1] for k in obj.get_property_keys()]


def bus_id(dev):
    """The device's PciBusId, or None where it states none."""
    try:
        return dev.get_property_str("PciBusId")
    except RuntimeError:
        return None


def owners():
    """(runtime name, device index) for each entry of knexus.get_devices()."""
    out = []
    for rt in knexus.get_runtimes():
        name = rt.get_property_str("Name")
        out.extend((name, i) for i in range(len(rt.get_devices())))
    return out


def facts(link):
    """The link's properties and endpoints as a dict."""
    out = {"Type": link.get_property_str("Type")}
    out.update({c: link.get_property_int(c) for c in INT_COLUMNS})
    out["ends"] = list(link.get_devices())
    return out


if knexus is not None:
    DEVICES = list(knexus.get_devices())
    OWNERS = owners()
    LINKS = [facts(l) for l in knexus.get_links()]
    BUSES = [bus_id(d) for d in DEVICES]
    CPU = [i for i, (rt, _) in enumerate(OWNERS) if rt == "cpu"]
else:
    DEVICES, OWNERS, LINKS, BUSES, CPU = [], [], [], [], []


def index_of(dev):
    """The endpoint's index in knexus.get_devices(): by PciBusId, else by Name among the
    devices with no bus id (the cpu device); None for an empty or unmatched Device."""
    if not dev:
        return None
    bus = bus_id(dev)
    if bus is not None:
        hits = [i for i, b in enumerate(BUSES) if b == bus]
    else:
        name = dev.get_property_str("Name")
        hits = [i for i, b in enumerate(BUSES)
                if b is None and DEVICES[i].get_property_str("Name") == name]
    return hits[0] if len(hits) == 1 else None


def is_host(link):
    """A PCIe link whose second end states no PciBusId: the cpu device, or empty."""
    b = ends(link)[1]
    return link["Type"] == "PCIe" and (b is None or BUSES[b] is None)


def ends(link):
    return tuple(index_of(d) for d in link["ends"])


@unittest.skipIf(not LINKS, "the System states no link")
class TestLinks(unittest.TestCase):

    def test_system_devices(self):
        """get_devices() is every runtime's devices, runtime order then device order."""
        self.assertEqual(len(DEVICES), len(OWNERS))
        for rt in knexus.get_runtimes():
            name = rt.get_property_str("Name")
            mine = [i for i, (r, _) in enumerate(OWNERS) if r == name]
            for i, dev in zip(mine, rt.get_devices()):
                with self.subTest(runtime=name, device=OWNERS[i][1]):
                    self.assertEqual(bus_id(DEVICES[i]), bus_id(dev))
                    self.assertEqual(DEVICES[i].get_property_str("Name"), dev.get_property_str("Name"))

    def test_endpoints(self):
        """Two valid, distinct Device endpoints; a PCI end's PciBusId is a bus id."""
        for k, link in enumerate(LINKS):
            with self.subTest(link=k, type=link["Type"]):
                self.assertIn(link["Type"], TYPES)
                self.assertEqual(len(link["ends"]), 2)
                a, b = ends(link)
                self.assertIsNotNone(a, "first end is a System device")
                if CPU or link["Type"] != "PCIe":
                    self.assertIsNotNone(b, "second end is a System device")
                self.assertNotEqual(a, b, "a link joins two devices")
                for e in (a, b):
                    if e is not None and BUSES[e] is not None:
                        self.assertRegex(BUSES[e], BUS_ID)

    def test_host_links(self):
        """One PCIe host link per PciBusId device, its second end the cpu device."""
        hosts = [l for l in LINKS if is_host(l)]
        by_device = {}
        for l in hosts:
            by_device.setdefault(ends(l)[0], []).append(l)
        for i, bus in enumerate(BUSES):
            with self.subTest(device=OWNERS[i]):
                if OWNERS[i][0] == "cuda":
                    self.assertIsNotNone(bus, "NVML states every device's bus id")
                if bus is None:
                    self.assertNotIn(i, by_device, "no PciBusId, no host link")
                    continue
                self.assertEqual(len(by_device.get(i, [])), 1, "one host link per PciBusId device")
                host = by_device[i][0]
                if CPU:
                    self.assertIn(ends(host)[1], CPU, "the host end is the cpu device")
                self.assertGreaterEqual(host["LinkGenMax"], host["LinkGenCurrent"])
                self.assertGreaterEqual(host["LinkGenCurrent"], 1)
                self.assertGreaterEqual(host["LinkWidthMax"], host["LinkWidthCurrent"])
                self.assertGreaterEqual(host["LinkWidthCurrent"], 1)
                self.assertEqual(host["LinkTopologyLevel"], -1)

    def test_pcie_paths(self):
        """One PCIe path per pair of PciBusId devices, across runtimes, never repeated."""
        paths = {}
        for l in LINKS:
            a, b = ends(l)
            if l["Type"] != "PCIe" or is_host(l):
                continue
            pair = frozenset((a, b))
            self.assertNotIn(pair, paths, "PCIe never repeats an unordered pair")
            paths[pair] = l
        pci = [i for i, b in enumerate(BUSES) if b is not None]
        for x in pci:
            for y in pci:
                if y <= x:
                    continue
                with self.subTest(pair=(OWNERS[x], OWNERS[y])):
                    self.assertIn(frozenset((x, y)), paths, "one PCIe path per PCIe pair")
                    self.assertIn(paths[frozenset((x, y))]["LinkTopologyLevel"], LEVELS)
        self.assertEqual(len(paths), len(pci) * (len(pci) - 1) // 2)

    def test_fabric_once(self):
        """A fabric pair may repeat (one link per lane) but never in both directions."""
        seen = set()
        for k, l in enumerate(LINKS):
            if l["Type"] == "PCIe":
                continue
            a, b = ends(l)
            with self.subTest(link=k, type=l["Type"]):
                self.assertIsNotNone(a)
                self.assertIsNotNone(b)
                self.assertNotIn((b, a), seen, f"{l['Type']} pair stated both ways")
                seen.add((a, b))
                self.assertEqual(l["LinkTopologyLevel"], -1)

    def test_fabric_table(self):
        """Each runtime's fabric table is its fabric Links: per row, the near end is
        device LinkDevice and the far end's PciBusId is LinkPeerId."""
        for rt in knexus.get_runtimes():
            name = rt.get_property_str("Name")
            if "LinkCount" not in key_names(rt):
                continue
            base = [i for i, (r, _) in enumerate(OWNERS) if r == name][0]
            with self.subTest(runtime=name):
                n = rt.get_property_int("LinkCount")
                kinds = rt.get_property_str_vec("LinkType")
                near = rt.get_property_int_vec("LinkDevice")
                peer = rt.get_property_int_vec("LinkPeer")
                peer_id = rt.get_property_str_vec("LinkPeerId")
                for col in (kinds, near, peer, peer_id):
                    self.assertEqual(len(col), n)
                rows = []
                for t, d, p, pid in zip(kinds, near, peer, peer_id):
                    self.assertIn(t, TYPES - {"PCIe"}, "a plugin states no PCIe link")
                    if p >= 0:
                        self.assertEqual(pid, BUSES[base + p] or "")
                    if pid:
                        self.assertRegex(pid, BUS_ID)
                    rows.append((t, BUSES[base + d], pid))
                bus = lambda e: (BUSES[e] or "") if e is not None else ""
                stated = [(l["Type"], bus(ends(l)[0]), bus(ends(l)[1])) for l in LINKS
                          if l["Type"] != "PCIe" and ends(l)[0] is not None
                          and OWNERS[ends(l)[0]][0] == name]
                self.assertEqual(sorted(rows), sorted(stated))

    def test_keys(self):
        """A Link lists exactly its properties; no Device lists a Link* key."""
        for k, link in enumerate(knexus.get_links()):
            with self.subTest(link=k):
                self.assertEqual(key_names(link), list(LINK_KEYS))
        for i, dev in enumerate(DEVICES):
            with self.subTest(device=OWNERS[i]):
                keys = key_names(dev)
                for key in DEVICE_LINK_NAMES:
                    self.assertNotIn(key, keys)
                self.assertEqual("PciBusId" in keys, BUSES[i] is not None)

    def test_bus_ids_distinct(self):
        ids = [b for b in BUSES if b is not None]
        self.assertEqual(len(ids), len(set(ids)))

    def test_endpoint_identity(self):
        """An endpoint's get_runtime() and get_id() lead back to the same device, and
        the dump's endpoint references say the same."""
        dumped = knexus.dump()["children"]["links"]
        self.assertEqual(len(dumped), len(LINKS))
        for k, link in enumerate(LINKS):
            with self.subTest(link=k, type=link["Type"]):
                for dev, ref in zip(link["ends"], dumped[k]["endpoints"]):
                    if not dev:
                        self.assertIsNone(ref)
                        continue
                    rt = dev.get_runtime()
                    self.assertTrue(rt, "an endpoint names its runtime")
                    name, idx = rt.get_property_str("Name"), dev.get_id()
                    same = knexus.get_runtime(name).get_device(idx)
                    self.assertEqual(bus_id(same), bus_id(dev))
                    self.assertEqual(same.get_property_str("Name"), dev.get_property_str("Name"))
                    self.assertEqual(ref["runtime"], name)
                    self.assertEqual(ref["device"], idx)
                    self.assertEqual(ref.get("PciBusId"), bus_id(dev))


if __name__ == '__main__':
    unittest.main()
