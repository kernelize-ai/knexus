#!/usr/bin/env python3
"""
Kernel runs on every tenstorrent device. The tt-metal runtime opens the system
mesh once and each plugin device is a 1x1 submesh of it
(plugins/tenstorrent/tt_runtime.h): every buffer, program and workload of a
device goes to its submesh. This runs add_vectors on each device in turn,
through the same flow as any knexus kernel (buffers, CB constants, args,
finalize, schedule run, copy back), and checks every element.

The kernel source is not in the repo: KNEXUS_TT_KERNEL names it, else
tt-kernel.h in the product root. Skips when knexus, torch, the tt-metal
runtime, its devices or the kernel source are missing. Host tensors are torch
bfloat16: the knexus buffer API reads a tensor's data_ptr(), which a numpy
array lacks.
"""

import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))

try:
    import torch
except ImportError:
    torch = None

try:
    import knexus
except ImportError:
    print("Warning: knexus module not found. Tests will be skipped.")
    knexus = None

# knexus.utils decides once, at its own import, whether torch is loaded; under
# run_tests.py another module imports knexus before this one imports torch,
# and create_buffer then cannot type a tensor ("Failed to get data type").
if knexus is not None and torch is not None:
    knexus.utils.torch_loaded = True

KERNEL = os.environ.get('KNEXUS_TT_KERNEL', os.path.join(ROOT, 'tt-kernel.h'))
N_CORES = 8
N_TILES = 2
TILE = 1024  # elements per add_vectors program instance: one bf16 tile
SIZE = TILE * N_TILES * N_CORES


def tt_runtime():
    """The tt-metal runtime, or None when it is not loaded."""
    if knexus is None:
        return None
    try:
        rt = knexus.get_runtime('tt-metal')
    except Exception:
        return None
    return rt if rt else None


RT = tt_runtime()


@unittest.skipIf(knexus is None, "knexus not importable")
@unittest.skipIf(torch is None, "torch not importable (knexus buffers need a tensor data_ptr)")
@unittest.skipIf(RT is None, "no tt-metal runtime")
class TestTTMesh(unittest.TestCase):

    def test_device_count(self):
        """A box with /dev/tenstorrent/3 states 4 devices, one per chip."""
        if not os.path.exists('/dev/tenstorrent/3'):
            self.skipTest("fewer than 4 tenstorrent chips on this host")
        self.assertEqual(RT.get_devices().size(), 4)

    def test_add_vectors_each_device(self):
        """add_vectors on each device, in turn, gives a + b in every element."""
        n = RT.get_devices().size()
        if n == 0:
            self.skipTest("the tt-metal runtime states no devices")
        if not os.path.isfile(KERNEL):
            self.skipTest(f"no tt kernel source at {KERNEL} (set KNEXUS_TT_KERNEL)")
        for i in range(n):
            with self.subTest(device=i):
                self.run_add(RT.get_device(i), i)

    def run_add(self, dev, i):
        a = torch.full((SIZE,), 1.0 + i, dtype=torch.bfloat16)
        b = torch.ones(SIZE, dtype=torch.bfloat16)
        res = torch.zeros(SIZE, dtype=torch.bfloat16)
        nb_a, nb_b, nb_res = dev.create_buffer(a), dev.create_buffer(b), dev.create_buffer(res)

        lib = dev.load_library(KERNEL)
        kern = lib.get_kernel('add_vectors')
        sched = dev.create_schedule()
        cmd = sched.create_command(kern)
        dtype = knexus.get_data_type(a)
        for cb in range(3):  # CB fifo depth 1 for a, b, res
            cmd.set_const(cb, 1, "CB", dtype)
        cmd.set_arg(0, nb_a, "a")
        cmd.set_arg(1, nb_b, "b")
        cmd.set_arg(2, nb_res, "res")
        cmd.set_arg(3, 1, "size")
        cmd.set_arg(4, N_TILES, "n_tiles")
        cmd.finalize([SIZE // TILE])  # one program instance per tile
        sched.run()
        nb_res.copy(res)

        want = 2.0 + i
        bad = torch.nonzero(res != want).flatten()
        self.assertEqual(bad.numel(), 0,
                         f"device {i}: {bad.numel()} of {SIZE} elements != {want}, "
                         f"first at {bad[:4].tolist()}: {res[bad[:4]].tolist()}")
        print(f"device {i}: add_vectors {SIZE} elements == {want}")
        for obj in (sched, lib, nb_res, nb_b, nb_a):
            obj.release()


if __name__ == '__main__':
    unittest.main()
