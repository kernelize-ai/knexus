# build — rebuilding knexus in place and cutting a wheel ※kmap's venv can pin

knexus is the sibling checkout `/media/sjw/dev/nexus.0` (`knexus/` under kqv; remote
`kernelize-ai/nexus`); the host and link rows were statused on kmap's
boards (※kmap/plans/knexus-discovery). Everything below was learned the hard way in
※kurs/kmap/folds/2026-10-26/20-knexus-discovery-2-host and
※kurs/kmap/folds/2026-10-26/20-knexus-discovery-4-release.

## rebuild in place

The build tree is `build/cmake.linux-x86_64-cpython-3.12`. It was configured
by pip's isolated build environment, whose cmake no longer exists, and
`ninja`/`cmake` live only in `nexus.0/.venv/bin`:

```bash
cd /media/sjw/dev/nexus.0
V=$PWD/.venv/bin; B=build/cmake.linux-x86_64-cpython-3.12
$V/cmake -DCMAKE_MAKE_PROGRAM=$V/ninja $B          # once, re-points the tree
$V/ninja -C $B -j24                                 # ~8 s incremental
cp $B/runtime_libs/*.so python/knexus/runtime_libs/ # setup.py's copy, by hand
```

`libknexus.so` lands in `python/knexus/_C/` directly. The Python suite is
`cd test/python && ../../.venv/bin/python run_tests.py`; it carries a
pre-existing 6 failures + 32 errors (pybind containers vs `list`, buffers)
that are not kmap's — count them, do not chase them. `test_host.py` and
`test_links.py` are the rows' tests; run them directly for a clean 4/4 each
(`run_tests.py --pattern` errors on modules lacking the class).

## cut a wheel

`python -m build --no-isolation` refuses (build-system pins `pybind11==3.1.0`,
the venv has 3.0.0 to match torch_nexus). The CI route works:

```bash
cd /media/sjw/dev/nexus.0 && export PATH=$PWD/.venv/bin:$PATH   # setup.py wants ninja on PATH
KNEXUS_WHEEL_VERSION=0.2.8 python setup.py -q bdist_wheel -d dist
```

Without the override the version comes from `git describe --tags`; keep
`setup.py` `DEFAULT_VERSION` and `python/knexus/__init__.py` `__version__` at
the same number as the tag you mean to cut (they drifted before). Debug is
the default build type; a Release build reconfigures and rebuilds everything.

## install where kmap's gate runs

There is no `uv` on this machine and `kmap/.venv` has no pip:

```bash
/usr/bin/python3 -m pip --python /media/sjw/dev/kmap/.venv/bin/python \
    install --no-deps --force-reinstall dist/knexus-0.2.8-*.whl
```

(`--python` goes before the subcommand.) kmap's probe refuses a knexus below
`kmap.probe.KNEXUS_FLOOR`, read from the wheel's metadata.

Since 0.2.12 kmap's gate venv carries the **PyPI wheel** instead (same pip
form, `install knexus==0.2.12`); the local wheel is for testing a knexus
change. The PyPI wheel's cuda plugin links `libcudart.so.12` and does not
bundle it: put a CUDA 12 runtime's lib dir on `LD_LIBRARY_PATH` (kmap's venv
installs `nvidia-cuda-runtime-cu12` for that, ※kurs/kmap/readme§the-gate)
or the plugin fails to `dlopen` and cuda shows up as an empty runtime —
`KNEXUS_LOG_ENABLE=1` prints the failing `dlopen`.

## knobs

- `KNEXUS_HOST_ROOT` — where the cpu plugin reads `sys/` and `proc/` from
  (default `/`); a fixture tree for tests. Read once per process.
- `KNEXUS_DEVICE_PATH` — knexus overwrites it at import; kmap restores the
  caller's value (※kurs/kmap/folds/2026-10-26/20-device-path-clobber).
