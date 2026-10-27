# knexus — the working cheat-sheet (stable facts the knexus session shares)

The facts every knexus session needs and should **not** re-derive: what knexus
is, where to read, the two checkouts, the gate, the build and release route,
and the commit convention the repository arrived with. This readme is the
**boot anchor** — generic **`/orient knexus`** loads it with the board and
recent commits; knexus ships no orient of its own.

> **Reference, don't restate.** Canonical truth lives in the product's own
> `docs/` and in each header's and module's comments. Follow the pointer.
>
> **knexus predates its governance.** The repository is `kernelize-ai/nexus`
> (GitHub also answers to `knexus`), a plain pull-request history with no
> kqv vocabulary in it. Its first governed work — the host and link
> properties kmap's probe reads — was planned and folded on kmap's side,
> ※kmap/plans/knexus-discovery; this tree starts after that.

## What knexus is (orientation)

**A device API: discovery, characterization and kernel deployment behind one
interface.** Four surfaces, two for users and two for vendors:

| surface | what it is | read |
| --- | --- | --- |
| Python API | `import knexus`: runtimes, devices, buffers, schedules, commands, events, and every **property** by name or enum | `docs/Python_API.md` |
| C++ source API | the same objects as `knexus::` classes over `std::optional<Property>` | `docs/Core_API.md` |
| runtime plugin C-API | one shared library per backend (`plugins/cpu`, `cuda`, `hip`, `metal`, `tenstorrent`) answering `nxsGet*Property` and the command verbs | `docs/Plugin_API.md` |
| JSON device DB | `device_lib/*.json`, one entry per chip, read through `Info` nodes | `docs/JSON_API.md` |

**Properties are the seam.** Every fact a plugin states crosses as bytes under
an `nxs_property` name (`include/knexus-api/_nxs_propertys.h`), typed int,
float, string, or a vector of each; a device's **links** are rows — `LinkCount`
of them, every `Link*` property a vector over rows, `LinkType` naming the
kind. The consumers are ※kmap (`kmap probe` quotes the host and the links)
and ※ktools (`kgen` reads the device brief through kmap's ladder).

## Repo layout

Two gitignored checkouts under kqv, **both branches of one repository**,
`git@github.com:kernelize-ai/nexus.git`:

- **product**: the working tree is a **sibling of kqv** (`../nexus.0`);
  `knexus/` under kqv is a **symlink** to it. If `/orient knexus` halts on
  "missing checkout", the symlink is what is missing: `ln -s ../nexus.0
  knexus` from the kqv root. The project's source branch is `main`; the
  checkout usually sits on a `sjw/<topic>` branch that reaches `main` by
  pull request (below).
- **governance**: `kurs/knexus/` on branch `kurs` — an orphan branch, no
  shared history with the product (※kqv/notes/project-guide).

## Where to read (the doc map)

| home | holds |
| --- | --- |
| `knexus/README.md` | the front door — the four interfaces, a Python walkthrough |
| `knexus/docs/` | `APIs.md` (the map), `Quick_Start.md`, `Build_and_CI.md`, `Logging.md`, and the four API docs above |
| header and module comments | **API truth** — `_nxs_propertys.h` is the property vocabulary; a plugin's `*_runtime.cpp` is what it states |
| `knexus/test/python/README.md` | the Python suite's own map |
| `kurs/knexus/` | governance — **`readme.md`** (this file), `plans/`, `tasks/`, **`folds/`**, `notes/` |

**Ranking when two disagree:** tested code wins, then a header's comment,
then `docs/`, then anything in kmap's history.

Project notes, `kurs/knexus/notes/`:

| note | holds — read when |
| --- | --- |
| ※kurs/knexus/notes/build | rebuilding in place with the venv's cmake and ninja, cutting a wheel with `setup.py`, installing it where ※kmap's gate runs, the `KNEXUS_HOST_ROOT` and `KNEXUS_DEVICE_PATH` knobs — before touching a plugin or releasing |

## Plans

- **board** — `plans/knexus.md`: one row per effort, the **single source of
  status** (glyphs: ※kqv/notes/boards).
- The host and link work is ※kmap/plans/knexus-discovery; the board's
  `discovery` row is its rollup.
- Per-plan paths: tasks `tasks/<plan>/N-slug.md`, folds
  `folds/YYYY-MM-DD/HH-<plan>-<task>.md`, status on the board.
- References are bare `※`/`§` tokens, never `../`-relative links —
  ※kqv/notes/cross-refs.

## The gate

Build first (※kurs/knexus/notes/build), then the Python suite from the
product checkout:

```bash
cd knexus/test/python && ../../.venv/bin/python run_tests.py      # the whole suite
cd knexus/test/python && ../../.venv/bin/python test_host.py       # one module, by file
```

The suite carries a **pre-existing red** — pybind containers compared to
`list`, buffer tests — that is not any row's; measure a change against the
baseline count the board's backlog names rather than chasing it. `test_host`
and `test_links` skip by name without Linux sysfs or a CUDA device. The
C++ half is gtest under `test/cpp`, built with `cmake`; CI's own run is
`docs/Build_and_CI.md`.

## The fold contract

A knexus task is **folded** (and only then commit-ready) when the gate has
been run and the fold, board row, and task status all exist — the general
kqv contract, ※kqv/notes/folds and ※kqv/notes/plan-task-fold.

## Commit scopes

The product keeps the repository's own convention, which predates kqv: a
subject `[Area] Imperative sentence`, bullets in the body, landed on `main`
by pull request (a direct push to `main` is not the house style).
Governance follows the generic shape, ※kqv/notes/commits:

| checkout | scope | example |
| --- | --- | --- |
| product `knexus/` (`main` via PR) | `[Area]`, the subsystem | `[Properties] …` · `[DeviceLibs] …` · `[Release] …` |
| governance `kurs/knexus/` (`kurs`) | the **plan**, no `knexus` prefix | `discovery: …` |

## Standing conventions

- **Nothing is measured.** A property is what the kernel, the driver or the
  entry declares, never a benchmark; a plugin that cannot state a property
  returns `NXS_InvalidProperty` rather than a guess.
- **A name has one home.** A property's name, type and description live in
  `_nxs_propertys.h` and its meaning in `docs/Python_API.md`; a plugin states
  it or not, and never under a second spelling.
- **Versions move together.** `python/knexus/__init__.py` and `setup.py`'s
  `DEFAULT_VERSION` match the tag a release cuts; a consumer pins a floor on
  that number (※kmap's `probe.KNEXUS_FLOOR`).
