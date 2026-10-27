# knexus — root board

> The **status home** for knexus, one row per plan; vocab ※kqv/notes/boards.
> History is the folds; what knexus *is* is ※kqv/kurs/knexus/readme.

## efforts

|     | plan | dep | done? |
| --- | ---- | --- | ----- |
| ● ▄ | discovery | — | knexus states the host (sockets, cores, threads, NUMA nodes) and each device's links as rows with a `LinkType`, and a string-vector property crosses into Python; planned and folded on kmap's side, ※kmap/plans/knexus-discovery, this row its rollup. |
| ○ ▷ ▁ | land-topo (no plan yet) | ← discovery | The property commits on the product's `sjw/topo` branch reach `main` by pull request and a tag cuts the release ※kmap's floor names, so a `pip install knexus` satisfies `kmap probe`. |

## backlog

Pointer-only — name the thing and where it would land.

- **The Python suite is red before any row** — a fixed count of failures and
  errors from pybind containers compared to `list` and from buffer tests;
  ※kurs/kmap/folds/2026-10-26/20-knexus-discovery-2-host measured it. A knexus
  row that makes `run_tests.py` exit 0, so the gate says green instead of
  "same red".
- **`python -m build` refuses** on the `pybind11` pin in `pyproject.toml`
  against the venv's torch-matched version; `setup.py bdist_wheel` is the
  route (※kurs/knexus/notes/build). Reconcile the pin or document the route
  in `docs/Build_and_CI.md`.
- **`setup.py`'s author line** carries an uncommitted change in the product
  checkout that no row made; decide it, don't absorb it.
- **Other plugins state no links** — hip, metal and tenstorrent answer none
  of the `Link*` rows; each is a plugin row with its own vendor API.
