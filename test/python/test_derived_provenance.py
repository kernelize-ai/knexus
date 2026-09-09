#!/usr/bin/env python3
"""
Provenance audit for every ``Source: "Derived"`` figure in the device library.

``Derived`` is the one provenance value in ``schema/device_info_schema.json`` that
claims an *arithmetic*: "it is computed from other figures in this file by an
arithmetic the surrounding Description states". Nothing enforced that. This audit
does, by proving each Derived figure sits in exactly one of two classes:

**field-reproducible** -- the value recomputes, within ``REL_TOLERANCE``, from other
    *fields* of the same device file by one of the named formulas in ``FIELD_ROUTES``
    below. The report prints the formula, every operand with its JSON pointer, the
    computed result and the relative error.

**prose-sourced** -- the value does not reconcile against any field, but the number is
    stated in that file's own ``Description``/``description`` prose. The report prints
    the JSON pointer of the prose and quotes the sentence. **If a field route fired and
    disagreed, only the carrying entry's OWN prose may vouch for the figure** -- see
    ``trace_in_prose``. A bare numeral is not evidence on its own: a board wattage of
    "700 W" in a chip description collides with 700 GB/s and would otherwise "prove" a
    stack bandwidth that the file's own aggregate arithmetic put at 670. This class is not a
    weaker excuse: it is the honest home for a figure whose real source is a vendor
    speed grade named in prose that the file's rounded bandwidth field cannot
    reproduce (every Apple ``TransferRate: 6.4`` is exactly this -- see
    ``_prose_relations``).

A Derived figure in neither class is a **finding**: nothing in the file supports the
number. The audit reports it and fails; it never edits device data.

What this audit deliberately does NOT do
----------------------------------------
It never asserts ``chip total == per-unit figure x Count``. Vendor rounding makes that
false all over this corpus while both figures are correct -- Blackhole publishes 664
TFLOP/s against a benchmarked "roughly 5.4" per engine (x 120 = 648), trn1 publishes
190 against 92 x 2, gfx950 publishes 8000 GB/s against 8 x 1024. Only ``Derived``
figures are inspected at all, so no ``Published`` figure is ever condemned by
arithmetic. Every route below is an *acceptor*: a route that does not fire moves the
figure to the next route, so a route can never force two honest numbers to agree.

Two pre-existing data defects are visible from here and are neither used nor fixed:
``nvidia-gpu-sm_90.json``'s ``L2`` and ``apple-gpu-m1m.json``'s
``SLC`` each carry the *DRAM* bandwidth on a cache row. The
``aggregate-share`` route only reads a memory entry that carries ``BankCount``
alongside ``MaxMemoryBandwidth``, which excludes both.

Like its neighbour ``test_device_schema.py`` this needs neither a build nor the
``knexus`` extension module -- and, unlike that one, not even ``jsonschema``: it is
stdlib-only, so it runs anywhere::

    # standalone, with the full per-figure report (no pytest needed)
    python3 test/python/test_derived_provenance.py

    # under the gate
    python3 -m pytest test/python/test_derived_provenance.py -q
    python3 test/python/run_tests.py
    ctest --test-dir build -R derived_provenance
"""

import fnmatch
import json
import os
import re
import sys
import unittest
from collections import namedtuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_LIB_DIR = "device_lib"

#: Relative tolerance for every numeric comparison in this file.
#:
#: It has to admit the roundings the corpus legitimately contains and reject the ones
#: that mean a different quantity was recorded. Measured over the corpus, the widest
#: legitimate gap is trn1's ``TransferRate: 3.2`` against 820 GB/s over a 2048-bit bus
#: (3.203125, 0.098% off) and the tightest illegitimate one is Apple's
#: ``TransferRate: 6.4`` against a bandwidth field rounded down from 409.6 to 400 GB/s
#: (6.25, 2.4% off). 0.5% is the geometric middle of that band: 5x clear of the widest
#: honest rounding, 5x clear of the narrowest real mismatch. Tightening it below 0.1%
#: would condemn trn1; loosening it past 2% would wave the Apple values through as
#: field-reproducible and lose the distinction this audit exists to make.
REL_TOLERANCE = 5e-3

#: A field route that misses by more than this is not reported as a near miss in the
#: per-figure report -- it was reconstructing a different quantity, not rounding.
#: Findings still list every route that was tried, however far off.
NEAR_MISS_LIMIT = 0.10


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def _pointer(parts):
    """Render path segments as an RFC 6901 JSON pointer."""
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def _close(actual, expected):
    """True when two figures agree to REL_TOLERANCE, relative to the larger."""
    scale = max(abs(actual), abs(expected))
    if scale == 0.0:
        return True
    return abs(actual - expected) <= REL_TOLERANCE * scale


def _error(actual, expected):
    scale = max(abs(actual), abs(expected))
    return 0.0 if scale == 0.0 else abs(actual - expected) / scale


def _num(value):
    """Render a number without a trailing '.0' on whole values."""
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _number_or_none(container, key):
    value = container.get(key) if isinstance(container, dict) else None
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# --------------------------------------------------------------------------------------
# the corpus
# --------------------------------------------------------------------------------------

#: One ``Derived`` figure and everything a route needs to explain it.
#:
#: ``kind`` is ``"scalar"`` (a ``ClockRate``/``LaneWidth``/``TransferRate``/``Latency``
#: under a ``Performance`` block whose ``Source`` is Derived) or ``"throughput"`` (an
#: item of ``Performance.Throughput`` whose own ``Source`` is Derived).
Figure = namedtuple(
    "Figure",
    "relpath path name value unit kind doc entry entry_path block block_path siblings")

#: How one figure was explained. ``cls`` is 'field-reproducible' or 'prose-sourced'.
Trace = namedtuple("Trace", "cls route detail quote quote_path")

#: An unexplained Derived figure, or a corpus-level problem.
Finding = namedtuple("Finding", "check locator message")

_SCALARS = ("ClockRate", "LaneWidth", "TransferRate", "Latency")


def load_documents():
    """Load every ``device_lib/*.json`` as ``(relpath, document)`` pairs."""
    directory = os.path.join(REPO_ROOT, DEVICE_LIB_DIR)
    documents = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name)) as handle:
            documents.append(("%s/%s" % (DEVICE_LIB_DIR, name), json.load(handle)))
    return documents


def collect(documents):
    """Every Derived figure in the corpus, plus the provenance census of all figures."""
    figures, census = [], {}

    def visit(doc, relpath, node, path, parent, parent_path):
        if isinstance(node, dict):
            if path and path[-1] == "Performance" and isinstance(parent, dict):
                block_source = node.get("Source")
                for name in _SCALARS:
                    if name not in node:
                        continue
                    census[block_source] = census.get(block_source, 0) + 1
                    if block_source == "Derived":
                        figures.append(Figure(
                            relpath, path + (name,), name, node[name], None, "scalar",
                            doc, parent, parent_path, node, path, ()))
                items = node.get("Throughput") or ()
                for index, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    census[item.get("Source")] = census.get(item.get("Source"), 0) + 1
                    if item.get("Source") == "Derived":
                        figures.append(Figure(
                            relpath, path + ("Throughput", index), "Rate",
                            item.get("Rate"), item.get("Unit"), "throughput",
                            doc, parent, parent_path, node, path, items))
            for key, value in node.items():
                visit(doc, relpath, value, path + (key,), node, path)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(doc, relpath, value, path + (index,), parent, parent_path)

    for relpath, document in documents:
        visit(document, relpath, document, (), None, ())
    return figures, census


def memory_entries(doc):
    """``(path, entry)`` for every ``MemorySubsystem.MemoryTypes`` entry of a file.

    ``MemoryTypes`` is a name-keyed map, so the path names the memory rather than
    numbering it: an operand pointer reads ``.../MemoryTypes/HBM3/...``
    and survives a memory being added or removed.
    """
    entries = doc.get("MemorySubsystem", {}).get("MemoryTypes", {}) or {}
    if not isinstance(entries, dict):
        return []
    return [(("MemorySubsystem", "MemoryTypes", name), e)
            for name, e in entries.items() if isinstance(e, dict)]


def prose_fragments(doc, path=()):
    """``(path, text)`` for every Description/description string in a file, in order."""
    out = []
    if isinstance(doc, dict):
        for key, value in doc.items():
            if key in ("Description", "description") and isinstance(value, str):
                out.append((path + (key,), value))
            else:
                out += prose_fragments(value, path + (key,))
    elif isinstance(doc, list):
        for index, value in enumerate(doc):
            out += prose_fragments(value, path + (index,))
    return out


# --------------------------------------------------------------------------------------
# class 1 -- field-reproducible
# --------------------------------------------------------------------------------------
#
# Each route returns (route-name, explanation, computed value) or None. A route reads
# only *fields* of the same file, never prose, and never invents an operand: if an
# operand is missing the route declines and the next one is tried.

def _best(fig, candidates):
    """The candidate closest to the recorded figure, or None if a route had none.

    A route that has operands but does not reproduce the value still returns its best
    attempt: the driver reports it as a near miss, so a failure says what the fields
    actually imply instead of going silent.
    """
    if not candidates:
        return None
    return min(candidates, key=lambda c: _error(float(fig.value), c[2]))


def route_transfer_rate_from_bandwidth(fig):
    """TransferRate (GT/s) = MaxMemoryBandwidth (GB/s) x 8 / maxBusWidth (bit).

    The inverse of the plan's formula, for the signalling rate of a memory entry that
    publishes both its aggregate bandwidth and its bus width.
    """
    if fig.kind != "scalar" or fig.name != "TransferRate":
        return None
    bandwidth = _number_or_none(fig.entry, "MaxMemoryBandwidth")
    width = _number_or_none(fig.entry, "maxBusWidth")
    if bandwidth is None or not width:
        return None
    return ("bandwidth-to-signalling-rate",
            "MaxMemoryBandwidth(%s @ %s) x 8 / maxBusWidth(%s @ %s)" % (
                _num(bandwidth), _pointer(fig.entry_path + ("MaxMemoryBandwidth",)),
                _num(width), _pointer(fig.entry_path + ("maxBusWidth",))),
            bandwidth * 8.0 / width)


def route_transfer_rate_from_memory_entry(fig):
    """TransferRate on a *unit* entry = the same arithmetic over the file's memory entry.

    A ``HBM Stack`` under ``CoreSubsystem`` carries no bandwidth or bus-width fields of
    its own -- those live on the ``MemorySubsystem`` entry for the memory it fronts. Only
    an entry that lacks both operands itself reaches across, and only a memory entry
    carrying both is read; the report names which entry supplied them.
    """
    if fig.kind != "scalar" or fig.name != "TransferRate":
        return None
    if _number_or_none(fig.entry, "MaxMemoryBandwidth") is not None:
        return None
    candidates = []
    for mem_path, entry in memory_entries(fig.doc):
        bandwidth = _number_or_none(entry, "MaxMemoryBandwidth")
        width = _number_or_none(entry, "maxBusWidth")
        if bandwidth is None or not width:
            continue
        candidates.append((
            "bandwidth-to-signalling-rate (via the memory entry)",
            "MaxMemoryBandwidth(%s @ %s) x 8 / maxBusWidth(%s @ %s)" % (
                _num(bandwidth), _pointer(mem_path + ("MaxMemoryBandwidth",)),
                _num(width), _pointer(mem_path + ("maxBusWidth",))),
            bandwidth * 8.0 / width))
    return _best(fig, candidates)


def route_clock_from_power_efficiency(fig):
    """ClockRate (MHz) = the file's PowerEfficiency BoostClock / BaseClock."""
    if fig.kind != "scalar" or fig.name != "ClockRate":
        return None
    power = fig.doc.get("PowerEfficiency") or {}
    candidates = []
    for key in ("BoostClock", "BaseClock"):
        clock = _number_or_none(power, key)
        if clock is None:
            continue
        candidates.append(("clock-from-power-efficiency",
                           "PowerEfficiency.%s(%s @ %s)" % (
                               key, _num(clock), _pointer(("PowerEfficiency", key))),
                           float(clock)))
    return _best(fig, candidates)


def route_bandwidth_from_transfer_rate(fig):
    """Throughput (GB/s) = TransferRate (GT/s) x maxBusWidth (bit) / 8.

    The formula the plan names. Both operands are in-file: the signalling rate on the
    entry's own Performance block, the bus width on the entry.
    """
    if fig.kind != "throughput" or fig.unit != "GB/s":
        return None
    rate = _number_or_none(fig.block, "TransferRate")
    width = _number_or_none(fig.entry, "maxBusWidth")
    if rate is None or width is None:
        return None
    return ("signalling-rate-to-bandwidth",
            "TransferRate(%s @ %s) x maxBusWidth(%s @ %s) / 8" % (
                _num(rate), _pointer(fig.block_path + ("TransferRate",)),
                _num(width), _pointer(fig.entry_path + ("maxBusWidth",))),
            rate * width / 8.0)


def route_restates_max_bandwidth(fig):
    """Throughput (GB/s) = the carrying entry's own MaxMemoryBandwidth."""
    if fig.kind != "throughput" or fig.unit != "GB/s":
        return None
    bandwidth = _number_or_none(fig.entry, "MaxMemoryBandwidth")
    if bandwidth is None:
        return None
    return ("restates-max-bandwidth",
            "MaxMemoryBandwidth(%s @ %s)" % (
                _num(bandwidth), _pointer(fig.entry_path + ("MaxMemoryBandwidth",))),
            float(bandwidth))


def route_aggregate_share(fig):
    """Throughput (GB/s) for one unit = a memory aggregate / the population count.

    A per-stack, per-channel or per-group bandwidth is the file's aggregate DRAM
    bandwidth divided by how many of them there are. The divisor is either the
    carrying entry's own ``Count`` or the memory entry's ``BankCount`` (which these
    files use for the stack/channel count) -- whichever reproduces the figure; the
    report names which.

    Only a memory entry carrying BOTH ``MaxMemoryBandwidth`` and ``BankCount`` is read,
    which is what keeps the two known DRAM-bandwidth-on-a-cache-row defects
    (sm_90's ``L2``, m1m's ``SLC``) out of the operand pool.

    This is not the forbidden ``chip == per-unit x Count`` assertion: it is an
    acceptor, applied only to figures that already claim to be Derived, and a
    mismatch simply means the next route is tried (gfx950's 1024 GB/s per stack takes
    exactly that path -- its published 8000 GB/s aggregate is rounded down from 8192).
    """
    if fig.kind != "throughput" or fig.unit != "GB/s":
        return None
    candidates = []
    for mem_path, entry in memory_entries(fig.doc):
        bandwidth = _number_or_none(entry, "MaxMemoryBandwidth")
        banks = _number_or_none(entry, "BankCount")
        if bandwidth is None or banks is None:
            continue
        divisors = [("this entry's Count", _number_or_none(fig.entry, "Count"),
                     fig.entry_path + ("Count",)),
                    ("BankCount", banks, mem_path + ("BankCount",))]
        for label, divisor, divisor_path in divisors:
            if not divisor:
                continue
            candidates.append((
                "aggregate-share",
                "MaxMemoryBandwidth(%s @ %s) / %s(%s @ %s)" % (
                    _num(bandwidth), _pointer(mem_path + ("MaxMemoryBandwidth",)),
                    label, _num(divisor), _pointer(divisor_path)),
                bandwidth / float(divisor)))
    return _best(fig, candidates)


def route_sparse_to_dense(fig):
    """A Dense rate = its Sparse sibling / 2.

    2:4 structured sparsity doubles the dense rate; the vendor publishes the sparse
    figure and the dense one is that halved. Required to be stated by the file: the
    route only fires when the file's own prose talks about sparsity.
    """
    if fig.kind != "throughput":
        return None
    item = fig.siblings[fig.path[-1]]
    mode = (item.get("Mode") or "")
    if "Dense" not in mode:
        return None
    if not any("sparsit" in text.lower() for _p, text in prose_fragments(fig.doc)):
        return None
    candidates = []
    for index, sibling in enumerate(fig.siblings):
        if index == fig.path[-1] or not isinstance(sibling, dict):
            continue
        if "Sparse" not in (sibling.get("Mode") or ""):
            continue
        if sibling.get("Precision") != item.get("Precision"):
            continue
        if sibling.get("Unit") != item.get("Unit"):
            continue
        rate = _number_or_none(sibling, "Rate")
        if rate is None:
            continue
        candidates.append((
            "sparse-to-dense",
            "sibling %s %s Rate(%s @ %s) / 2 (2:4 structured sparsity)" % (
                sibling.get("Precision"), sibling.get("Mode"), _num(rate),
                _pointer(fig.block_path + ("Throughput", index, "Rate"))),
            rate / 2.0))
    return _best(fig, candidates)


def route_ops_per_cycle_from_core_count(fig):
    """Throughput (op per cycle) for precision P = the file's count of P cores.

    An SM issues one operation per cycle per core of that precision, and these files
    record the cores as their own ``UnitTypes`` entry scoped to the same parent -- so
    ``64 OP/cycle FP64`` is the ``FP64 Core`` entry's ``Count``, exactly as the
    sibling Published FP32 and INT32 rates were built.
    """
    if fig.kind != "throughput" or fig.unit not in ("OP/cycle", "FLOP/cycle", "MAC/cycle"):
        return None
    item = fig.siblings[fig.path[-1]]
    precision = item.get("Precision")
    if not precision:
        return None
    units = (fig.doc.get("CoreSubsystem") or {}).get("UnitTypes") or {}
    for name, entry in units.items():
        if not isinstance(entry, dict) or not name.upper().startswith(precision.upper()):
            continue
        if not re.search(r"\b(Core|Cores|Unit|Lane|Lanes)$", name):
            continue
        count = _number_or_none(entry, "Count")
        if count is None or not _close(float(fig.value), float(count)):
            continue
        return ("cores-of-that-precision",
                "Count(%s @ %s) of the %r entry -- one %s op per core per cycle" % (
                    _num(count), _pointer(("CoreSubsystem", "UnitTypes", name, "Count")),
                    name, precision),
                float(count))
    return None


def route_stated_ratio_of_sibling(fig):
    """A rate = a sibling rate / N, where the file's prose states the '1/N' ratio.

    NVIDIA's consumer parts publish one FP32 number and state the FP64 rate as a
    fraction of it. The divisor is not invented: only an N that appears as the literal
    token ``1/N`` somewhere in this file's prose is tried, and the sentence is quoted.
    """
    if fig.kind != "throughput":
        return None
    item = fig.siblings[fig.path[-1]]
    precision = item.get("Precision")
    if not precision:
        return None
    ratios = []
    for path, text in prose_fragments(fig.doc):
        for match in re.finditer(r"\b1\s*/\s*(\d+)\b", text):
            sentence = _sentence_of(text, match.start())
            # The ratio has to be stated ABOUT this figure, not merely be present as a
            # numeral: "MPEG-1/2/4" in a video-decoder description is not a rate ratio.
            if precision not in sentence:
                continue
            ratios.append((int(match.group(1)), path, sentence))
    for divisor, path, sentence in ratios:
        if divisor <= 1:
            continue
        for index, sibling in enumerate(fig.siblings):
            if index == fig.path[-1] or not isinstance(sibling, dict):
                continue
            if (sibling.get("Precision") or "") not in sentence:
                continue
            rate = _number_or_none(sibling, "Rate")
            if rate is None:
                continue
            computed = rate / float(divisor)
            if _close(float(fig.value), computed):
                return ("stated-ratio-of-sibling",
                        "sibling %s %s Rate(%s @ %s) / %d -- ratio stated at %s: %r" % (
                            sibling.get("Precision"), sibling.get("Mode"), _num(rate),
                            _pointer(fig.block_path + ("Throughput", index, "Rate")),
                            divisor, _pointer(path), sentence),
                        computed)
    return None


#: Tried in order; the first route that reproduces the value explains it.
FIELD_ROUTES = (
    route_transfer_rate_from_bandwidth,
    route_transfer_rate_from_memory_entry,
    route_clock_from_power_efficiency,
    route_bandwidth_from_transfer_rate,
    route_restates_max_bandwidth,
    route_aggregate_share,
    route_sparse_to_dense,
    route_ops_per_cycle_from_core_count,
    route_stated_ratio_of_sibling,
)


# --------------------------------------------------------------------------------------
# class 2 -- prose-sourced
# --------------------------------------------------------------------------------------

#: A sentence break is a period or semicolon followed by the start of the next one
#: -- an uppercase letter or a digit. That keeps an abbreviation such as
#: "i.e. the 256 GB/s figure" in one piece, and decimals never split at all.
_SENTENCE_SPLIT = re.compile(r"(?<=[.;]) +(?=[A-Z0-9])")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d\d\d\b)")
_NUMERAL = re.compile(r"\d+(?:\.\d+)?")


def _sentence_of(text, offset):
    """The sentence of ``text`` containing character ``offset``."""
    start = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        if match.end() > offset:
            break
        start = match.end()
    end = len(text)
    for match in _SENTENCE_SPLIT.finditer(text, start):
        if match.start() >= offset:
            end = match.start()
            break
    return text[start:end].strip()


def _prose_relations(fig):
    """How a prose numeral may stand for this figure. Each is a named convention.

    ``verbatim``      -- the figure itself is written out.
    ``unit-scale``    -- the same quantity in the neighbouring metric unit: a signalling
                         rate of 6.4 GT/s is universally written "6400 MT/s" or, as a
                         speed grade, "LPDDR5-6400"; a 1024 GB/s stack is written as its
                         1024-bit interface. Only a factor of exactly 1000 is accepted.
    ``population``    -- the prose gives the aggregate over the entry's whole
                         population and the field gives the per-unit share, so the
                         prose numeral is the figure times the entry's own ``Count``.
    """
    value = float(fig.value)
    relations = [("verbatim", value, "the figure as written")]
    relations.append(("unit-scale", value * 1000.0,
                      "%s x 1000 -- the same quantity one metric prefix down "
                      "(GT/s stated as MT/s, TB/s as GB/s)" % _num(value)))
    count = _number_or_none(fig.entry, "Count")
    if count and count > 1:
        relations.append(("population", value * count,
                          "%s x this entry's Count(%s @ %s) -- prose states the "
                          "aggregate, the field states one unit's share" % (
                              _num(value), _num(count),
                              _pointer(fig.entry_path + ("Count",)))))
    return relations


def trace_in_prose(fig, own_only=False):
    """Find the figure in this file's prose. The entry's own Description wins.

    ``own_only`` narrows the search to the carrying entry's own prose. The driver sets
    it whenever a field route fired and DISAGREED: once the file's own fields say this
    figure should be X and it is not X, a bare numeral somewhere else in the file
    cannot vouch for it. Without that restriction a wrong value is validated by any
    unrelated number that happens to collide with it -- a board wattage of "700 W" in
    the chip's description will happily "prove" a 700 GB/s stack bandwidth whose own
    aggregate arithmetic said 670. A numeral in the entry's OWN prose is a different
    claim: it is that entry describing itself, which is exactly how the honest
    prose-sourced figures are written ("LPDDR5-6400", "a 1024-bit interface").

    When no field route fires at all, nothing contradicts the figure and the whole
    file's prose stays available -- m2's GPU throughput is stated in the neighbouring
    ``GPU Core`` description and is correct there.
    """
    own, rest = [], []
    for path, text in prose_fragments(fig.doc):
        (own if path[:-1] == fig.entry_path else rest).append((path, text))
    if own_only:
        rest = []

    # Relations outermost: a figure written out in full anywhere in the searched prose
    # beats a scaled or aggregated reading of some other sentence's numeral.
    best, distinct = None, set()
    for name, expected, explanation in _prose_relations(fig):
        for path, text in own + rest:
            clean = _THOUSANDS.sub("", text)
            for match in _NUMERAL.finditer(clean):
                numeral = float(match.group(0))
                if not _close(numeral, expected):
                    continue
                distinct.add(round(numeral, 6))
                if best is not None:
                    continue
                where = "the entry's own prose" if (path, text) in own else (
                    "prose at %s" % _pointer(path))
                best = (name, "numeral %s in %s = %s" % (
                    _num(numeral), where, explanation),
                    _sentence_of(clean, match.start()), _pointer(path))
    if best is None:
        return None

    # How discriminating was the match? One candidate means the prose names this figure
    # and nothing else in the searched text could be mistaken for it. Several means the
    # trace rests on a numeral that had competitors -- still reported, but say so.
    name, detail, quote, quote_path = best
    corroboration = ("sole matching numeral in the %s prose searched"
                     % ("entry's own" if own_only else "file's")) if len(distinct) == 1 \
        else ("AMBIGUOUS: %d different numerals in the searched prose match this "
              "figure (%s)" % (len(distinct), ", ".join(_num(n) for n in sorted(distinct))))
    return Trace(cls="prose-sourced", route="prose-%s" % name,
                 detail="%s -- %s" % (detail, corroboration),
                 quote=quote, quote_path=quote_path)


# --------------------------------------------------------------------------------------
# negative probes -- the audit's own proof that it is not a tautology
# --------------------------------------------------------------------------------------

#: A Derived figure, a WRONG value for it, and why that wrong value is a trap. Each is
#: applied to an in-memory copy of the file (nothing on disk is touched) and the audit
#: must refuse to explain it. Without these an audit that accepted everything would look
#: exactly like an audit that verified everything.
#:
#: ``path`` is the figure's own path: a Performance scalar ends in the field name, a
#: Throughput item ends in its index.
Probe = namedtuple("Probe", "relpath path value why")

NEGATIVE_PROBES = (
    Probe("device_lib/apple-gpu-m4.json",
          ("MemorySubsystem", "MemoryTypes", "Unified Memory", "Performance",
           "TransferRate"), 7.8,
          "a field-reproducible signalling rate falsified; the fields say 7.5"),
    Probe("device_lib/apple-gpu-apple7-m1max.json",
          ("MemorySubsystem", "MemoryTypes", "Unified Memory", "Performance",
           "TransferRate"), 6.9,
          "a prose-sourced signalling rate falsified; neither 6.9 nor 6900 is written "
          "anywhere in the file"),
    Probe("device_lib/apple-gpu-applegpu_g16s.json",
          ("MemorySubsystem", "MemoryTypes", "Unified Memory", "Performance",
           "TransferRate"), 8.60,
          "a tolerance probe: 0.80% off 8.53125, well inside any loose tolerance and "
          "outside this one"),
    Probe("device_lib/tenstorrent-npu-blackhole.json",
          ("CoreSubsystem", "UnitTypes", "DRAM Controller", "Performance", "Throughput", 0),
          66,
          "a derived per-controller bandwidth falsified; the aggregate says 512/8 = 64"),
    Probe("device_lib/nvidia-gpu-sm_90.json",
          ("CoreSubsystem", "UnitTypes", "Chip", "Performance", "Throughput", 10), 1900,
          "a dense tensor rate falsified; its sparse sibling halves to 1979"),
    Probe("device_lib/nvidia-gpu-sm_90.json",
          ("CoreSubsystem", "UnitTypes", "HBM3 Stack", "Performance", "Throughput", 0), 700,
          "THE COLLISION CASE: 700 GB/s is wrong (3350/5 = 670) but '700 W' -- the board "
          "power -- appears in the Chip description. A cross-entry numeral must not "
          "vouch for a figure the file's own arithmetic contradicts"),
    Probe("device_lib/nvidia-gpu-sm_120.json",
          ("CoreSubsystem", "UnitTypes", "GDDR7 Device", "Performance", "Throughput", 0),
          575,
          "the same trap on another file and another vendor: 575 GB/s is wrong "
          "(1792/16 = 112) and '575 W' is the RTX 5090 board power in the Chip "
          "description"),
)


def apply_probe(document, probe):
    """Return a copy of ``document`` with the probe's figure falsified."""
    copy = json.loads(json.dumps(document))
    node = copy
    for segment in probe.path[:-1]:
        node = node[segment]
    last = probe.path[-1]
    if isinstance(last, int):
        node[last]["Rate"] = probe.value      # a Throughput item
    else:
        node[last] = probe.value              # a Performance scalar
    return copy


def run_probe(probe):
    """Audit the falsified figure alone. Returns its Audited result, or None if the
    probe no longer names a Derived figure (which is itself a failure -- the probe has
    gone stale and is proving nothing)."""
    directory = os.path.join(REPO_ROOT, DEVICE_LIB_DIR)
    name = probe.relpath.split("/")[-1]
    with open(os.path.join(directory, name)) as handle:
        document = json.load(handle)
    figures, _census = collect([(probe.relpath, apply_probe(document, probe))])
    for fig in figures:
        if fig.path == probe.path:
            return audit_figure(fig)
    return None


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

Audited = namedtuple("Audited", "figure trace near_misses")
Report = namedtuple("Report", "files census audited findings")


def audit_figure(fig):
    """Classify one Derived figure, recording every field route that was tried."""
    near_misses = []
    for route in FIELD_ROUTES:
        try:
            result = route(fig)
        except (TypeError, ZeroDivisionError):
            result = None
        if result is None:
            continue
        name, explanation, computed = result
        if _close(float(fig.value), computed):
            return Audited(fig, Trace(
                cls="field-reproducible", route=name,
                detail="%s = %s (%.4f%% from the recorded %s)" % (
                    explanation, _num(computed),
                    100.0 * _error(float(fig.value), computed), _num(fig.value)),
                quote=None, quote_path=None), near_misses)
        near_misses.append((
            _error(float(fig.value), computed),
            "%s: %s = %s, %.2f%% from the recorded %s" % (
                name, explanation, _num(computed),
                100.0 * _error(float(fig.value), computed), _num(fig.value))))

    # A field route that fired and missed is a contradiction on the record: restrict the
    # prose search to the entry's own words so an unrelated numeral cannot overrule it.
    trace = trace_in_prose(fig, own_only=bool(near_misses))
    return Audited(fig, trace, near_misses)


def run_checks():
    documents = load_documents()
    figures, census = collect(documents)

    audited, findings = [], []
    for fig in figures:
        if not isinstance(fig.value, (int, float)) or isinstance(fig.value, bool):
            findings.append(Finding(
                check="derived-not-numeric",
                locator="%s#%s" % (fig.relpath, _pointer(fig.path)),
                message="Derived figure is not a number: %r" % (fig.value,)))
            continue
        result = audit_figure(fig)
        audited.append(result)
        if result.trace is None:
            findings.append(Finding(
                check="derived-unsupported",
                locator="%s#%s" % (fig.relpath, _pointer(fig.path)),
                message="%s = %s is marked Derived but is NEITHER reproducible from "
                        "same-file fields by any known formula NOR stated in %s. "
                        "Tried: %s" % (
                            fig.name, _num(fig.value),
                            "this entry's own prose -- and because a field route fired "
                            "and disagreed, prose elsewhere in the file was NOT "
                            "consulted" if result.near_misses else "this file's prose",
                            "; ".join(t for _e, t in result.near_misses)
                            or "no formula applied")))

    if not documents:
        findings.append(Finding("corpus", DEVICE_LIB_DIR, "no device files were read"))
    if not audited:
        findings.append(Finding(
            "corpus", DEVICE_LIB_DIR,
            "no Derived figure was found -- the audit validated nothing"))

    return Report(len(documents), census, audited, findings)


def format_report(report):
    lines = []
    total = sum(report.census.values())
    lines.append("corpus:  %s/*.json -- %d files, %d provenance-tagged figures "
                 "(%s)" % (
                     DEVICE_LIB_DIR, report.files, total,
                     ", ".join("%d %s" % (n, k) for k, n in sorted(
                         report.census.items(), key=lambda kv: -kv[1]))))
    lines.append("tolerance: %.3g relative on every comparison" % REL_TOLERANCE)
    lines.append("")

    for cls, title in (("field-reproducible",
                        "FIELD-REPRODUCIBLE -- recomputed from same-file fields"),
                       ("prose-sourced",
                        "PROSE-SOURCED -- traced to a quoted sentence in the same file")):
        members = [a for a in report.audited if a.trace and a.trace.cls == cls]
        lines.append("%s (%d)" % (title, len(members)))
        lines.append("-" * 86)
        for item in members:
            fig = item.figure
            lines.append("  %s#%s" % (fig.relpath, _pointer(fig.path)))
            lines.append("      value   %s = %s%s" % (
                fig.name, _num(fig.value), " %s" % fig.unit if fig.unit else ""))
            lines.append("      route   %s" % item.trace.route)
            lines.append("      trace   %s" % item.trace.detail)
            if item.trace.quote:
                lines.append("      quote   %s" % item.trace.quote_path)
                lines.append("              %r" % item.trace.quote)
            # Only a *near* miss earns a note: a route whose operands are an order of
            # magnitude away was answering a different question, and saying so is noise.
            for error, miss in item.near_misses:
                if error <= NEAR_MISS_LIMIT:
                    lines.append(
                        "      note    field arithmetic does NOT reconcile -- %s" % miss)
        lines.append("")

    if report.findings:
        lines.append("UNSUPPORTED DERIVED FIGURES (%d) -- neither class explains these:"
                     % len(report.findings))
        for finding in report.findings:
            lines.append("  [%s] %s" % (finding.check, finding.locator))
            lines.append("      %s" % finding.message)
    else:
        lines.append("UNSUPPORTED DERIVED FIGURES: none -- every Derived figure is in "
                     "exactly one class.")
    return "\n".join(lines)


def main(argv=None):
    report = run_checks()
    print(format_report(report))
    print("")
    ok = not report.findings
    print("RESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
# the pytest / unittest wrapper -- a thin skin over run_checks()
# --------------------------------------------------------------------------------------

_REPORT = None


def _report():
    global _REPORT
    if _REPORT is None:
        _REPORT = run_checks()
    return _REPORT


class TestDerivedProvenance(unittest.TestCase):
    """Every Derived figure is field-reproducible or prose-sourced, and says which."""

    def _assert_clean(self, check):
        report = _report()
        offenders = [f for f in report.findings if fnmatch.fnmatch(f.check, check)]
        if offenders:
            self.fail("%d Derived figure(s) matching %r are unsupported\n\n%s" % (
                len(offenders), check, format_report(report)))

    def test_corpus_is_not_empty(self):
        """Guard against a green run that audited nothing."""
        report = _report()
        self.assertGreater(report.files, 0, "no device_lib/*.json files were read")
        self.assertGreater(len(report.audited), 0, "no Derived figure was audited")
        self.assertGreater(report.census.get("Published", 0), 0,
                           "no Published figure was seen -- the walk missed the corpus")

    def test_every_derived_figure_is_supported(self):
        """A Derived value must reproduce from fields or be stated in prose."""
        self._assert_clean("derived-*")

    def test_every_derived_figure_is_numeric(self):
        self._assert_clean("derived-not-numeric")

    def test_every_audited_figure_has_exactly_one_class(self):
        """The report must say which class each figure is in -- never both, never neither."""
        for item in _report().audited:
            locator = "%s#%s" % (item.figure.relpath, _pointer(item.figure.path))
            self.assertIsNotNone(item.trace, "%s has no class" % locator)
            self.assertIn(item.trace.cls, ("field-reproducible", "prose-sourced"),
                          "%s has an unknown class %r" % (locator, item.trace.cls))

    def test_a_falsified_derived_figure_is_rejected(self):
        """The audit must FAIL on wrong data, or its PASS means nothing.

        Every probe in NEGATIVE_PROBES is applied to an in-memory copy of its file --
        nothing on disk is touched -- and must end up in neither class.
        """
        for probe in NEGATIVE_PROBES:
            result = run_probe(probe)
            self.assertIsNotNone(
                result, "probe is stale, it no longer names a Derived figure: %s#%s"
                % (probe.relpath, _pointer(probe.path)))
            if result.trace is not None:
                self.fail(
                    "FALSE PASS: %s#%s set to %s was accepted as %s via %r (%s).\n"
                    "This probe exists because: %s" % (
                        probe.relpath, _pointer(probe.path), _num(probe.value),
                        result.trace.cls, result.trace.route, result.trace.detail,
                        probe.why))

    def test_prose_sourced_figures_quote_their_prose(self):
        """A prose-sourced figure is only traced if the sentence is actually quoted."""
        for item in _report().audited:
            if item.trace and item.trace.cls == "prose-sourced":
                locator = "%s#%s" % (item.figure.relpath, _pointer(item.figure.path))
                self.assertTrue(item.trace.quote, "%s quotes no prose" % locator)
                self.assertTrue(item.trace.quote_path, "%s cites no pointer" % locator)


if __name__ == "__main__":
    sys.exit(main())
