#!/usr/bin/env python3
"""
Schema validation for the device library.

Every ``device_lib/*.json`` is validated against ``schema/device_info_schema.json``.
Known, not-yet-fixed violations live in the ``KNOWN_VIOLATIONS`` allowlist below --
each with a one-line reason -- so the checker is green today and stays honest: a *new*
violation fails, and an allowlist entry that no longer matches anything also fails,
so the allowlist cannot rot.

Unlike its neighbours in this directory, this test needs neither the ``knexus``
extension module nor a build -- it reads JSON and one header. It only needs
``jsonschema`` (see ``requirements.txt``). That makes it runnable anywhere:

    # standalone, with a human-readable report (no pytest needed)
    python3 test/python/test_device_schema.py

    # under the gate
    python3 -m pytest test/python/test_device_schema.py -q
    python3 test/python/run_tests.py
    ctest --test-dir build -R device_schema
"""

import fnmatch
import json
import os
import re
import sys
import unittest
from collections import namedtuple

try:
    import jsonschema
except ImportError:  # pragma: no cover - reported by the tests / main() below
    print("Warning: jsonschema module not found. Schema tests will be skipped.")
    jsonschema = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA_PATH = os.path.join("schema", "device_info_schema.json")
DEVICE_LIB_DIR = "device_lib"
PROP_TABLE_PATH = os.path.join("include", "knexus-api", "_nxs_propertys.h")


# --------------------------------------------------------------------------------------
# the allowlist
# --------------------------------------------------------------------------------------

#: One known violation. ``check`` and ``locator`` name *where* it is (both are matched as
#: shell globs, so one entry can cover a violation repeated across the corpus); ``reason``
#: is the one-line why-it-is-still-here.
Waiver = namedtuple("Waiver", "check locator reason")

#: A single unresolved schema/device-library violation found by a check below.
Finding = namedtuple("Finding", "check locator message")

# The known violations, in full. Grep for KNOWN_VIOLATIONS -- this tuple is the only
# place an exemption may be expressed; no check hard-codes an escape hatch of its own.
# It is empty: the corpus has no exemptions in force. An entry left behind after its
# violation is fixed fails test_allowlist_has_no_stale_entries.
KNOWN_VIOLATIONS = ()


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------

# Every keyword draft-07 defines, plus its annotation keywords. Anything else sitting in a
# subschema is inert: a validator silently ignores it, so it documents a constraint that is
# not enforced.
_DRAFT7_KEYWORDS = frozenset((
    "$schema", "$id", "$ref", "$comment", "title", "description", "default", "examples",
    "readOnly", "writeOnly", "definitions", "type", "enum", "const", "multipleOf",
    "maximum", "exclusiveMaximum", "minimum", "exclusiveMinimum", "maxLength", "minLength",
    "pattern", "items", "additionalItems", "maxItems", "minItems", "uniqueItems",
    "contains", "maxProperties", "minProperties", "required", "additionalProperties",
    "properties", "patternProperties", "dependencies", "propertyNames", "if", "then",
    "else", "allOf", "anyOf", "oneOf", "not", "format", "contentMediaType",
    "contentEncoding",
))

# Keywords whose value is itself a subschema / a map of them / a list of them.
_SUBSCHEMA = ("additionalProperties", "additionalItems", "contains", "propertyNames",
              "not", "if", "then", "else")
_SUBSCHEMA_MAP = ("properties", "patternProperties", "definitions")
_SUBSCHEMA_LIST = ("allOf", "anyOf", "oneOf")


def _pointer(parts):
    """Render path segments as an RFC 6901 JSON pointer."""
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def lint_schema(schema, path=()):
    """Report keys that sit in a subschema but are not JSON Schema keywords.

    These are the silent failures: the schema author wrote a constraint, and the
    validator throws it away. A subschema misplaced as a sibling of ``properties``
    rather than a member of it is the usual shape.
    """
    findings = []
    if not isinstance(schema, dict):
        return findings

    for key, value in schema.items():
        if key not in _DRAFT7_KEYWORDS:
            findings.append(Finding(
                check="schema-lint",
                locator="%s#%s" % (SCHEMA_PATH, _pointer(path + (key,))),
                message="not a JSON Schema keyword -- silently ignored, so nothing it "
                        "describes is enforced (misplaced sibling of 'properties'?)",
            ))
            continue

        sub = path + (key,)
        if key in _SUBSCHEMA:
            findings += lint_schema(value, sub)
        elif key in _SUBSCHEMA_MAP and isinstance(value, dict):
            for name, child in value.items():
                findings += lint_schema(child, sub + (name,))
        elif key in _SUBSCHEMA_LIST and isinstance(value, list):
            for i, child in enumerate(value):
                findings += lint_schema(child, sub + (i,))
        elif key == "items":
            if isinstance(value, list):
                for i, child in enumerate(value):
                    findings += lint_schema(child, sub + (i,))
            else:
                findings += lint_schema(value, sub)

    return findings


def _kind(instance):
    """A one-phrase summary of a JSON value, for error messages."""
    if isinstance(instance, list):
        return "array of %d item(s)" % len(instance)
    if isinstance(instance, dict):
        return "object with key(s) %s" % ", ".join(sorted(instance)[:6])
    return repr(instance)


def _describe(error):
    """jsonschema inlines the whole failing instance; keep the message readable."""
    message = error.message
    if len(message) > 200:
        message = "%s does not satisfy %r: %r" % (
            _kind(error.instance), error.validator, error.validator_value)
    return message


def validate_instances(schema, documents):
    """Validate each device document against the schema."""
    validator = jsonschema.validators.validator_for(schema)(schema)
    findings = []
    for relpath, _index, document in documents:
        for error in sorted(validator.iter_errors(document),
                            key=lambda e: list(e.absolute_path)):
            findings.append(Finding(
                check="instance",
                locator="%s#%s" % (relpath, _pointer(tuple(error.absolute_path))),
                message=_describe(error),
            ))
    return findings


def find_undeclared_keys(schema, documents):
    """Report instance keys the schema declares nothing about.

    The schema sets ``additionalProperties`` nowhere, so a validator accepts any extra
    key in silence -- a key the schema never declares is carried by every device file
    and checked by nothing. This walk is what a strict schema would have caught.
    """
    findings = []

    def walk(instance, subschema, relpath, path):
        if not isinstance(subschema, dict):
            return
        if isinstance(instance, dict):
            properties = subschema.get("properties", {})
            additional = subschema.get("additionalProperties")
            for key, value in sorted(instance.items()):
                if key in properties:
                    walk(value, properties[key], relpath, path + (key,))
                elif isinstance(additional, dict):
                    walk(value, additional, relpath, path + (key,))
                elif properties or subschema.get("type") == "object":
                    findings.append(Finding(
                        check="undeclared-key",
                        locator="%s#%s" % (relpath, _pointer(path + (key,))),
                        message="key is not declared by the schema, so no constraint "
                                "applies to it",
                    ))
        elif isinstance(instance, list):
            items = subschema.get("items")
            if isinstance(items, dict):
                for i, value in enumerate(instance):
                    walk(value, items, relpath, path + (i,))

    for relpath, _index, document in documents:
        walk(document, schema, relpath, ())
    return findings


def _declared_list(names):
    """Render the declared memory names for an error message, without a wall of text."""
    if not names:
        return "(this file declares no MemoryTypes at all)"
    shown = ", ".join(repr(name) for name in names[:8])
    return shown if len(names) <= 8 else "%s, (+%d more)" % (shown, len(names) - 8)


def check_memory_references(documents):
    """Report ``UnitTypes.*.Memory`` entries that name no memory the file declares.

    Each entry is a *name reference* into ``MemorySubsystem.MemoryTypes``: the schema
    types it as a plain string and says nothing about the target existing, so a
    reference that resolves to nothing is invisible to stock validation. The reference
    and the ``MemoryTypes`` key it points at must match verbatim, in the same file.
    """
    findings = []
    for relpath, index, document in documents:
        if not isinstance(document, dict):
            continue
        # An element of a top-level array is addressed through its index.
        prefix = () if index is None else (index,)

        subsystem = document.get("MemorySubsystem")
        entries = subsystem.get("MemoryTypes") if isinstance(subsystem, dict) else None
        # MemoryTypes is a name-keyed map, so the reference target is a key.
        declared = list(entries) if isinstance(entries, dict) else []
        known = frozenset(declared)

        core = document.get("CoreSubsystem")
        units = core.get("UnitTypes") if isinstance(core, dict) else None
        if not isinstance(units, dict):
            continue

        for name, unit in sorted(units.items()):
            if not isinstance(unit, dict):
                continue
            references = unit.get("Memory")
            if not isinstance(references, list):
                continue  # a wrong shape is the instance check's finding, not this one
            for i, reference in enumerate(references):
                if isinstance(reference, str) and reference in known:
                    continue
                findings.append(Finding(
                    check="memory-ref",
                    locator="%s#%s" % (relpath, _pointer(
                        prefix + ("CoreSubsystem", "UnitTypes", name, "Memory", i))),
                    message="%r matches no MemorySubsystem.MemoryTypes key in this "
                            "file, so the reference resolves to nothing; declared "
                            "here: %s" % (reference, _declared_list(declared)),
                ))
    return findings


_KEY_PUNCT = re.compile(r"[()/~]")


def check_memory_key_shape(documents):
    """Report ``MemoryTypes`` keys that are captions rather than identifiers.

    A key is the memory's short name -- ``HBM3``, ``LDS``, ``Tensix L1`` -- not a
    display string carrying a parenthetical scope gloss. The scope belongs in the
    entry's ``description``, which is where a reader looks for it and where it can be
    stated in a sentence. Three concrete reasons the punctuation is banned rather than
    merely discouraged:

    * ``(`` / ``)`` mark a gloss that duplicates the description and drifts from it.
    * ``/`` and ``~`` force RFC 6901 escaping on every JSON pointer built over the key,
      for a character that carries no meaning a space could not.
    * a key is what ``UnitTypes.*.Memory`` references and what ``Keys`` enumerates, so
      it is an identifier in an API, not a caption in a table.

    Uniqueness needs no check here: duplicate keys cannot survive JSON parsing.
    """
    findings = []
    for relpath, index, document in documents:
        if not isinstance(document, dict):
            continue
        prefix = () if index is None else (index,)
        subsystem = document.get("MemorySubsystem")
        entries = subsystem.get("MemoryTypes") if isinstance(subsystem, dict) else None
        if not isinstance(entries, dict):
            continue
        for name in entries:
            hits = sorted(set(_KEY_PUNCT.findall(name)))
            if not hits:
                continue
            findings.append(Finding(
                check="memory-key",
                locator="%s#%s" % (relpath, _pointer(
                    prefix + ("MemorySubsystem", "MemoryTypes", name))),
                message="%r is a caption, not an identifier: %s has no place in a "
                        "memory key. Drop the scope gloss into 'description' and key "
                        "the entry by the memory's short name"
                        % (name, ", ".join(repr(c) for c in hits)),
            ))
    return findings


_PROP_ROW = re.compile(r"^KNEXUS_API_PROP\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", re.MULTILINE)


def check_prop_table_casing(text):
    """Report property names that differ only by casing.

    ``nxs_property`` names are the runtime's vocabulary for schema keys. Two spellings of
    one concept means two enum values, and a reader that asks for the wrong one gets
    nothing back.
    """
    by_lowercase = {}
    for name in _PROP_ROW.findall(text):
        by_lowercase.setdefault(name.lower(), set()).add(name)

    findings = []
    for _key, spellings in sorted(by_lowercase.items()):
        if len(spellings) > 1:
            names = sorted(spellings)
            findings.append(Finding(
                check="prop-table-casing",
                locator="%s#%s" % (PROP_TABLE_PATH, "~".join(names)),
                message="%d spellings of one property name: %s" % (
                    len(names), ", ".join(names)),
            ))
    return findings


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

Report = namedtuple("Report", "documents inventory findings waived stale")


def load_documents():
    """Load every ``device_lib/*.json`` as ``(relpath, index, document)`` triples.

    A file whose top level is an array contributes one triple per element, so the rest of
    the corpus is still checked while that shape is being repaired.
    """
    directory = os.path.join(REPO_ROOT, DEVICE_LIB_DIR)
    documents = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        relpath = "%s/%s" % (DEVICE_LIB_DIR, name)
        with open(os.path.join(directory, name)) as handle:
            content = json.load(handle)
        if isinstance(content, list):
            # The array itself is reported by the instance check; still inspect the
            # elements so their contents are not skipped.
            documents.append((relpath, None, content))
            for i, element in enumerate(content):
                documents.append((relpath, i, element))
        else:
            documents.append((relpath, None, content))
    return documents


def inventory(documents):
    """Count what the corpus actually holds -- a canary that the checks saw real data."""
    files, units, memories, objects = set(), 0, 0, 0
    for relpath, _index, document in documents:
        files.add(relpath)
        if not isinstance(document, dict):
            continue
        objects += 1
        units += len(document.get("CoreSubsystem", {}).get("UnitTypes", {}) or {})
        memories += len(document.get("MemorySubsystem", {}).get("MemoryTypes", {}) or {})
    return {"files": len(files), "documents": objects,
            "UnitTypes": units, "MemoryTypes": memories}


def _waiver_for(finding):
    for waiver in KNOWN_VIOLATIONS:
        if (fnmatch.fnmatch(finding.check, waiver.check)
                and fnmatch.fnmatch(finding.locator, waiver.locator)):
            return waiver
    return None


def run_checks():
    """Run every check and split its findings into unwaived / waived / stale."""
    with open(os.path.join(REPO_ROOT, SCHEMA_PATH)) as handle:
        schema = json.load(handle)
    documents = load_documents()

    findings = lint_schema(schema)
    if jsonschema is not None:
        validator_cls = jsonschema.validators.validator_for(schema)
        try:
            validator_cls.check_schema(schema)
        except jsonschema.exceptions.SchemaError as error:
            findings.append(Finding(
                check="schema-meta",
                locator="%s#%s" % (SCHEMA_PATH, _pointer(tuple(error.absolute_path))),
                message=_describe(error),
            ))
        findings += validate_instances(schema, documents)
    findings += find_undeclared_keys(schema, documents)
    findings += check_memory_references(documents)
    findings += check_memory_key_shape(documents)

    with open(os.path.join(REPO_ROOT, PROP_TABLE_PATH)) as handle:
        findings += check_prop_table_casing(handle.read())

    unwaived, waived = [], {}
    for finding in findings:
        waiver = _waiver_for(finding)
        if waiver is None:
            unwaived.append(finding)
        else:
            waived.setdefault(waiver, []).append(finding)

    stale = [w for w in KNOWN_VIOLATIONS if w not in waived]
    return Report(documents, inventory(documents), unwaived, waived, stale)


def format_report(report):
    lines = []
    counts = report.inventory
    lines.append("schema:  %s" % SCHEMA_PATH)
    lines.append("corpus:  %s/*.json -- %d files, %d device objects, "
                 "%d UnitTypes entries, %d MemoryTypes entries" % (
                     DEVICE_LIB_DIR, counts["files"], counts["documents"],
                     counts["UnitTypes"], counts["MemoryTypes"]))
    if jsonschema is None:
        lines.append("")
        lines.append("!! jsonschema is not installed -- instance validation was SKIPPED.")
        lines.append("   pip install -r requirements.txt")

    lines.append("")
    if report.waived:
        lines.append("KNOWN_VIOLATIONS (allowlisted, %d entr(ies), %d occurrence(s)):" % (
            len(report.waived), sum(len(v) for v in report.waived.values())))
        for waiver in KNOWN_VIOLATIONS:
            hits = report.waived.get(waiver)
            if not hits:
                continue
            lines.append("  [%s] %s  x%d" % (waiver.check, waiver.locator, len(hits)))
            lines.append("      %s" % waiver.reason)
    else:
        lines.append("KNOWN_VIOLATIONS: empty -- no exemptions in force.")

    lines.append("")
    if report.stale:
        lines.append("STALE allowlist entries -- the violation is gone, delete the entry:")
        for waiver in report.stale:
            lines.append("  [%s] %s" % (waiver.check, waiver.locator))
        lines.append("")

    if report.findings:
        lines.append("VIOLATIONS (%d, not allowlisted):" % len(report.findings))
        for finding in report.findings:
            lines.append("  [%s] %s" % (finding.check, finding.locator))
            lines.append("      %s" % finding.message)
    else:
        lines.append("VIOLATIONS: none outside the allowlist.")

    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    report = run_checks()
    print(format_report(report))
    print("")
    if jsonschema is None:
        print("RESULT: INCOMPLETE (jsonschema missing)")
        return 2
    ok = not report.findings and not report.stale
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


class TestDeviceSchema(unittest.TestCase):
    """Every device_lib file validates, modulo the KNOWN_VIOLATIONS allowlist."""

    def _assert_clean(self, check):
        report = _report()
        offenders = [f for f in report.findings if fnmatch.fnmatch(f.check, check)]
        if offenders:
            self.fail("%d unallowlisted %r violation(s)\n\n%s" % (
                len(offenders), check, format_report(report)))

    def test_corpus_is_not_empty(self):
        """Guard against a green run that validated nothing."""
        counts = _report().inventory
        self.assertGreater(counts["files"], 0, "no device_lib/*.json files were read")
        self.assertGreater(counts["UnitTypes"], 0, "no UnitTypes entries were read")
        self.assertGreater(counts["MemoryTypes"], 0, "no MemoryTypes entries were read")

    @unittest.skipIf(jsonschema is None, "jsonschema module not available")
    def test_schema_is_a_valid_json_schema(self):
        self._assert_clean("schema-meta")

    @unittest.skipIf(jsonschema is None, "jsonschema module not available")
    def test_device_library_validates_against_schema(self):
        self._assert_clean("instance")

    def test_schema_has_no_ignored_keys(self):
        """A key that is not a JSON Schema keyword documents an unenforced constraint."""
        self._assert_clean("schema-lint")

    def test_device_library_has_no_undeclared_keys(self):
        """The schema declares no additionalProperties, so extra keys go unvalidated."""
        self._assert_clean("undeclared-key")

    def test_unit_memory_references_resolve(self):
        """Every UnitTypes.*.Memory entry names a memory the same file declares."""
        self._assert_clean("memory-ref")

    def test_memory_keys_are_identifiers_not_captions(self):
        """No MemoryTypes key carries a parenthetical gloss, a '/' or a '~'."""
        self._assert_clean("memory-key")

    def test_property_table_has_one_spelling_per_name(self):
        self._assert_clean("prop-table-casing")

    def test_allowlist_has_no_stale_entries(self):
        report = _report()
        if report.stale:
            self.fail("%d KNOWN_VIOLATIONS entr(ies) match nothing and must be "
                      "deleted\n\n%s" % (len(report.stale), format_report(report)))


if __name__ == "__main__":
    sys.exit(main())
