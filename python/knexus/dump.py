"""Dump the KNexus object tree as deterministic JSON.

``system()`` returns one JSON-serializable dict for the System: each Runtime, each
Device, and every child collection a getter returns (buffers, libraries and their
kernels, schedules and their commands, streams, events). Each node carries its class
name, its index in the parent, its property keys, every readable property value, and
its children keyed by collection name.

The walker only reads: it never creates a buffer, schedule, stream or event.

Run ``python -m knexus.dump [--runtime NAME] [--info] [--no-children]``. Two runs on
the same host print byte-identical output, so ``diff`` compares two dumps.
"""

import argparse
import json
import os
import sys

import knexus
from knexus import property as _prop

# Typed getter per property type, as reported by ``_type``.
# ``flt_vec`` has no binding yet, and ``obj_vec`` is never read as a value (its
# content is the device's ``Info`` tree): both are listed under ``keys`` only.
_GETTERS = {
    'int': 'get_property_int',
    'flt': 'get_property_flt',
    'str': 'get_property_str',
    'int_vec': 'get_property_int_vec',
    'flt_vec': 'get_property_flt_vec',
    'str_vec': 'get_property_str_vec',
}

# Child collections per class, in output order: (collection name, getter name).
# A getter the binding does not have is skipped, so a collection added later
# (e.g. the System's links) appears once its binding exists.
_CHILDREN = {
    'System': [('runtimes', 'get_runtimes'), ('links', 'get_links')],
    'Runtime': [('devices', 'get_devices')],
    'Device': [('buffers', 'get_buffers'), ('libraries', 'get_libraries'),
               ('schedules', 'get_schedules'), ('streams', 'get_streams'),
               ('events', 'get_events')],
    'Library': [('kernels', 'get_kernels')],
    'Schedule': [('commands', 'get_commands')],
}


class _System:
    """The System as a node: the Python API exposes it as module-level getters."""

    def __getattr__(self, name):
        if name.startswith('get_') and hasattr(knexus, name):
            return getattr(knexus, name)
        raise AttributeError(name)


def _class_name(obj):
    return 'System' if isinstance(obj, _System) else type(obj).__name__


def _type(obj, key):
    """The property's type: the object's own answer where it has one (a Link states
    the Link* names as scalars, though they are declared as the fabric table's
    vectors), else the declared type."""
    own = getattr(obj, 'get_property_type', None)
    if own is not None:
        try:
            return own(key)
        except Exception:
            return ''
    return _prop.get_type(int(key))


def _properties(obj):
    """Return (keys, values): every listed key, and each value its typed getter reads."""
    try:
        keys = list(obj.get_property_keys())
    except Exception:
        return [], {}
    names, values = [], {}
    for key in keys:
        name = _prop.get_name(int(key))
        names.append(name)
        getter = getattr(obj, _GETTERS.get(_type(obj, key), ''), None)
        if getter is None:
            continue
        try:
            values[name] = getter(key)
        except Exception:
            pass  # a property that raises is omitted, never guessed
    return names, values


def _info(obj):
    get_info = getattr(obj, 'get_info', None)
    if get_info is None:
        return None
    try:
        info = get_info()
        return info.get() if info else None
    except Exception:
        return None


def to_dict(obj, index=None, info=False, children=True):
    """Return one node for ``obj`` and, unless ``children`` is False, its subtree."""
    cls = _class_name(obj)
    node = {'class': cls}
    if index is not None:
        node['index'] = index
    if hasattr(obj, 'get_property_keys'):
        node['keys'], node['properties'] = _properties(obj)
    if info:
        tree = _info(obj)
        if tree is not None:
            node['info'] = tree
    if children:
        kids = {}
        for coll, getter in _CHILDREN.get(cls, []):
            get = getattr(obj, getter, None)
            if get is None:
                continue
            try:
                items = list(get())
            except Exception:
                continue
            kids[coll] = [to_dict(item, i, info, children) for i, item in enumerate(items)]
        if kids:
            node['children'] = kids
    return node


def system(runtime=None, info=False, children=True):
    """Return the whole tree as a dict; ``runtime`` limits it to one runtime by name."""
    if runtime is None:
        return to_dict(_System(), info=info, children=children)
    for i, rt in enumerate(knexus.get_runtimes()):
        if rt.get_property_str('Name') == runtime:
            return {'class': 'System', 'children': {'runtimes': [to_dict(rt, i, info, children)]}}
    raise ValueError('no runtime named ' + repr(runtime))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m knexus.dump', description=__doc__.splitlines()[0])
    parser.add_argument('--runtime', metavar='NAME', help='dump only this runtime')
    parser.add_argument('--info', action='store_true', help="also dump each object's Info tree (the device DB)")
    parser.add_argument('--no-children', action='store_true', help='omit child collections (with --runtime, the runtime is still shown)')
    args = parser.parse_args(argv)

    # Runtimes (tt-metal) log to stdout: keep stdout for the JSON, send the rest to stderr.
    sys.stdout.flush()
    out = os.fdopen(os.dup(1), 'w')
    os.dup2(2, 1)
    try:
        tree = system(args.runtime, args.info, not args.no_children)
    except ValueError as e:
        print('knexus.dump: ' + str(e), file=sys.stderr)
        return 2
    json.dump(tree, out, indent=2, sort_keys=True)
    out.write('\n')
    out.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
