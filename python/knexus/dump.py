"""Dump the KNexus object tree as deterministic JSON.

``knexus.dump(obj=None)`` returns one JSON-serializable dict for ``obj``, by default
the System: each Runtime, each Device, and every child collection a getter returns
(buffers, libraries and their kernels, schedules and their commands, streams, events).
Pass a Runtime, Device, Link or any other object to dump that subtree. Each node carries its class
name, its index in the parent, its property keys, every readable property value, and
its children keyed by collection name. A Link node also names its two endpoints as
references (runtime name and device id), since the devices live under their runtimes.

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


def _endpoint(dev):
    """A reference to a link endpoint: where the Device sits in the tree, plus its bus
    id when it has one. ``None`` for an empty Device (a host link with no cpu runtime)."""
    if not dev:
        return None
    ref = {'runtime': dev.get_runtime().get_property_str('Name'), 'device': dev.get_id()}
    try:
        ref['PciBusId'] = dev.get_property_str('PciBusId')
    except Exception:
        pass
    return ref


def to_dict(obj, index=None, info=False, children=True):
    """Return one node for ``obj`` and, unless ``children`` is False, its subtree."""
    cls = _class_name(obj)
    node = {'class': cls}
    if index is not None:
        node['index'] = index
    if hasattr(obj, 'get_property_keys'):
        node['keys'], node['properties'] = _properties(obj)
    if cls == 'Link':
        node['endpoints'] = [_endpoint(dev) for dev in obj.get_devices()]
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


def dump(obj=None, info=False, children=True):
    """Return ``obj``'s node and subtree as a JSON-serializable dict.

    ``obj`` defaults to the System; a Runtime, Device, Link or any other knexus object
    dumps that subtree. ``info`` adds each device's Info tree, ``children=False`` omits
    the child collections.
    """
    return to_dict(_System() if obj is None else obj, info=info, children=children)


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
    obj = None
    if args.runtime is not None:
        obj = knexus.get_runtime(args.runtime)
        if not obj:
            print('knexus.dump: no runtime named ' + repr(args.runtime), file=sys.stderr)
            return 2
    tree = dump(obj, args.info, not args.no_children)
    json.dump(tree, out, indent=2, sort_keys=True)
    out.write('\n')
    out.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
