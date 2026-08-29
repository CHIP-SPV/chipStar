#!/usr/bin/env python3
"""Assert that build outputs are ordered after a file in the Ninja build graph.

usage: check_build_order.py --ninja NINJA --build-dir DIR --required FILE OUTPUT...

Each OUTPUT (or `;`-joined list of outputs, as $<TARGET_OBJECTS:...> expands
to) must reach FILE through the transitive inputs of its build edge. Explicit,
implicit and order-only inputs are all followed, because CMake's Ninja
generator expresses target-level dependencies as order-only edges through
phony cmake_object_order_depends_target_* nodes, and it is those edges that
decide whether an object compile may start before FILE has been produced.

Exit status is 1 when any output can be built without FILE existing first.
"""

import argparse
import os
import subprocess
import sys


def query_inputs(ninja, build_dir, nodes):
    """Map each node in `nodes` to the inputs of the edge that produces it."""
    result = subprocess.run(
        [ninja, "-t", "query", *nodes],
        cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ninja -t query failed:\n{result.stdout}{result.stderr}")

    inputs = {}
    current = None
    in_inputs = False
    for line in result.stdout.splitlines():
        if line and not line.startswith(" ") and line.endswith(":"):
            current = line[:-1]
            inputs[current] = []
            in_inputs = False
        elif line.startswith("  input:"):
            in_inputs = True
        elif line.startswith("  outputs:"):
            in_inputs = False
        elif in_inputs and line.startswith("    "):
            dep = line.strip()
            for marker in ("|| ", "| "):  # order-only and implicit inputs
                if dep.startswith(marker):
                    dep = dep[len(marker):]
                    break
            inputs[current].append(dep)
    return inputs


def transitive_inputs(ninja, build_dir, root, cache):
    """Every node reachable from `root` through inputs, one query per level."""
    seen = {root}
    frontier = [root]
    while frontier:
        unknown = [n for n in frontier if n not in cache]
        if unknown:
            cache.update(query_inputs(ninja, build_dir, unknown))
        next_frontier = []
        for node in frontier:
            for dep in cache.get(node, []):
                if dep not in seen:
                    seen.add(dep)
                    next_frontier.append(dep)
        frontier = next_frontier
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ninja", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--required", required=True,
                        help="file every OUTPUT must be ordered after")
    parser.add_argument("outputs", nargs="+")
    args = parser.parse_args()

    build_dir = os.path.abspath(args.build_dir)

    def ninja_name(path):
        # CMake writes paths under the build directory relative to it.
        if os.path.isabs(path) and path.startswith(build_dir + os.sep):
            return os.path.relpath(path, build_dir)
        return path

    required = ninja_name(args.required)
    outputs = [ninja_name(o) for arg in args.outputs for o in arg.split(";") if o]

    cache = {}
    missing = []
    for output in outputs:
        closure = {ninja_name(n) for n in
                   transitive_inputs(args.ninja, build_dir, output, cache)}
        ordered = required in closure
        print(f"{output}: {'ordered' if ordered else 'NOT ordered'} after {required}")
        if not ordered:
            missing.append(output)

    if missing:
        print(f"\n{len(missing)} of {len(outputs)} outputs may be built before "
              f"{required} exists", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
