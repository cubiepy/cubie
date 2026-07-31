"""Merge junit XML reports, optionally keeping only named tests.

Usage::

    python ci/tools/merge_junit.py OUT.xml IN.xml [IN.xml ...] \
        [--only NODES.json]

``--only`` takes a JSON list of pytest node ids; cases outside it are
dropped. Counts in the merged suite are recomputed from the cases kept.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def node_id(case):
    """Rebuild the pytest node id from a junit testcase."""
    parts = case.get("classname", "").split(".")
    module = []
    for part in parts:
        module.append(part)
        if part.startswith("test_") or part.startswith("conftest"):
            break
    rest = parts[len(module):]
    file_path = "/".join(module) + ".py"
    return "::".join([file_path] + rest + [case.get("name", "")])


def suites(path):
    root = ET.parse(path).getroot()
    return list(root) if root.tag == "testsuites" else [root]


def main(argv):
    keep = None
    if "--only" in argv:
        index = argv.index("--only")
        keep = set(json.loads(
            Path(argv[index + 1]).read_text(encoding="utf-8")
        ))
        argv = argv[:index] + argv[index + 2:]
    if len(argv) < 2:
        print(__doc__)
        return 1

    out_path, inputs = argv[0], argv[1:]
    cases = []
    name = "pytest"
    for path in inputs:
        for suite in suites(path):
            name = suite.get("name", name)
            for case in suite.findall("testcase"):
                if keep is None or node_id(case) in keep:
                    cases.append(case)

    counts = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    seconds = 0.0
    for case in cases:
        seconds += float(case.get("time", 0.0))
        for child in case:
            if child.tag == "failure":
                counts["failures"] += 1
            elif child.tag == "error":
                counts["errors"] += 1
            elif child.tag == "skipped":
                counts["skipped"] += 1

    merged = ET.Element("testsuite", {
        "name": name,
        "time": f"{seconds:.3f}",
        **{key: str(value) for key, value in counts.items()},
    })
    merged.extend(cases)
    root = ET.Element("testsuites")
    root.append(merged)
    ET.ElementTree(root).write(
        out_path, encoding="utf-8", xml_declaration=True
    )
    print(f"{out_path}: {counts['tests']} tests, "
          f"{counts['failures']} failures, {counts['errors']} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
