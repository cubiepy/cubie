"""Bind completed implicit references and links to the generic direct runner."""

import argparse
import json
from pathlib import Path

from capture import asset, checked, read, require


ROOT = Path(__file__).parent


def run(capture_path, link_path, output):
    stage = read(ROOT / "stage_author_receipt.json")
    for value in stage["sources"]:
        checked(value)
    plan = read(checked(stage["capture_plan"]))
    capture = read(capture_path)
    links = read(link_path)
    require(
        capture["status"]
        == "IMPLICIT_CONTRACT_FALSE_OWN_REFERENCES_AND_IR_COMPLETE",
        "Incomplete source references",
    )
    require(
        links["status"] == "IMPLICIT_SINGLETON_LINK_PAIR_COMPLETE",
        "Incomplete native link pair",
    )
    require(
        capture["plan"] == stage["capture_plan"] == links["plan"],
        "Source plan differs",
    )
    require(
        links["capture"] == asset(capture_path),
        "Native links use different source capture",
    )
    require(
        [x["case"] for x in capture["records"]] == plan["cases"],
        "Reference case coverage",
    )
    require(
        [(x["case"], x["fma"]) for x in links["records"]]
        == [(case, fma) for case in plan["cases"] for fma in (True, False)],
        "Native image case coverage",
    )
    manifest = dict(
        runner=asset(ROOT / "direct_functional.py"),
        wrapper=plan["wrapper"],
        prior_author=plan["previous_author"],
        prepared=plan["prepared"],
        capture=asset(capture_path),
        links=asset(link_path),
        functional=asset(
            ROOT.parent
            / "controlled_carveout_author_e5/controlled_carveout.py"
        ),
        naming_provider=next(
            x
            for x in plan["bindings"]
            if x["path"]
            .replace("\\", "/")
            .endswith("numba_cuda_mlir/mlir_lowering.py")
        ),
        cases=plan["cases"],
        workloads=plan["workloads"],
        stage=asset(ROOT / "stage_author_receipt.json"),
        scope="Four implicit actions with independent own references and exact native image pairs.",
    )
    bindings = {x["path"]: x for x in plan["bindings"]}

    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                item = dict(path=value["path"], sha256=value["sha256"])
                checked(item)
                bindings[item["path"]] = item
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for value in (manifest, stage, capture, links):
        visit(value)
    manifest["bindings"] = list(bindings.values())
    output = Path(output)
    require(not output.exists(), "Refusing to replace a frozen manifest")
    output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(manifest=asset(output), bindings=len(bindings))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--links", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.capture, args.links, args.output)
