"""Opaquify a BigModel model into a provenance-free "black box" copy.

This dev tool rewrites a source BigModel file so that every human- and
domain-readable label is replaced by an opaque token, and all metadata
that could reveal the model's origin is stripped:

- ``<component>`` names become ``c0, c1, ...`` (document order).
- ``<variable>`` names become ``v0, v1, ...`` within each component.
- Every reference site is rewritten consistently: MathML ``<ci>``,
  ``<connection>``/``<map_variables>``, ``<group>``/``<component_ref>``.
- All RDF annotation blocks, ``cmeta:id`` attributes, XML comments, and
  the model ``name`` are removed or genericised.

Because cubie derives its symbol names as ``<component>_<variable>``, the
resulting system exposes only opaque names (e.g. ``c3_v7``) for states,
observables, and constants. Units and numeric ``initial_value``s are left
untouched so the dynamics are numerically identical to the source.

State ordering is preserved: bigmodelmanip orders states by document
appearance, and this transform never reorders elements, so column ``i`` of
the black-box output corresponds to column ``i`` of the source output.

Usage
-----
    python _make_blackbox.py SOURCE.bigmodel OUTPUT.bigmodel MODEL_NAME
"""

import sys
from pathlib import Path

from lxml import etree


MATHML_NS = "http://www.w3.org/1998/Math/MathML"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
CMETA_NS = "http://www.bigmodel.org/metadata/1.0#"


def _local(tag):
    """Return the local (namespace-stripped) name of an element tag."""
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def opaquify(source_path, output_path, model_name):
    """Rewrite ``source_path`` into a provenance-free copy.

    Parameters
    ----------
    source_path : str or pathlib.Path
        BigModel file to read.
    output_path : str or pathlib.Path
        Destination for the rewritten BigModel file.
    model_name : str
        Opaque model name written to the ``<model>`` element.
    """
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(source_path), parser)
    root = tree.getroot()

    bigmodel_ns = etree.QName(root).namespace
    def c(name):
        return f"{{{bigmodel_ns}}}{name}"

    def m(name):
        return f"{{{MATHML_NS}}}{name}"

    # --- Build rename maps in document order ---
    component_rename = {}
    # variable_rename[component_old_name][variable_old_name] = new_name
    variable_rename = {}

    for comp in root.findall(c("component")):
        old_comp = comp.get("name")
        new_comp = f"c{len(component_rename)}"
        component_rename[old_comp] = new_comp
        var_map = {}
        for var in comp.findall(c("variable")):
            old_var = var.get("name")
            var_map[old_var] = f"v{len(var_map)}"
        variable_rename[old_comp] = var_map

    # --- Apply renames ---
    for comp in root.findall(c("component")):
        old_comp = comp.get("name")
        var_map = variable_rename[old_comp]
        comp.set("name", component_rename[old_comp])

        for var in comp.findall(c("variable")):
            var.set("name", var_map[var.get("name")])

        # MathML variable references live only in <ci> elements, whose
        # text names a variable declared in this same component.
        for ci in comp.iter(m("ci")):
            ref = (ci.text or "").strip()
            if ref not in var_map:
                raise KeyError(
                    f"<ci>{ref!r}</ci> in component {old_comp!r} does not "
                    f"name a declared variable"
                )
            ci.text = var_map[ref]

    # Connections: map_variables reference variables per connected component.
    for conn in root.findall(c("connection")):
        map_comp = conn.find(c("map_components"))
        comp_1 = map_comp.get("component_1")
        comp_2 = map_comp.get("component_2")
        map_comp.set("component_1", component_rename[comp_1])
        map_comp.set("component_2", component_rename[comp_2])
        for mv in conn.findall(c("map_variables")):
            mv.set(
                "variable_1", variable_rename[comp_1][mv.get("variable_1")]
            )
            mv.set(
                "variable_2", variable_rename[comp_2][mv.get("variable_2")]
            )

    # Encapsulation/containment groups reference component names.
    for group in root.findall(c("group")):
        for cref in group.iter(c("component_ref")):
            cref.set("component", component_rename[cref.get("component")])

    # --- Strip provenance / metadata ---
    # RDF annotation blocks.
    for rdf in root.iter(f"{{{RDF_NS}}}RDF"):
        rdf.getparent().remove(rdf)

    # cmeta:id attributes everywhere.
    for el in root.iter():
        for key in list(el.attrib):
            if key.startswith(f"{{{CMETA_NS}}}"):
                del el.attrib[key]

    # XML comments inside the model.
    for comment in list(root.iter(etree.Comment)):
        comment.getparent().remove(comment)

    # Genericise the model name.
    root.set("name", model_name)

    # Serialising only the root element drops any document-level comment
    # header (the original provenance banner).
    xml_bytes = etree.tostring(
        root, xml_declaration=True, encoding="utf-8", pretty_print=False
    )
    Path(output_path).write_bytes(xml_bytes)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(1)
    opaquify(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"wrote {sys.argv[2]}")
