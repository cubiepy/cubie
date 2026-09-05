"""Keep typed values when source copies cross materialization forms."""


class PromotedCellValues:
    """Track the exact typed version held by each promoted scalar cell."""

    def memory(self, node):
        if node["cell"][0] not in self.promoted:
            return super().memory(node)
        if not hasattr(self, "promoted_cell_values"):
            self.promoted_cell_values = {}
            self.promoted_copy_bindings = []
        key = tuple(node["cell"])
        source_value = node["inputs"][0]
        if node["kind"] == "element_write_alias":
            value = self.mapped(source_value, node)
            self.promoted_cell_values[key] = value
        elif node["kind"] == "element_read_alias":
            if key not in self.promoted_cell_values:
                self.promoted_cell_values[key] = self.mapped(
                    source_value, node
                )
            value = self.promoted_cell_values[key]
            self.read_values[node["id"]] = value
        else:
            raise ValueError("Unknown promoted cell operation")
        if self.values[value]["dtype"] != node["cell"][3]:
            raise ValueError("Promoted cell copy changes typed width")
        self.promoted_copy_bindings.append(
            dict(
                source_node=node["id"],
                source_value=source_value,
                cell=node["cell"],
                typed_value=value,
                operation=node["kind"],
            )
        )
        return super().memory(node)

    def build(self):
        result = super().build()
        result["promoted_copy_bindings"] = getattr(
            self, "promoted_copy_bindings", []
        )
        return result
