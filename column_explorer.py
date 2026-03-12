"""Column Value Explorer — Column value distribution visualization.

Reads Knowledge Layer output and renders text-based bar charts showing
column value distributions for low/medium cardinality columns. Useful
for understanding data skew and filter selectivity.

This is a showcase/demo component — runs once at flow build time.
"""

from typing import Optional

from agentcore.custom.custom_node.node import Node
from agentcore.inputs.inputs import HandleInput, IntInput
from agentcore.schema.data import Data
from agentcore.schema.message import Message
from agentcore.template.field.base import Output
from agentcore.logging import logger


class ColumnExplorerComponent(Node):
    """Renders column value distributions as text-based bar charts."""

    display_name = "Column Value Explorer"
    description = (
        "Visualizes column value distributions from a Knowledge Layer. "
        "Shows bar charts with frequencies and percentages for low/medium "
        "cardinality columns (TIER_1 and TIER_2)."
    )
    icon = "bar-chart-2"
    name = "ColumnExplorer"

    inputs = [
        HandleInput(
            name="knowledge_context",
            display_name="Knowledge Context",
            input_types=["Data"],
            info="Connect the Knowledge Context output from a Knowledge Layer component.",
            required=True,
        ),
        IntInput(
            name="max_values_per_column",
            display_name="Max Values Per Column",
            value=10,
            info="Maximum number of values to show per column.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Explorer View",
            name="explorer_view",
            method="build_explorer_view",
            types=["Message"],
        ),
    ]

    def _get_knowledge(self) -> Optional[dict]:
        kc = getattr(self, "knowledge_context", None)
        if kc is None or kc == "":
            return None
        if isinstance(kc, Data):
            return kc.data if kc.data else None
        if isinstance(kc, dict):
            return kc if kc else None
        return None

    def _bar(self, pct: float, max_width: int = 30) -> str:
        """Generate a Unicode bar of proportional width."""
        filled = max(1, int(pct / 100 * max_width)) if pct > 0 else 0
        return "\u2588" * filled

    def _tier_label(self, tier: str) -> str:
        """Friendly label for tier."""
        labels = {
            "TIER_1_EXCELLENT": "excellent filter",
            "TIER_2_GOOD": "good filter",
            "TIER_3_MODERATE": "moderate filter",
            "TIER_4_MODERATE_HIGH": "many values",
            "TIER_5_HIGH_CARDINALITY": "high cardinality",
        }
        return labels.get(tier, tier)

    def build_explorer_view(self) -> Message:
        knowledge = self._get_knowledge()
        if not knowledge:
            return Message(text="**Column Value Explorer:** No knowledge context connected.")

        cvd = knowledge.get("column_values_detailed", {})
        if not cvd:
            return Message(
                text="**Column Value Explorer:** No column value data found. "
                "Ensure the Knowledge Layer has a column_values file loaded."
            )

        max_vals = getattr(self, "max_values_per_column", 10) or 10

        # Only show TIER_1 and TIER_2 columns (low cardinality, meaningful)
        show_tiers = {"TIER_1_EXCELLENT", "TIER_2_GOOD", "TIER_3_MODERATE"}

        parts = ["**Column Value Explorer**\n"]
        columns_shown = 0

        for col_name, col_info in cvd.items():
            tier = col_info.get("tier", "UNKNOWN")
            if tier not in show_tiers:
                continue

            cardinality = col_info.get("cardinality", 0)
            values = col_info.get("values", [])
            if not values:
                continue

            columns_shown += 1
            tier_label = self._tier_label(tier)
            parts.append(f"**{col_name}** ({cardinality} unique) \\[{tier_label}\\]\n")
            parts.append("```")

            # Show top N values
            for v in values[:max_vals]:
                val = str(v.get("value", "?"))
                pct = v.get("pct", 0)
                freq = v.get("frequency", 0)
                bar = self._bar(pct)

                # Truncate long values
                if len(val) > 25:
                    val = val[:22] + "..."

                # Format: bar value (pct%)
                parts.append(f"  {bar:<30s} {val:<25s} ({pct:.1f}%)")

            remaining = len(values) - max_vals
            if remaining > 0:
                parts.append(f"  ... and {remaining} more values")

            parts.append("```\n")

        if columns_shown == 0:
            parts.append("No low/medium cardinality columns found to display.")

        # Summary
        parts.append(f"---\n*Showing {columns_shown} columns from {len(cvd)} total*")

        self.status = f"Displayed {columns_shown} columns"
        return Message(text="\n".join(parts))
