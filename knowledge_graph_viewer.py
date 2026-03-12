"""Knowledge Graph Viewer — Interactive entity-relationship visualization.

Reads Knowledge Layer output and renders an entity-relationship graph
as a Mermaid diagram + structured text summary. Place this component in
the UI flow connected to the Knowledge Layer's Knowledge Context output.

This is a showcase/demo component — it runs once at flow build time,
not per query.
"""

from typing import Optional

from agentcore.custom.custom_node.node import Node
from agentcore.inputs.inputs import HandleInput
from agentcore.schema.data import Data
from agentcore.schema.message import Message
from agentcore.template.field.base import Output
from agentcore.logging import logger


class KnowledgeGraphViewerComponent(Node):
    """Renders an interactive entity-relationship graph from Knowledge Layer output."""

    display_name = "Knowledge Graph Viewer"
    description = (
        "Visualizes the entity-relationship graph from a Knowledge Layer. "
        "Shows entities, their columns, relationships, and hierarchies as a "
        "Mermaid diagram and structured summary."
    )
    icon = "share-2"
    name = "KnowledgeGraphViewer"

    inputs = [
        HandleInput(
            name="knowledge_context",
            display_name="Knowledge Context",
            input_types=["Data"],
            info="Connect the Knowledge Context output from a Knowledge Layer component.",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Graph View",
            name="graph_view",
            method="build_graph_view",
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

    def build_graph_view(self) -> Message:
        knowledge = self._get_knowledge()
        if not knowledge:
            return Message(text="**Knowledge Graph Viewer:** No knowledge context connected.")

        entities = knowledge.get("entities", {})
        hierarchies = knowledge.get("hierarchies", {})
        valid_combos = knowledge.get("valid_combinations", {})

        if not entities:
            return Message(text="**Knowledge Graph Viewer:** No entities found in knowledge context.")

        parts = []

        # --- Mermaid diagram ---
        parts.append("**Entity-Relationship Graph**\n")
        parts.append("```mermaid")
        parts.append("graph TD")

        # Add entity nodes
        for name, info in entities.items():
            etype = info.get("type", "dimension")
            pk = info.get("primary_key", "?")
            display = info.get("display_column", "?")
            cols = info.get("columns", [])
            col_count = len(cols) if isinstance(cols, list) else 0

            # Style fact entities differently
            if etype == "fact":
                parts.append(f'    {name}[("{name}<br/>PK: {pk}<br/>{col_count} columns")]')
            else:
                parts.append(f'    {name}["{name}<br/>PK: {pk}<br/>Display: {display}"]')

        # Add relationships from entity synonyms/usage rules or hierarchies
        # Infer relationships from shared columns across entities
        added_rels = set()
        for name, info in entities.items():
            cols = set()
            if isinstance(info.get("columns"), list):
                cols = {str(c).upper() for c in info["columns"]}
            elif isinstance(info.get("column_roles"), dict):
                cols = {str(c).upper() for c in info["column_roles"].keys()}

            # Check if this entity's columns appear in other entities (implies relationship)
            for other_name, other_info in entities.items():
                if other_name == name:
                    continue
                other_cols = set()
                if isinstance(other_info.get("columns"), list):
                    other_cols = {str(c).upper() for c in other_info["columns"]}

                shared = cols & other_cols
                if shared and (name, other_name) not in added_rels and (other_name, name) not in added_rels:
                    rel_label = ", ".join(list(shared)[:2])
                    parts.append(f"    {name} -->|{rel_label}| {other_name}")
                    added_rels.add((name, other_name))

        # Add hierarchy relationships
        for h_name, h_info in hierarchies.items():
            levels = h_info.get("levels", [])
            for i in range(len(levels) - 1):
                parent = levels[i].get("name", levels[i].get("column", "?"))
                child = levels[i + 1].get("name", levels[i + 1].get("column", "?"))
                parts.append(f"    {parent} -->|contains| {child}")

        # Style classes
        parts.append("")
        parts.append("    classDef fact fill:#f96,stroke:#333,stroke-width:2px")
        parts.append("    classDef dim fill:#69f,stroke:#333,stroke-width:1px")
        for name, info in entities.items():
            if info.get("type") == "fact":
                parts.append(f"    class {name} fact")
            else:
                parts.append(f"    class {name} dim")

        parts.append("```\n")

        # --- Entity Details Table ---
        parts.append("**Entity Details**\n")
        parts.append("| Entity | Type | Primary Key | Display Column | Columns |")
        parts.append("|--------|------|-------------|----------------|---------|")
        for name, info in entities.items():
            etype = info.get("type", "dimension")
            pk = info.get("primary_key", "—")
            display = info.get("display_column", "—")
            cols = info.get("columns", [])
            col_str = ", ".join(str(c) for c in cols[:5]) if isinstance(cols, list) else "—"
            if isinstance(cols, list) and len(cols) > 5:
                col_str += f" (+{len(cols) - 5} more)"
            parts.append(f"| {name} | {etype} | {pk} | {display} | {col_str} |")

        # --- Hierarchy Summary ---
        if hierarchies:
            parts.append("\n**Hierarchies**\n")
            for h_name, h_info in hierarchies.items():
                levels = h_info.get("levels", [])
                level_str = " → ".join(
                    l.get("name", l.get("column", "?")) for l in levels
                )
                parts.append(f"- **{h_info.get('name', h_name)}:** {level_str}")

        # --- Stats ---
        parts.append(f"\n---\n*{len(entities)} entities, {len(hierarchies)} hierarchies*")

        self.status = f"Rendered {len(entities)} entities"
        return Message(text="\n".join(parts))
