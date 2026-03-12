"""Query Resolution Debugger — Step-by-step pipeline trace visualization.

Reads the pipeline result from the Talk to Data Tool (via raw_data output)
and renders a step-by-step resolution view showing how the user's question
was processed through each pipeline stage.

Wire this component to the Talk to Data Tool's Data output for per-query
trace visualization.
"""

from typing import Optional

from agentcore.custom.custom_node.node import Node
from agentcore.inputs.inputs import HandleInput
from agentcore.schema.data import Data
from agentcore.schema.message import Message
from agentcore.template.field.base import Output
from agentcore.logging import logger


class QueryResolutionViewerComponent(Node):
    """Renders a step-by-step query resolution trace from Talk to Data output."""

    display_name = "Query Resolution Debugger"
    description = (
        "Visualizes the step-by-step query resolution pipeline from a Talk to Data "
        "Tool. Shows normalization, intent classification, schema linking, template "
        "matching, anti-pattern fixes, validation, and execution details."
    )
    icon = "bug"
    name = "QueryResolutionViewer"

    inputs = [
        HandleInput(
            name="pipeline_result",
            display_name="Pipeline Result",
            input_types=["Data"],
            info="Connect the Data output from a Talk to Data Tool or NL-to-SQL component.",
            required=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Debug View",
            name="debug_view",
            method="build_debug_view",
            types=["Message"],
        ),
    ]

    def _get_result(self) -> Optional[dict]:
        pr = getattr(self, "pipeline_result", None)
        if pr is None or pr == "":
            return None
        if isinstance(pr, Data):
            return pr.data if pr.data else None
        if isinstance(pr, dict):
            return pr if pr else None
        return None

    def _confidence_bar(self, confidence: float) -> str:
        """Generate a text-based confidence bar."""
        filled = int(confidence * 10)
        empty = 10 - filled
        return "\u2588" * filled + "\u2591" * empty

    def build_debug_view(self) -> Message:
        result = self._get_result()
        if not result:
            return Message(text="**Query Resolution Debugger:** No pipeline result connected.")

        if result.get("error"):
            return Message(
                text=f"**Query Resolution Debugger:** Pipeline error\n\n{result.get('message', 'Unknown error')}"
            )

        trace = result.get("pipeline_trace", {})
        pipeline_log = result.get("pipeline_log", [])
        user_query = result.get("user_query", "")
        sql = result.get("generated_sql", "")

        parts = []
        parts.append(f'**Query:** "{user_query}"\n')
        parts.append("---\n")

        # Stage 0: Normalization
        norm = trace.get("stage_0_normalizer", {})
        if norm:
            parts.append("**Step 1: Normalization**")
            normalized = norm.get("normalized_query", user_query)
            if normalized != user_query:
                parts.append(f"- Cleaned: `{normalized}`")
            expansions = norm.get("expansions", [])
            for exp in expansions:
                parts.append(f'- Expanded: "{exp["from"]}" \u2192 "{exp["to"]}"')
            alias_res = norm.get("alias_resolutions", [])
            for alias in alias_res:
                parts.append(
                    f'- Alias: "{alias["alias"]}" \u2192 {alias.get("type", "?")}: '
                    f'`{alias.get("sql_filter", alias.get("canonical_value", "?"))}`'
                )
            numbers = norm.get("extracted_numbers", [])
            if numbers:
                parts.append(f"- Extracted numbers: {numbers}")
            parts.append("")

        # Stage 1: Schema Linking
        sl = trace.get("stage_1_schema_linking", {})
        if sl:
            parts.append("**Step 2: Schema Linking**")
            resolved = sl.get("resolved_columns", {})
            for term, col in resolved.items():
                parts.append(f'- "{term}" \u2192 `{col}`')
            entities = sl.get("detected_entities", [])
            if entities:
                parts.append(f"- Entities: {', '.join(entities)}")
            filters = sl.get("suggested_filters", [])
            if filters:
                parts.append(f"- Filters: {', '.join(str(f) for f in filters)}")
            parts.append("")

        # Stage 2: Intent
        intent = trace.get("stage_2_intent", {})
        if intent:
            parts.append("**Step 3: Intent Classification**")
            primary = intent.get("primary_intent", "unknown")
            confidence = intent.get("confidence", 0)
            level = intent.get("confidence_level", "?")
            bar = self._confidence_bar(confidence)
            parts.append(f"- Intent: **{primary}** (confidence: {confidence:.2f}) {bar} {level.upper()}")
            secondary = intent.get("secondary_intents", [])
            if secondary:
                parts.append(f"- Secondary: {', '.join(secondary)}")
            matched = intent.get("matched_phrases", [])
            if matched:
                phrases = [f'"{m["phrase"]}"' for m in matched[:3]]
                parts.append(f"- Matched phrases: {', '.join(phrases)}")
            parts.append("")

        # Stage 3: Examples
        examples = trace.get("stage_3_examples", {})
        if examples:
            parts.append("**Step 4: Example Selection**")
            parts.append(f"- Selected: {examples.get('selected', 0)} of {examples.get('total', 0)} examples")
            parts.append("")

        # Stage 4.25: Template
        template = trace.get("stage_4_25_template", {})
        if template and template.get("matched"):
            parts.append("**Step 5: SQL Generation**")
            parts.append("- Method: **TEMPLATE** (no LLM call)")
            parts.append("")
        elif trace.get("stage_4_sql_gen"):
            parts.append("**Step 5: SQL Generation**")
            parts.append("- Method: **LLM**")
            parts.append("")

        # Stage 4.5: Anti-patterns
        ap_fixes = trace.get("stage_4_5_anti_patterns", [])
        if ap_fixes:
            parts.append("**Step 6: Anti-Pattern Fixes**")
            for fix in ap_fixes:
                parts.append(f"- {fix}")
            parts.append("")

        # Stage 5: Validation
        validation = trace.get("stage_5_validation", {})
        ont_warnings = validation.get("ontology_warnings", [])
        parts.append("**Step 7: Validation**")
        if ont_warnings:
            for w in ont_warnings:
                parts.append(f"- \u26a0 {w}")
        else:
            parts.append("- \u2705 All checks passed")
        parts.append("")

        # Stage 6: Execution
        execution = trace.get("stage_6_execution", {})
        if execution:
            parts.append("**Step 8: Execution**")
            parts.append(f"- Rows: {execution.get('rows', '?')}")
            parts.append(f"- Time: {execution.get('time_ms', '?')}ms")
            parts.append("")

        # Stage 7: Post-validation
        post = trace.get("stage_7_post_validation", {})
        post_warnings = post.get("warnings", [])
        if post_warnings:
            parts.append("**Step 9: Data Quality**")
            for w in post_warnings:
                parts.append(f"- \u26a0 {w}")
            parts.append("")

        # Generated SQL
        if sql:
            parts.append(f"**Generated SQL:**\n```sql\n{sql}\n```")

        # Full pipeline log
        if pipeline_log:
            parts.append("\n<details><summary>Full Pipeline Log</summary>\n")
            for entry in pipeline_log:
                parts.append(f"- {entry}")
            parts.append("\n</details>")

        self.status = f"Traced {len(trace)} stages"
        return Message(text="\n".join(parts))
