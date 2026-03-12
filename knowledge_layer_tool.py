"""Knowledge Layer Tool — LCToolNode wrapper for Worker Node integration.

Provides knowledge context as a LangChain StructuredTool so Worker Node agents
can query the knowledge layer dynamically — retrieving synonym mappings, entity
info, relevant few-shot examples, or the full context for SQL generation.
"""

import json
from typing import Any, List

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from agentcore.base.langchain_utilities.model import LCToolNode
from agentcore.field_typing import Tool
from agentcore.inputs.inputs import HandleInput, MultilineInput, TableInput
from agentcore.schema.data import Data
from agentcore.logging import logger

from agentcore.components.tools.knowledge_layer import (
    _safe_parse,
    classify_intent,
    select_relevant_examples,
    _build_intent_index,
    KnowledgeLayerComponent,
)


class KnowledgeLayerTool(LCToolNode):
    """Provides knowledge context for NL-to-SQL queries — as a Worker Node tool.

    Connects to the same file inputs as the component version but exposes the
    knowledge as a LangChain StructuredTool.  The Worker agent can request:
    - 'schema_link': resolve NL terms to column names
    - 'get_examples': get relevant few-shot examples for a query
    - 'full_context': get the complete knowledge summary
    """

    display_name = "Knowledge Layer Tool"
    description = (
        "Provides structured knowledge context for NL-to-SQL queries. "
        "Connect knowledge files and wire into a Worker Node's Tools input."
    )
    icon = "brain"
    name = "KnowledgeLayerTool"

    inputs = [
        HandleInput(
            name="knowledge_context",
            display_name="Knowledge Context",
            input_types=["Data"],
            required=True,
            info="Structured knowledge context from a Knowledge Layer component.",
        ),
    ]

    # ----- Pydantic schema for tool arguments -----

    class _ToolSchema(BaseModel):
        query: str = Field(
            ...,
            description="The user's natural language query to analyze against the knowledge context.",
        )
        operation: str = Field(
            default="full_context",
            description=(
                "Operation to perform: "
                "'schema_link' - resolve NL terms to column names; "
                "'get_examples' - get relevant few-shot SQL examples; "
                "'full_context' - get complete knowledge summary for SQL generation."
            ),
        )

    # ----- LCToolNode interface -----

    def run_model(self) -> List[Data]:
        """Standalone execution."""
        return [Data(data={"message": "Use via Worker Node tool."})]

    def build_tool(self) -> Tool:
        """Build the LangChain StructuredTool for Worker Node."""
        return StructuredTool.from_function(
            name="knowledge_layer",
            description=(
                "Use this tool to get knowledge context for SQL generation. "
                "Operations: 'schema_link' resolves natural language terms to actual "
                "database column names; 'get_examples' returns relevant few-shot SQL "
                "examples; 'full_context' returns the complete knowledge summary. "
                "Call with 'schema_link' first to understand column mappings before "
                "generating SQL."
            ),
            func=self._tool_invoke,
            args_schema=self._ToolSchema,
        )

    # ----- Core logic -----

    def _get_knowledge(self) -> dict:
        """Extract knowledge context from the connected Knowledge Layer."""
        kc = self.knowledge_context
        if isinstance(kc, Data):
            return kc.data or {}
        if isinstance(kc, dict):
            return kc
        return {}

    def _tool_invoke(self, query: str, operation: str = "full_context") -> str:
        """Entry point when called by the Worker Node agent."""
        try:
            knowledge = self._get_knowledge()
            if not knowledge:
                return "Error: No knowledge context available. Connect a Knowledge Layer component."

            if operation == "schema_link":
                return self._schema_link(query, knowledge)
            elif operation == "get_examples":
                return self._get_examples(query, knowledge)
            else:
                return self._full_context(query, knowledge)

        except Exception as e:
            raise ToolException(str(e)) from e

    def _schema_link(self, query: str, knowledge: dict) -> str:
        """Resolve NL terms in the query to actual column names."""
        synonym_map = knowledge.get("synonym_map", {})
        entities = knowledge.get("entities", {})
        column_hints = knowledge.get("column_value_hints", {})

        # Tokenize query and match against synonyms
        import re
        query_tokens = re.split(r"\s+", query.lower())
        resolved = {}
        detected_entities = set()

        # Check single tokens and bigrams
        for i, token in enumerate(query_tokens):
            token_clean = token.strip(".,?!'\"")
            # Single token match
            if token_clean in synonym_map:
                info = synonym_map[token_clean]
                resolved[token_clean] = info
                if info.get("entity"):
                    detected_entities.add(info["entity"])
            # Bigram match
            if i + 1 < len(query_tokens):
                bigram = f"{token_clean} {query_tokens[i + 1].strip('.,?!\"')}"
                if bigram in synonym_map:
                    info = synonym_map[bigram]
                    resolved[bigram] = info
                    if info.get("entity"):
                        detected_entities.add(info["entity"])

        # Format output
        parts = ["**Schema Linking Results:**\n"]
        if resolved:
            parts.append("Resolved column mappings:")
            for term, info in resolved.items():
                col = info.get("column", "?")
                entity = info.get("entity", "")
                parts.append(f"  - \"{term}\" → {col}" + (f" (entity: {entity})" if entity else ""))
        else:
            parts.append("No direct synonym matches found. The LLM should use the schema DDL and column descriptions.")

        if detected_entities:
            parts.append(f"\nDetected entities: {', '.join(sorted(detected_entities))}")

        # Add relevant column value hints
        for term, info in resolved.items():
            col = info.get("column", "")
            if col in column_hints:
                hint = column_hints[col]
                examples = hint.get("examples", [])
                if examples:
                    parts.append(f"\nValues for {col}: {', '.join(str(v) for v in examples[:10])}")

        return "\n".join(parts)

    def _get_examples(self, query: str, knowledge: dict) -> str:
        """Get relevant few-shot examples for the query."""
        all_examples = knowledge.get("examples", [])
        intent_index = knowledge.get("intent_index", {})

        # Classify intent
        intent_result = classify_intent(query, intent_index) if intent_index else {
            "primary_intent": "unknown", "secondary_intents": [], "confidence": 0
        }

        # Select relevant examples
        selected = select_relevant_examples(
            intent_result=intent_result,
            detected_entities=[],
            all_examples=all_examples,
            max_examples=10,
        )

        parts = [f"**Relevant Examples** (selected {len(selected)} of {len(all_examples)} total):\n"]
        parts.append(f"Detected intent: {intent_result['primary_intent']} (confidence: {intent_result.get('confidence', 0)})\n")

        for i, ex in enumerate(selected, 1):
            parts.append(f"Example {i}:")
            parts.append(f"  Q: {ex.get('question', '')}")
            parts.append(f"  SQL: {ex.get('sql', '')}\n")

        return "\n".join(parts)

    def _full_context(self, query: str, knowledge: dict) -> str:
        """Get the complete knowledge summary formatted for LLM consumption."""
        parts = ["**Knowledge Context Summary:**\n"]

        # Synonym highlights for this query
        schema_link = self._schema_link(query, knowledge)
        parts.append(schema_link)

        # Business rules - metrics
        rules = knowledge.get("business_rules", {})
        metrics = rules.get("metrics", {})
        if metrics:
            parts.append("\n**KPI Definitions:**")
            for name, sql_expr in list(metrics.items())[:20]:
                parts.append(f"  - {name}: {sql_expr}")

        # Exclusion rules
        exclusions = rules.get("exclusion_rules", [])
        if exclusions:
            parts.append("\n**Exclusion Rules:**")
            for rule in exclusions[:10]:
                parts.append(f"  - {rule}")

        # Oracle syntax
        oracle = rules.get("oracle_syntax", {})
        if oracle:
            parts.append("\n**Oracle SQL Syntax:**")
            for key, val in oracle.items():
                parts.append(f"  - {key}: {val}")

        # Hierarchies
        hierarchies = knowledge.get("hierarchies", {})
        if hierarchies:
            parts.append("\n**Hierarchies (drill-down/roll-up):**")
            for name, info in hierarchies.items():
                levels = " → ".join(l.get("name", l.get("column", "?")) for l in info.get("levels", []))
                parts.append(f"  - {info.get('name', name)}: {levels}")

        # Selected examples
        examples_text = self._get_examples(query, knowledge)
        parts.append(f"\n{examples_text}")

        return "\n".join(parts)
