"""Talk to Data (NL → SQL) component for the Agent Builder.

Takes a natural language question, generates SQL using an LLM with schema context,
executes the query against the database, and returns formatted results.

Enhanced with 7-stage pipeline:
  Stage 1: Schema Linking (LLM) — resolve NL terms to columns
  Stage 2: Intent Classification (deterministic) — detect query type
  Stage 3: Dynamic Example Selection (deterministic) — pick relevant few-shots
  Stage 4: SQL Generation (LLM) — generate SQL with full context
  Stage 5: Pre-Execution Validation (deterministic) — safety + ontology checks
  Stage 6: SQL Execution — run query against database
  Stage 7: Post-Result Validation (deterministic) — sanity checks on results
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import sqlparse

from agentcore.custom.custom_node.node import Node
from agentcore.inputs.inputs import (
    BoolInput,
    DropdownInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
    TableInput,
)
from agentcore.schema.data import Data
from agentcore.schema.message import Message
from agentcore.template.field.base import Output
from agentcore.logging import logger


# SQL safety: Only these statement types are allowed
_ALLOWED_SQL_TYPES = {"SELECT"}
_BLOCKED_KEYWORDS = {
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE",
    "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL", "MERGE",
    "RENAME", "REPLACE", "PURGE", "FLASHBACK", "COMMENT",
    "ANALYZE", "AUDIT", "NOAUDIT", "LOCK", "SHUTDOWN",
}
_ORACLE_BLOCKED_KEYWORDS = {
    "EXECUTE IMMEDIATE", "DBMS_", "UTL_", "ALTER SESSION",
    "ALTER SYSTEM", "CREATE TABLE", "CREATE VIEW", "CREATE INDEX",
    "DROP TABLE", "DROP VIEW", "DROP INDEX", "TRUNCATE TABLE",
    "FOR UPDATE", "RETURNING INTO",
}

# Mandatory date filter for Oracle to avoid full-table scans on 200M+ rows
_ORACLE_MANDATORY_DATE_FILTER = "INVOICE_DATE > DATE '2024-04-01'"


def _validate_sql(sql: str, provider: str = "postgresql") -> Tuple[bool, str]:
    """Validate that SQL is a safe read-only SELECT query."""
    sql_stripped = sql.strip().rstrip(";")
    if not sql_stripped:
        return False, "Empty SQL query"

    try:
        parsed = sqlparse.parse(sql_stripped)
        if not parsed:
            return False, "Failed to parse SQL"

        for statement in parsed:
            stmt_type = statement.get_type()
            if stmt_type and stmt_type.upper() not in _ALLOWED_SQL_TYPES:
                return False, f"Only SELECT queries are allowed. Got: {stmt_type}"
    except Exception as e:
        return False, f"SQL parse error: {e!s}"

    # Keyword check as extra safety layer
    sql_upper = sql_stripped.upper()
    tokens = sql_upper.split()
    for kw in _BLOCKED_KEYWORDS:
        if kw in tokens:
            return False, f"Blocked keyword detected: {kw}"

    # Oracle-specific blocked keywords (multi-word)
    if provider == "oracle":
        for kw in _ORACLE_BLOCKED_KEYWORDS:
            if kw in sql_upper:
                return False, f"Blocked Oracle keyword detected: {kw}"

    # Must start with SELECT (after optional WITH for CTEs)
    first_keyword = sql_upper.lstrip().split()[0] if sql_upper.strip() else ""
    if first_keyword not in ("SELECT", "WITH"):
        return False, f"Query must start with SELECT or WITH. Got: {first_keyword}"

    # Block multiple statements (semicolon injection)
    if ";" in sql_stripped:
        return False, "Multiple statements not allowed (semicolon found in query body)"

    return True, "OK"


def _post_process_sql(sql: str, provider: str = "postgresql", max_rows: int = 100) -> Tuple[str, List[str]]:
    """Hardcoded post-processing fixes applied to every generated SQL.

    These run independently of customer anti-pattern files.
    ``max_rows`` comes from the UI-configurable "Max Result Rows" input.
    Returns (fixed_sql, list_of_fixes).
    """
    fixes = []

    # 1. Cap absurd FETCH FIRST values — use the UI-configured max_rows
    fetch_match = re.search(r"FETCH\s+FIRST\s+(\d+)\s+ROWS?\s+ONLY", sql, re.IGNORECASE)
    if fetch_match:
        n = int(fetch_match.group(1))
        if n > max_rows:
            sql = re.sub(
                r"FETCH\s+FIRST\s+\d+\s+ROWS?\s+ONLY",
                f"FETCH FIRST {max_rows} ROWS ONLY",
                sql, flags=re.IGNORECASE,
            )
            fixes.append(f"Capped FETCH FIRST {n} → {max_rows} (configured max rows)")

    # 2. Cap absurd LIMIT values — use the UI-configured max_rows
    limit_match = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
    if limit_match:
        n = int(limit_match.group(1))
        if n > max_rows:
            sql = re.sub(r"\bLIMIT\s+\d+", f"LIMIT {max_rows}", sql, flags=re.IGNORECASE)
            fixes.append(f"Capped LIMIT {n} → {max_rows} (configured max rows)")

    # 3. Remove redundant TO_CHAR fiscal year filters when mandatory date filter exists
    # Pattern: AND TO_CHAR(INVOICE_DATE, 'YYYY') = TO_CHAR(SYSDATE, 'YYYY')
    fiscal_pattern = re.compile(
        r"\s*AND\s+TO_CHAR\s*\(\s*INVOICE_DATE\s*,\s*'YYYY'\s*\)\s*=\s*TO_CHAR\s*\(\s*SYSDATE\s*,\s*'YYYY'\s*\)",
        re.IGNORECASE,
    )
    if fiscal_pattern.search(sql):
        sql = fiscal_pattern.sub("", sql)
        fixes.append("Removed redundant TO_CHAR(INVOICE_DATE,'YYYY')=TO_CHAR(SYSDATE,'YYYY') — mandatory date filter already constrains timeframe")

    # 4. Strip trailing semicolons (Oracle driver chokes on them)
    if sql.rstrip().endswith(";"):
        sql = sql.rstrip().rstrip(";").rstrip()
        fixes.append("Removed trailing semicolon")

    return sql, fixes


def _enforce_oracle_date_filter(sql: str) -> Tuple[str, bool]:
    """Ensure Oracle queries always have the mandatory INVOICE_DATE filter.

    Returns (possibly_modified_sql, was_injected).
    If the query already has an INVOICE_DATE filter, returns it unchanged.
    Otherwise, injects the mandatory filter.
    """
    sql_upper = sql.upper()

    # Check if any INVOICE_DATE filter already exists
    if "INVOICE_DATE" in sql_upper:
        return sql, False

    # Inject the mandatory filter
    # Strategy: add to existing WHERE clause, or insert a new one before GROUP BY / ORDER BY / FETCH
    if "WHERE" in sql_upper:
        # Find the WHERE keyword and append our filter
        where_idx = sql_upper.index("WHERE")
        # Insert after "WHERE "
        insert_point = where_idx + len("WHERE")
        sql = sql[:insert_point] + f" {_ORACLE_MANDATORY_DATE_FILTER} AND" + sql[insert_point:]
    else:
        # No WHERE clause — insert before GROUP BY, ORDER BY, HAVING, FETCH, or at end
        insert_before = None
        for keyword in ("GROUP BY", "ORDER BY", "HAVING", "FETCH FIRST"):
            idx = sql_upper.find(keyword)
            if idx != -1:
                if insert_before is None or idx < insert_before:
                    insert_before = idx

        if insert_before is not None:
            sql = sql[:insert_before] + f"WHERE {_ORACLE_MANDATORY_DATE_FILTER}\n" + sql[insert_before:]
        else:
            # Append at the very end
            sql = sql.rstrip() + f"\nWHERE {_ORACLE_MANDATORY_DATE_FILTER}"

    return sql, True


def _format_results_as_markdown(columns: List[str], rows: List[tuple], max_display: int = 50) -> str:
    """Format query results as a markdown table."""
    if not rows:
        return "_No results found._"

    display_rows = rows[:max_display]

    # Build header
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    # Build rows
    row_lines = []
    for row in display_rows:
        cells = []
        for val in row:
            if val is None:
                cells.append("NULL")
            elif isinstance(val, float):
                cells.append(f"{val:,.2f}")
            elif isinstance(val, int) and abs(val) > 999:
                cells.append(f"{val:,}")
            else:
                cells.append(str(val)[:100])  # Truncate long values
        row_lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join([header, separator] + row_lines)

    if len(rows) > max_display:
        table += f"\n\n_...showing {max_display} of {len(rows)} total rows._"

    return table


class NLtoSQLComponent(Node):
    """Talk to Data: Converts natural language questions to SQL queries.

    Uses an LLM to understand the user's question in the context of the database schema,
    generates a safe SQL query, executes it, and returns formatted results.
    """

    display_name = "Talk to Data (NL→SQL)"
    description = "Convert natural language questions to SQL queries, execute them, and return results."
    icon = "message-square-code"
    name = "NLtoSQL"
    hidden = True

    inputs = [
        HandleInput(
            name="llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            required=True,
            info="The LLM that will generate SQL queries from natural language.",
        ),
        HandleInput(
            name="db_connection",
            display_name="Database Connection",
            input_types=["Data"],
            required=True,
            info="Database connection config from a Database Connector component.",
        ),
        HandleInput(
            name="knowledge_context",
            display_name="Knowledge Context",
            input_types=["Data"],
            info="Structured knowledge context from a Knowledge Layer component. Enables multi-stage pipeline with schema linking, intent classification, and dynamic example selection.",
            required=False,
        ),
        MessageTextInput(
            name="user_query",
            display_name="User Question",
            required=True,
            info="The natural language question to answer using database data.",
        ),
        IntInput(
            name="max_rows",
            display_name="Max Result Rows",
            value=100,
            info="Maximum number of rows to return from the query.",
            advanced=True,
        ),
        BoolInput(
            name="include_sql",
            display_name="Show Generated SQL",
            value=True,
            info="Include the generated SQL query in the response.",
            advanced=True,
        ),
        DropdownInput(
            name="sql_dialect",
            display_name="SQL Dialect",
            options=["auto", "postgresql", "oracle", "sqlserver"],
            value="auto",
            info="Auto-detected from database connector. Controls SQL syntax rules in the prompt.",
            advanced=True,
        ),
        IntInput(
            name="num_examples",
            display_name="Few-Shot Examples Count",
            value=10,
            info="Max number of dynamically selected few-shot examples to include in prompt (from knowledge context). 0 = all.",
            advanced=True,
        ),
        BoolInput(
            name="enable_schema_linking",
            display_name="Enable Schema Linking",
            value=True,
            info="Pre-process query to resolve entity/column references using knowledge graph (extra LLM call).",
            advanced=True,
        ),
        BoolInput(
            name="enable_intent_classification",
            display_name="Enable Intent Classification",
            value=True,
            info="Classify query intent using context graph patterns before SQL generation.",
            advanced=True,
        ),
        BoolInput(
            name="enable_ontology_validation",
            display_name="Enable Ontology Validation",
            value=True,
            info="Validate generated SQL against ontology rules (valid combinations, constraints).",
            advanced=True,
        ),
        BoolInput(
            name="enable_query_normalization",
            display_name="Enable Query Normalization",
            value=True,
            info="Pre-process query to expand abbreviations, resolve aliases, remove filler words.",
            advanced=True,
        ),
        BoolInput(
            name="enable_anti_pattern_check",
            display_name="Enable Anti-Pattern Check",
            value=True,
            info="Auto-detect and fix common SQL anti-patterns (LIMIT→FETCH FIRST, semicolons, etc.).",
            advanced=True,
        ),
        BoolInput(
            name="enable_template_matching",
            display_name="Enable Template Matching",
            value=True,
            info="For high-confidence intents, try to fill a SQL template before calling the LLM.",
            advanced=True,
        ),
        TableInput(
            name="few_shot_examples",
            display_name="Example Q&A Pairs",
            info="Provide example question-to-SQL pairs to improve accuracy. Overridden by Knowledge Layer examples if connected.",
            table_schema=[
                {
                    "name": "question",
                    "display_name": "Question",
                    "type": "str",
                    "description": "Example natural language question",
                },
                {
                    "name": "sql",
                    "display_name": "SQL Query",
                    "type": "str",
                    "description": "The correct SQL for this question",
                },
            ],
            value=[],
            advanced=True,
        ),
        MultilineInput(
            name="additional_context",
            display_name="Domain Context",
            value="",
            info="Additional context about the data domain to help the LLM generate better SQL.",
            advanced=True,
        ),
        TableInput(
            name="table_relationships",
            display_name="Table Relationships",
            info="Define foreign key relationships between tables to help the LLM generate correct JOINs.",
            table_schema=[
                {
                    "name": "source_table",
                    "display_name": "Source Table",
                    "type": "str",
                    "description": "The table containing the foreign key column",
                },
                {
                    "name": "source_column",
                    "display_name": "Source Column",
                    "type": "str",
                    "description": "The FK column in the source table",
                },
                {
                    "name": "target_table",
                    "display_name": "Target Table",
                    "type": "str",
                    "description": "The referenced (parent) table",
                },
                {
                    "name": "target_column",
                    "display_name": "Target Column",
                    "type": "str",
                    "description": "The referenced column (usually the primary key)",
                },
            ],
            value=[],
            advanced=True,
        ),
        TableInput(
            name="column_descriptions",
            display_name="Column Descriptions",
            info="Add business-friendly descriptions for columns to help the LLM understand domain semantics.",
            table_schema=[
                {
                    "name": "table_name",
                    "display_name": "Table",
                    "type": "str",
                    "description": "Table name",
                },
                {
                    "name": "column_name",
                    "display_name": "Column",
                    "type": "str",
                    "description": "Column name",
                },
                {
                    "name": "description",
                    "display_name": "Description",
                    "type": "str",
                    "description": "Business-friendly description of what this column represents",
                },
            ],
            value=[],
            advanced=True,
        ),
        MultilineInput(
            name="business_rules",
            display_name="Business Rules",
            value="",
            info="Business rules the LLM should follow when generating SQL (e.g., 'status=active means not deleted', 'revenue = amount - discount - refund').",
            advanced=True,
        ),
        IntInput(
            name="query_timeout",
            display_name="Query Timeout (seconds)",
            value=30,
            info="Maximum time to wait for query execution.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Result",
            name="result",
            method="run_query",
            types=["Message"],
        ),
        Output(
            display_name="Raw Data",
            name="raw_data",
            method="run_query_raw",
            types=["Data"],
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached_result = None

    def _get_provider(self, db_config: dict) -> str:
        """Detect SQL dialect from db_config or manual override."""
        if self.sql_dialect and self.sql_dialect != "auto":
            return self.sql_dialect
        return db_config.get("provider", "postgresql")

    def _get_knowledge(self) -> Optional[dict]:
        """Extract knowledge context from the connected Knowledge Layer."""
        kc = getattr(self, "knowledge_context", None)
        if kc is None:
            return None
        if isinstance(kc, Data):
            return kc.data if kc.data else None
        if isinstance(kc, dict):
            return kc if kc else None
        return None

    def _get_dialect_rules(self, provider: str) -> str:
        """Get SQL dialect-specific rules."""
        if provider == "oracle":
            return """**Oracle SQL Rules:**
1. Use FETCH FIRST {max_rows} ROWS ONLY to limit results (NEVER use LIMIT)
2. Use SYSDATE for the current date
3. Use TRUNC(SYSDATE, 'YEAR') for the start of the current year
4. Use TO_CHAR(date_col, 'YYYY') to extract year from date
5. Use ADD_MONTHS(date, n) for date arithmetic
6. Use NVL(col, default) for null handling
7. String comparison is case-sensitive — use UPPER() for case-insensitive matching
8. Column names with spaces or mixed case MUST be quoted with double quotes (e.g., "Material Group")
9. Use TO_DATE('value', 'YYYY-MM-DD') for date literals
10. DUAL table is available for single-row queries (SELECT 1 FROM DUAL)
11. **MANDATORY: EVERY query MUST include a date filter: INVOICE_DATE > DATE '2024-04-01'** — the table has 200M+ rows, queries without this filter will timeout. Add this to the WHERE clause ALWAYS, even if the user does not mention a date.""".replace(
                "{max_rows}", str(self.max_rows)
            )
        elif provider == "postgresql":
            return f"""**PostgreSQL SQL Rules:**
1. Use LIMIT {self.max_rows} to cap results
2. Use NOW() for the current timestamp
3. Use DATE_TRUNC('year', col) for date truncation
4. Use EXTRACT(YEAR FROM col) for year extraction
5. Use COALESCE(col, default) for null handling"""
        return f"Use LIMIT {self.max_rows} to cap results."

    def _build_schema_linking_prompt(self, user_query: str, knowledge: dict) -> str:
        """Build prompt for Stage 1: Schema Linking."""
        synonym_map = knowledge.get("synonym_map", {})
        entities = knowledge.get("entities", {})
        column_hints = knowledge.get("column_value_hints", {})

        # Format synonym map (only first 100 to keep prompt manageable)
        syn_lines = []
        for term, info in list(synonym_map.items())[:100]:
            col = info.get("column", "?")
            syn_lines.append(f'  "{term}" → {col}')
        syn_text = "\n".join(syn_lines) if syn_lines else "  (none)"

        # Format entities
        ent_lines = []
        for name, info in entities.items():
            pk = info.get("primary_key", "?")
            display = info.get("display_column", "?")
            ent_lines.append(f"  {name}: PK={pk}, Display={display}, Type={info.get('type', '?')}")
        ent_text = "\n".join(ent_lines) if ent_lines else "  (none)"

        # Format column value hints
        hint_lines = []
        for col, hint in column_hints.items():
            examples = hint.get("examples", [])
            if examples:
                hint_lines.append(f"  {col}: {', '.join(str(v) for v in examples[:8])}")
        hint_text = "\n".join(hint_lines) if hint_lines else "  (none)"

        return f"""You are a schema linking agent. Given a user query, resolve natural language terms to actual database column names.

SYNONYM MAP (natural language term → column name):
{syn_text}

ENTITY DEFINITIONS:
{ent_text}

COLUMN VALUE EXAMPLES (for WHERE clauses):
{hint_text}

User query: "{user_query}"

Respond with a JSON object containing:
- "resolved_columns": dict of user term → column name mapping
- "detected_entities": list of entity names involved
- "suggested_groupby": list of columns to GROUP BY
- "suggested_filters": list of "COLUMN = 'value'" conditions
- "suggested_orderby": ORDER BY clause suggestion (or null)
- "suggested_limit": integer limit (or null)

Return ONLY the JSON, no explanations."""

    def _parse_schema_linking(self, llm_response: str) -> dict:
        """Parse the schema linking LLM response."""
        text = llm_response.strip()
        # Remove markdown fences
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"NL-to-SQL: Failed to parse schema linking response: {text[:200]}")
            return {}

    def _build_sql_generation_prompt(
        self,
        schema_ddl: str,
        user_query: str,
        db_config: Optional[dict] = None,
        *,
        schema_linking_result: Optional[dict] = None,
        intent_result: Optional[dict] = None,
        selected_examples: Optional[List[dict]] = None,
        knowledge: Optional[dict] = None,
    ) -> str:
        """Build the prompt for SQL generation (Stage 4).

        When knowledge context is available, produces a richer prompt with
        resolved column mappings, intent hints, synonym reference, KPI
        definitions, column value hints, and dynamically selected examples.
        """
        provider = self._get_provider(db_config or {})
        sections = []

        # --- System role ---
        sections.append(
            f"You are an expert SQL analyst for {provider.upper()} databases. "
            "Given a database schema and a natural language question, "
            "generate a precise SQL query to answer the question."
        )

        # --- Schema DDL ---
        sections.append(f"\n**Database Schema:**\n```sql\n{schema_ddl}\n```")

        # --- Schema Linking Output (Stage 1 result) ---
        if schema_linking_result:
            resolved = schema_linking_result.get("resolved_columns", {})
            if resolved:
                lines = [f'  "{term}" → {col}' for term, col in resolved.items()]
                sections.append(
                    "\n**Resolved Column Mappings (from schema linking):**\n"
                    "The user's query has been analyzed. Here are the resolved references:\n"
                    + "\n".join(lines)
                )
            suggested_filters = schema_linking_result.get("suggested_filters", [])
            if suggested_filters:
                sections.append("Suggested filters: " + ", ".join(str(f) for f in suggested_filters))
            suggested_groupby = schema_linking_result.get("suggested_groupby", [])
            if suggested_groupby:
                sections.append("Suggested GROUP BY: " + ", ".join(str(g) for g in suggested_groupby))
            suggested_orderby = schema_linking_result.get("suggested_orderby")
            if suggested_orderby:
                sections.append(f"Suggested ORDER BY: {suggested_orderby}")

        # --- Intent Classification (Stage 2 result) ---
        if intent_result and intent_result.get("primary_intent") != "unknown":
            sections.append(
                f"\n**Detected Query Intent:** {intent_result['primary_intent']}"
                f" (confidence: {intent_result.get('confidence', 0)})"
            )
            template = intent_result.get("matched_template")
            if template:
                sections.append(f"Suggested SQL template: {template}")

        # --- Table Relationships ---
        all_relationships = []
        if db_config:
            auto_fks = db_config.get("foreign_keys", [])
            if auto_fks and isinstance(auto_fks, list):
                all_relationships.extend(auto_fks)
        if self.table_relationships:
            all_relationships.extend(self.table_relationships)

        seen = set()
        unique_rels = []
        for rel in all_relationships:
            if isinstance(rel, dict) and rel.get("source_table") and rel.get("target_table"):
                key = (rel["source_table"], rel.get("source_column", ""),
                       rel["target_table"], rel.get("target_column", ""))
                if key not in seen:
                    seen.add(key)
                    unique_rels.append(rel)

        if unique_rels:
            rels = [
                f"  {rel['source_table']}.{rel.get('source_column', '?')} -> "
                f"{rel['target_table']}.{rel.get('target_column', '?')}"
                for rel in unique_rels
            ]
            sections.append(
                "\n**Table Relationships (Foreign Keys):**\n"
                + "\n".join(rels)
                + "\nUse these relationships for JOIN conditions."
            )

        # --- Column Descriptions (from knowledge or manual) ---
        col_desc_lines = []
        if knowledge:
            col_meta = knowledge.get("column_metadata", {})
            for col_name, info in col_meta.items():
                desc = info.get("description", "")
                if desc:
                    col_desc_lines.append(f"  {col_name}: {desc}")
        if self.column_descriptions:
            for cd in self.column_descriptions:
                if isinstance(cd, dict) and cd.get("column_name") and cd.get("description"):
                    col_desc_lines.append(f"  {cd.get('table_name', '')}.{cd['column_name']}: {cd['description']}")
        if col_desc_lines:
            sections.append("\n**Column Descriptions:**\n" + "\n".join(col_desc_lines))

        # --- Knowledge-enhanced sections ---
        if knowledge:
            # Synonym Reference
            synonym_map = knowledge.get("synonym_map", {})
            if synonym_map:
                syn_lines = [f'  "{term}" → {info.get("column", "?")}' for term, info in list(synonym_map.items())[:60]]
                sections.append("\n**Synonym Reference** (natural language term → column name):\n" + "\n".join(syn_lines))

            # Column Value Hints
            col_hints = knowledge.get("column_value_hints", {})
            if col_hints:
                hint_lines = []
                for col, hint in col_hints.items():
                    examples = hint.get("examples", [])
                    card = hint.get("cardinality", "?")
                    if examples:
                        hint_lines.append(f"  {col} ({card} cardinality): {', '.join(str(v) for v in examples[:8])}")
                if hint_lines:
                    sections.append("\n**Column Value Reference** (for WHERE clauses):\n" + "\n".join(hint_lines))

            # KPI Definitions
            rules = knowledge.get("business_rules", {})
            metrics = rules.get("metrics", {})
            if metrics:
                kpi_lines = [f"  {name}: {expr}" for name, expr in list(metrics.items())[:20]]
                sections.append("\n**KPI Definitions** (use these exact SQL expressions):\n" + "\n".join(kpi_lines))

            # Classification Rules
            class_rules = rules.get("classification_rules", {})
            if class_rules:
                cr_lines = [f"  {name}: {desc}" for name, desc in class_rules.items()]
                sections.append("\n**Classification Rules:**\n" + "\n".join(cr_lines))

            # Hierarchies
            hierarchies = knowledge.get("hierarchies", {})
            if hierarchies:
                h_lines = []
                for name, info in hierarchies.items():
                    levels = " → ".join(l.get("name", l.get("column", "?")) for l in info.get("levels", []))
                    h_lines.append(f"  {info.get('name', name)}: {levels}")
                sections.append("\n**Hierarchies** (drill-down support):\n" + "\n".join(h_lines))

            # Exclusion Rules
            exclusions = rules.get("exclusion_rules", [])
            if exclusions:
                sections.append("\n**Exclusion Rules:**\n" + "\n".join(f"  - {r}" for r in exclusions[:10]))

            # Additional domain context from knowledge
            additional_ctx = knowledge.get("additional_domain_context", "")
            if additional_ctx:
                sections.append(f"\n**Domain Context:**\n{additional_ctx}")

            # Additional business rules from knowledge
            additional_br = knowledge.get("additional_business_rules", "")
            if additional_br:
                sections.append(f"\n**Additional Business Rules:**\n{additional_br}")

        # --- Manual domain context & business rules ---
        if self.additional_context and self.additional_context.strip():
            sections.append(f"\n**Domain Context:**\n{self.additional_context.strip()}")
        if self.business_rules and self.business_rules.strip():
            sections.append(f"\n**Business Rules:**\n{self.business_rules.strip()}")

        # --- Few-shot Examples (dynamically selected or manual) ---
        examples_to_use = selected_examples or []
        if not examples_to_use and self.few_shot_examples:
            examples_to_use = [
                ex for ex in self.few_shot_examples
                if isinstance(ex, dict) and ex.get("question") and ex.get("sql")
            ]
        if examples_to_use:
            ex_lines = []
            for ex in examples_to_use:
                q = ex.get("question") or ex.get("input", "")
                s = ex.get("sql") or ex.get("output", "")
                if q and s:
                    ex_lines.append(f"Q: {q}\nSQL: {s}")
            if ex_lines:
                sections.append("\n**Example question-to-SQL pairs:**\n" + "\n\n".join(ex_lines))

        # --- SQL Dialect Rules (provider-aware) ---
        sections.append(f"\n{self._get_dialect_rules(provider)}")

        # --- User Question + Rules ---
        sections.append(f"\n**User Question:** {user_query}")
        sections.append("""
**Rules:**
1. Generate ONLY a SELECT query — never use INSERT, UPDATE, DELETE, DROP, or any DDL/DML
2. Use proper table and column names from the schema exactly as shown
3. Use appropriate JOINs when data spans multiple tables
4. Add meaningful aliases for calculated columns
5. Follow the SQL dialect rules above for row limiting and date functions
6. For aggregations, always include GROUP BY
7. Return ONLY the SQL query, no explanations
8. **LIKE for text matching:** For ALL text/name/string column filters (SUPPLIER_NAME, MATERIAL_GROUP, PLANT_NAME, REGION, COUNTRY, etc.), ALWAYS use `UPPER(column) LIKE '%VALUE%'` instead of `= 'VALUE'`. Names in the database have variations in spelling, casing, and formatting — exact match with = will miss valid rows. Example: use `UPPER(SUPPLIER_NAME) LIKE '%3M%'` NOT `SUPPLIER_NAME = '3M'`
9. **No redundant date filters:** The system automatically injects a mandatory `INVOICE_DATE > DATE '2024-04-01'` filter. Do NOT add your own fiscal year or calendar year filters like `TO_CHAR(INVOICE_DATE, 'YYYY') = TO_CHAR(SYSDATE, 'YYYY')` unless the user explicitly asks for a specific year. The mandatory date filter already constrains the timeframe.
10. **Reasonable row limits:** When using FETCH FIRST / LIMIT, use a reasonable number (max """ + str(self.max_rows) + """). Never use absurdly large numbers like FETCH FIRST 10000000000 ROWS ONLY. If the user asks for "all" data, use FETCH FIRST """ + str(self.max_rows) + """ ROWS ONLY as a safety cap.

**SQL Query:**""")

        return "\n".join(sections)

    def _execute_sql(self, db_config: dict, sql: str) -> Tuple[List[str], List[tuple]]:
        """Execute SQL against the database and return columns + rows."""
        # Strip trailing semicolons — Oracle's oracledb driver chokes on them
        sql = sql.strip().rstrip(";").strip()

        provider = db_config.get("provider", "postgresql")

        if provider == "postgresql":
            import psycopg2

            conn_kwargs = {
                "host": db_config["host"],
                "port": db_config["port"],
                "dbname": db_config["database_name"],
                "user": db_config["username"],
                "password": db_config["password"],
                "connect_timeout": 15,
                "options": f"-c statement_timeout={self.query_timeout * 1000}",
            }
            if db_config.get("ssl_enabled"):
                conn_kwargs["sslmode"] = "require"

            conn = psycopg2.connect(**conn_kwargs)
            try:
                cur = conn.cursor()
                cur.execute(sql)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall() if columns else []
                cur.close()
                return columns, rows
            finally:
                conn.close()
        elif provider == "oracle":
            import oracledb

            dsn = oracledb.makedsn(
                db_config["host"], db_config["port"],
                service_name=db_config["database_name"],
            )
            conn = oracledb.connect(
                user=db_config["username"],
                password=db_config["password"],
                dsn=dsn,
            )
            conn.call_timeout = self.query_timeout * 1000  # milliseconds
            try:
                cur = conn.cursor()
                cur.execute(sql)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall() if columns else []
                cur.close()
                return columns, rows
            finally:
                conn.close()
        else:
            raise ValueError(f"Provider '{provider}' not yet supported for query execution")

    def _invoke_llm_sync(self, prompt: str) -> str:
        """Invoke the LLM synchronously, handling event loop conflicts."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.llm.ainvoke(prompt))
                    response = future.result()
            else:
                response = loop.run_until_complete(self.llm.ainvoke(prompt))
        except RuntimeError:
            response = asyncio.run(self.llm.ainvoke(prompt))

        if hasattr(response, "content"):
            return response.content
        if hasattr(response, "text"):
            return response.text
        return str(response)

    def _filter_knowledge_for_prompt(
        self,
        knowledge: dict,
        schema_linking_result: Optional[dict],
        intent_result: Optional[dict],
        user_query: str,
    ) -> dict:
        """Smart context filtering — reduce prompt from ~50K tokens to ~3-4K."""
        filtered = {}
        query_lower = user_query.lower()

        resolved_cols = set()
        if schema_linking_result:
            for col in schema_linking_result.get("resolved_columns", {}).values():
                resolved_cols.add(str(col).upper())
            for entity in schema_linking_result.get("detected_entities", []):
                entities = knowledge.get("entities", {})
                ent_info = entities.get(entity, {})
                for col in ent_info.get("columns", []):
                    resolved_cols.add(str(col).upper())

        # Column metadata — only resolved columns
        col_meta = knowledge.get("column_metadata", {})
        filtered["column_metadata"] = ({k: v for k, v in col_meta.items() if k.upper() in resolved_cols}
                                        if resolved_cols else col_meta)

        # Column value hints — only resolved columns
        col_hints = knowledge.get("column_value_hints", {})
        filtered["column_value_hints"] = ({k: v for k, v in col_hints.items() if k.upper() in resolved_cols}
                                           if resolved_cols else col_hints)

        # Column values detailed — only resolved columns
        cvd = knowledge.get("column_values_detailed", {})
        if resolved_cols and cvd:
            filtered["column_values_detailed"] = {k: v for k, v in cvd.items() if k.upper() in resolved_cols}

        # Business rules — always include exclusion rules + oracle syntax
        rules = knowledge.get("business_rules", {})
        filtered_rules = {
            "exclusion_rules": rules.get("exclusion_rules", []),
            "oracle_syntax": rules.get("oracle_syntax", {}),
        }
        metric_terms = {"total", "sum", "average", "avg", "count", "spend", "cost", "kpi", "metric"}
        if any(t in query_lower for t in metric_terms):
            filtered_rules["metrics"] = rules.get("metrics", {})
        time_terms = {"year", "month", "quarter", "date", "period", "ytd", "yoy", "fy", "fiscal"}
        if any(t in query_lower for t in time_terms):
            filtered_rules["time_filters"] = rules.get("time_filters", {})
        class_terms = {"type", "category", "class", "material", "oem", "abc"}
        if any(t in query_lower for t in class_terms):
            filtered_rules["classification_rules"] = rules.get("classification_rules", {})
        filtered["business_rules"] = filtered_rules

        # Hierarchies — only if resolved columns intersect
        hierarchies = knowledge.get("hierarchies", {})
        if resolved_cols:
            filtered_h = {}
            for name, info in hierarchies.items():
                h_cols = {l.get("column", "").upper() for l in info.get("levels", [])}
                if h_cols & resolved_cols:
                    filtered_h[name] = info
            filtered["hierarchies"] = filtered_h
        else:
            filtered["hierarchies"] = hierarchies

        # Entities — only detected entities
        entities = knowledge.get("entities", {})
        if schema_linking_result:
            detected = set(schema_linking_result.get("detected_entities", []))
            filtered["entities"] = {k: v for k, v in entities.items() if k in detected}
        else:
            filtered["entities"] = entities

        filtered["additional_domain_context"] = knowledge.get("additional_domain_context", "")
        filtered["additional_business_rules"] = knowledge.get("additional_business_rules", "")
        return filtered

    def _try_template_match(
        self,
        intent_result: dict,
        schema_linking_result: Optional[dict],
        knowledge: dict,
        provider: str,
    ) -> Optional[str]:
        """Stage 4.25: Try to fill a SQL template instead of calling the LLM."""
        if not intent_result or intent_result.get("confidence_level") != "high":
            return None

        templates = knowledge.get("sql_templates", {})
        if not templates:
            return None

        primary_intent = intent_result.get("primary_intent", "")
        sl = schema_linking_result or {}

        intent_to_template = {
            "enumerate": "enumerate_distinct",
            "top_n": "top_n_by_spend",
            "bottom_n": "top_n_by_spend",
            "aggregation_grouped": "aggregation_grouped",
            "aggregation": "aggregation_grouped",
            "time_series": "time_series_monthly",
            "comparison": "comparison_yoy",
            "count": "count_distinct",
        }

        template_name = intent_to_template.get(primary_intent)
        if not template_name or template_name not in templates:
            return None

        template_info = templates[template_name]
        sql_template = template_info.get("template", "")
        if not sql_template:
            return None

        resolved = sl.get("resolved_columns", {})
        filters = sl.get("suggested_filters", [])
        groupby = sl.get("suggested_groupby", [])
        limit = sl.get("suggested_limit")

        where_parts = list(filters) if filters else []
        where_clause = "WHERE " + " AND ".join(str(f) for f in where_parts) if where_parts else ""

        dimension = ""
        if groupby:
            dimension = str(groupby[0])
        elif resolved:
            dimension = next(iter(resolved.values()), "")

        if not dimension:
            return None

        try:
            sql = sql_template.format(
                column=dimension, dimension=dimension,
                dimension1=dimension,
                dimension2=groupby[1] if len(groupby) > 1 else dimension,
                where_clause=where_clause, n=limit or 10,
                filter_column="", filter_value="",
                count_column=dimension, alias="COUNT",
                columns=f"{dimension}, SUM(AMOUNT) AS TOTAL_SPEND",
                group_by=f"GROUP BY {dimension}", threshold=0,
            )
        except (KeyError, IndexError):
            return None

        sql = sql.strip()
        is_valid, _ = _validate_sql(sql, provider)
        return sql if is_valid else None

    def _apply_anti_patterns(self, sql: str, knowledge: dict) -> Tuple[str, List[str]]:
        """Stage 4.5: Detect and auto-fix SQL anti-patterns."""
        anti_patterns = knowledge.get("anti_patterns", [])
        if not anti_patterns:
            return sql, []

        fixes_applied = []
        fixed_sql = sql

        for ap in anti_patterns:
            compiled = ap.get("compiled")
            if not compiled:
                continue
            severity = ap.get("severity", "warning")
            ap_name = ap.get("name", ap.get("id", "unknown"))
            if ap.get("required") or ap.get("forbidden"):
                continue

            if compiled.search(fixed_sql):
                fix_text = ap.get("fix", "")

                if "LIMIT" in ap_name.upper() or "LIMIT" in fix_text.upper():
                    limit_match = re.search(r"\bLIMIT\s+(\d+)", fixed_sql, re.IGNORECASE)
                    if limit_match:
                        n = limit_match.group(1)
                        fixed_sql = re.sub(r"\bLIMIT\s+\d+", f"FETCH FIRST {n} ROWS ONLY", fixed_sql, flags=re.IGNORECASE)
                        fixes_applied.append(f"Fixed {ap_name}: LIMIT {n} → FETCH FIRST {n} ROWS ONLY")
                        continue

                if "TOP" in ap_name.upper():
                    top_match = re.search(r"\bTOP\s+(\d+)\b", fixed_sql, re.IGNORECASE)
                    if top_match:
                        n = top_match.group(1)
                        fixed_sql = re.sub(r"\bSELECT\s+TOP\s+\d+\b", "SELECT", fixed_sql, flags=re.IGNORECASE)
                        if "FETCH FIRST" not in fixed_sql.upper():
                            fixed_sql = fixed_sql.rstrip() + f"\nFETCH FIRST {n} ROWS ONLY"
                        fixes_applied.append(f"Fixed {ap_name}: TOP {n} → FETCH FIRST {n} ROWS ONLY")
                        continue

                if "semicolon" in ap_name.lower():
                    fixed_sql = fixed_sql.rstrip().rstrip(";").rstrip()
                    fixes_applied.append(f"Fixed {ap_name}: removed trailing semicolon")
                    continue

                if "ILIKE" in ap_name.upper():
                    fixed_sql = re.sub(r"(\w+)\s+ILIKE\s+'([^']*)'", r"UPPER(\1) LIKE UPPER('\2')", fixed_sql, flags=re.IGNORECASE)
                    fixes_applied.append(f"Fixed {ap_name}: ILIKE → UPPER(...) LIKE UPPER(...)")
                    continue

                if severity == "error":
                    fixes_applied.append(f"WARNING ({ap_name}): {ap.get('description', fix_text)}")
                elif severity == "warning":
                    fixes_applied.append(f"Note ({ap_name}): {ap.get('description', fix_text)}")

        return fixed_sql, fixes_applied

    def _validate_against_ontology(self, sql: str, knowledge: dict) -> Tuple[bool, str, List[str]]:
        """Stage 5: Validate generated SQL against ontology rules."""
        warnings = []
        sql_upper = sql.upper()

        valid_combos = knowledge.get("valid_combinations", {})
        col_meta = knowledge.get("column_metadata", {})

        if not col_meta:
            return True, "OK", []

        known_columns = {c.upper() for c in col_meta.keys()}

        # Check columns in SELECT
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_upper, re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            potential_cols = re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", select_clause)
            for col in potential_cols:
                if col not in known_columns and col not in {
                    "SUM", "COUNT", "AVG", "MAX", "MIN", "DISTINCT", "CASE",
                    "WHEN", "THEN", "ELSE", "END", "NULL", "EXTRACT", "YEAR",
                    "MONTH", "QUARTER", "TOTAL_SPEND", "TRANSACTION_COUNT",
                    "SUPPLIER_COUNT", "AVG_SPEND", "ROWS", "ONLY", "FIRST",
                    "FETCH", "DESC", "ASC", "FROM",
                }:
                    if not re.search(rf"\bAS\s+{col}\b", sql_upper):
                        warnings.append(f"UNKNOWN_COLUMN: '{col}' not found in known schema columns")

        # Check measure-dimension validity
        combos = valid_combos.get("valid_combinations", {})
        if combos and ("SUM(AMOUNT)" in sql_upper or "SUM( AMOUNT )" in sql_upper):
            group_match = re.search(r"GROUP\s+BY\s+(.*?)(?:ORDER|HAVING|FETCH|$)", sql_upper, re.DOTALL)
            if group_match:
                group_cols = re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", group_match.group(1))
                spend_dims = combos.get("Spend", combos.get("spend", {}))
                if isinstance(spend_dims, dict):
                    valid_dims = {d.upper() for d in spend_dims.get("dimensions", [])}
                    for gc in group_cols:
                        if gc in known_columns and valid_dims and gc not in valid_dims:
                            warnings.append(f"ONTOLOGY: Grouping SUM(AMOUNT) by '{gc}' may not be a standard analysis dimension")

        return True, "OK", warnings

    def _post_result_validation(
        self, columns: List[str], rows: List[tuple], sql: str, knowledge: Optional[dict] = None,
    ) -> List[str]:
        """Stage 7: Post-Result Validation — sanity-check query results.

        Deterministic checks:
        1. Empty results detection
        2. High NULL prevalence per column
        3. Unexpected negative values in spend/amount columns
        4. Extreme magnitude spread (potential outliers)
        5. Duplicate row detection (missing GROUP BY / DISTINCT)
        6. NULL aggregate detection (single-row single-column NULL)
        """
        warnings = []

        if not rows:
            warnings.append(
                "EMPTY_RESULT: Query returned no rows. The filters may be too "
                "restrictive or data may not exist for the specified criteria."
            )
            return warnings

        num_rows = len(rows)

        # --- NULL prevalence per column ---
        for col_idx, col_name in enumerate(columns):
            null_count = sum(1 for row in rows if row[col_idx] is None)
            if null_count == num_rows:
                warnings.append(f"ALL_NULL: Column '{col_name}' is entirely NULL.")
            elif num_rows > 1 and null_count > num_rows * 0.5:
                pct = round(null_count / num_rows * 100)
                warnings.append(f"HIGH_NULL: Column '{col_name}' has {pct}% NULL values.")

        # --- Negative values in spend/amount columns ---
        spend_keywords = {"AMOUNT", "SPEND", "COST", "TOTAL", "SUM", "REVENUE"}
        for col_idx, col_name in enumerate(columns):
            if any(kw in col_name.upper() for kw in spend_keywords):
                neg_count = sum(
                    1 for row in rows
                    if row[col_idx] is not None
                    and isinstance(row[col_idx], (int, float))
                    and row[col_idx] < 0
                )
                if neg_count > 0:
                    warnings.append(
                        f"NEGATIVE_VALUES: Column '{col_name}' has {neg_count} negative "
                        f"value(s). May indicate credits/returns or data quality issue."
                    )

        # --- Magnitude spread (outlier detection) ---
        for col_idx, col_name in enumerate(columns):
            numeric_vals = [
                row[col_idx] for row in rows
                if row[col_idx] is not None and isinstance(row[col_idx], (int, float))
            ]
            if len(numeric_vals) >= 2:
                abs_vals = [abs(v) for v in numeric_vals if v != 0]
                if abs_vals:
                    max_val = max(abs_vals)
                    min_val = min(abs_vals)
                    if min_val > 0 and max_val / min_val > 10000:
                        warnings.append(
                            f"MAGNITUDE_SPREAD: Column '{col_name}' has extreme value "
                            f"spread (max/min ratio > 10,000). Check for outliers."
                        )

        # --- Duplicate rows ---
        if num_rows > 1:
            unique_count = len({tuple(row) for row in rows})
            if unique_count < num_rows:
                dup_count = num_rows - unique_count
                warnings.append(
                    f"DUPLICATES: {dup_count} duplicate row(s) detected. "
                    f"The query may need GROUP BY or DISTINCT."
                )

        # --- NULL aggregate (single aggregation returning NULL) ---
        if num_rows == 1 and len(columns) == 1 and rows[0][0] is None:
            warnings.append(
                "NULL_AGGREGATE: The aggregation returned NULL. The table may be "
                "empty or all values are NULL for the given filters."
            )

        return warnings

    def _format_pipeline_trace(
        self,
        trace: dict,
        normalized_query: str,
        original_query: str,
        pipeline_log: list,
    ) -> str:
        """Format the pipeline trace as a visible step-by-step breakdown."""
        lines = ["\n---\n**Pipeline Trace** (7-Stage NL-to-SQL Resolution)\n"]

        # Stage 0: Normalization
        s0 = trace.get("stage_0_normalizer")
        if s0:
            lines.append("**Stage 0 — Query Normalization** (CODE)")
            if original_query != normalized_query:
                lines.append(f"- Original: `{original_query}`")
                lines.append(f"- Normalized: `{normalized_query}`")
            else:
                lines.append(f"- Query: `{original_query}` (no changes)")
            expansions = s0.get("expansions", [])
            if expansions:
                lines.append(f"- Expansions: {', '.join(expansions)}")
            aliases = s0.get("alias_resolutions", [])
            if aliases:
                for a in aliases:
                    lines.append(f"- Alias: `{a.get('alias', '?')}` → `{a.get('sql_filter', '?')}`")
            lines.append("")

        # Stage 1: Schema Linking
        s1 = trace.get("stage_1_schema_linking")
        if s1:
            lines.append("**Stage 1 — Schema Linking** (LLM)")
            resolved = s1.get("resolved_columns", {})
            if resolved:
                for term, col in resolved.items():
                    lines.append(f"- `{term}` → `{col}`")
            entities = s1.get("detected_entities", [])
            if entities:
                lines.append(f"- Entities: {', '.join(entities)}")
            filters = s1.get("suggested_filters", [])
            if filters:
                lines.append(f"- Filters: {', '.join(str(f) for f in filters)}")
            lines.append("")

        # Stage 2: Intent Classification
        s2 = trace.get("stage_2_intent")
        if s2:
            intent = s2.get("primary_intent", "unknown")
            confidence = s2.get("confidence", 0)
            level = s2.get("confidence_level", "?")
            bar_len = int(confidence * 20)
            bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
            lines.append("**Stage 2 — Intent Classification** (CODE)")
            lines.append(f"- Intent: `{intent}` ({confidence:.0%}) {bar} {level.upper()}")
            secondary = s2.get("secondary_intents", [])
            if secondary:
                lines.append(f"- Secondary: {', '.join(secondary)}")
            lines.append("")

        # Stage 3: Example Selection
        s3 = trace.get("stage_3_examples")
        if s3:
            lines.append("**Stage 3 — Example Selection** (CODE)")
            lines.append(f"- Selected {s3.get('selected', 0)} of {s3.get('total', 0)} examples")
            lines.append("")

        # Stage 4.25: Template Match
        s425 = trace.get("stage_4_25_template")
        if s425 and s425.get("matched"):
            lines.append("**Stage 4.25 — Template Match** (CODE)")
            lines.append("- Template matched — **LLM call skipped**")
            lines.append("")

        # Stage 4: SQL Generation
        s4 = trace.get("stage_4_sql_gen")
        if s4:
            source = s4.get("source", "llm")
            lines.append(f"**Stage 4 — SQL Generation** ({source.upper()})")
            lines.append("")

        # Post-processing & Anti-patterns
        s45 = trace.get("stage_4_5_anti_patterns", [])
        pp = trace.get("post_process_fixes", [])
        all_fixes = s45 + pp
        if all_fixes:
            lines.append("**Stage 4.5 — Anti-Pattern & Post-Processing Fixes** (CODE)")
            for fix in all_fixes:
                lines.append(f"- {fix}")
            lines.append("")

        # Date filter injection
        if trace.get("date_filter_injected"):
            lines.append("**Safety** — Mandatory date filter injected: `INVOICE_DATE > DATE '2024-04-01'`\n")

        # Stage 5: Validation
        s5 = trace.get("stage_5_validation")
        if s5:
            ont_warnings = s5.get("ontology_warnings", [])
            lines.append("**Stage 5 — Validation** (CODE)")
            if ont_warnings:
                for w in ont_warnings:
                    lines.append(f"- \u26a0\ufe0f {w}")
            else:
                lines.append("- \u2705 SQL safety check passed")
                lines.append("- \u2705 Ontology validation passed")
            lines.append("")

        # Stage 6: Execution
        s6 = trace.get("stage_6_execution")
        if s6:
            lines.append("**Stage 6 — Execution** (DB)")
            lines.append(f"- {s6.get('rows', 0)} rows returned in {s6.get('time_ms', 0)}ms")
            lines.append("")

        # Stage 7: Post-Result Validation
        s7 = trace.get("stage_7_post_validation")
        if s7:
            warnings = s7.get("warnings", [])
            lines.append("**Stage 7 — Post-Result Validation** (CODE)")
            if warnings:
                for w in warnings:
                    lines.append(f"- \u26a0\ufe0f {w}")
            else:
                lines.append("- \u2705 No data quality issues detected")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def _run_nl_to_sql(self) -> dict:
        """Multi-stage pipeline with smart filtering, template matching, and anti-pattern fixing."""
        if self._cached_result is not None:
            return self._cached_result

        db_data = self.db_connection
        logger.info(f"NL-to-SQL: db_connection type={type(db_data).__name__}")

        if isinstance(db_data, Data):
            db_config = db_data.data
        elif isinstance(db_data, dict):
            db_config = db_data
        else:
            db_config = {}

        schema_ddl = db_config.get("schema_ddl", "")
        if not schema_ddl:
            self._cached_result = {
                "error": True,
                "message": "No schema information available. Please check the Database Connector.",
            }
            return self._cached_result

        user_query = self.user_query
        if not user_query or not user_query.strip():
            self._cached_result = {
                "error": True,
                "message": "No question provided.",
            }
            return self._cached_result

        provider = self._get_provider(db_config)
        knowledge = self._get_knowledge()
        pipeline_log = []
        pipeline_trace = {}

        # ===== STAGE 0: Query Normalization =====
        normalized_query = user_query
        normalizer_result = None
        if knowledge and getattr(self, "enable_query_normalization", True):
            try:
                from agentcore.components.tools.knowledge_layer import normalize_query
                normalizer_result = normalize_query(user_query, knowledge)
                normalized_query = normalizer_result.get("normalized_query", user_query)
                expansions = normalizer_result.get("expansions", [])
                alias_res = normalizer_result.get("alias_resolutions", [])
                pipeline_log.append(f"Stage 0: Normalized ({len(expansions)} expansions, {len(alias_res)} aliases)")
                pipeline_trace["stage_0_normalizer"] = normalizer_result
            except Exception as e:
                logger.warning(f"NL-to-SQL Stage 0 (Normalizer) failed (non-fatal): {e}")

        # ===== STAGE 1: Schema Linking (LLM call) =====
        schema_linking_result = None
        if knowledge and self.enable_schema_linking:
            try:
                sl_query = normalized_query
                if normalizer_result and normalizer_result.get("alias_resolutions"):
                    alias_hints = [f'{a["alias"]} → {a["sql_filter"]}' for a in normalizer_result["alias_resolutions"]]
                    sl_query = f"{normalized_query}\n[Alias hints: {'; '.join(alias_hints)}]"

                sl_prompt = self._build_schema_linking_prompt(sl_query, knowledge)
                sl_response = self._invoke_llm_sync(sl_prompt)
                schema_linking_result = self._parse_schema_linking(sl_response)

                if normalizer_result and normalizer_result.get("alias_resolutions"):
                    existing_filters = schema_linking_result.get("suggested_filters", [])
                    for alias in normalizer_result["alias_resolutions"]:
                        sf = alias.get("sql_filter", "")
                        if sf and sf not in existing_filters:
                            existing_filters.append(sf)
                    schema_linking_result["suggested_filters"] = existing_filters

                pipeline_log.append(f"Stage 1: Schema linking resolved {len(schema_linking_result.get('resolved_columns', {}))} columns")
                pipeline_trace["stage_1_schema_linking"] = schema_linking_result
            except Exception as e:
                logger.warning(f"NL-to-SQL Stage 1 (Schema Linking) failed (non-fatal): {e}")
                pipeline_log.append(f"Stage 1: Schema linking skipped ({e})")

        # ===== STAGE 2: Intent Classification =====
        intent_result = None
        if knowledge and self.enable_intent_classification:
            try:
                from agentcore.components.tools.knowledge_layer import classify_intent
                intent_index = knowledge.get("intent_index", {})
                if intent_index:
                    intent_result = classify_intent(normalized_query, intent_index)
                    pipeline_log.append(
                        f"Stage 2: Intent={intent_result.get('primary_intent', 'unknown')} "
                        f"(confidence: {intent_result.get('confidence', 0)}, "
                        f"level: {intent_result.get('confidence_level', '?')})"
                    )
                    pipeline_trace["stage_2_intent"] = intent_result
            except Exception as e:
                logger.warning(f"NL-to-SQL Stage 2 (Intent) failed (non-fatal): {e}")

        # ===== STAGE 3: Dynamic Example Selection =====
        selected_examples = None
        if knowledge:
            try:
                from agentcore.components.tools.knowledge_layer import select_relevant_examples
                all_examples = knowledge.get("examples", [])
                detected_entities = (schema_linking_result or {}).get("detected_entities", [])
                max_ex = self.num_examples if self.num_examples > 0 else len(all_examples)
                if max_ex > 3 and knowledge.get("sql_templates"):
                    max_ex = min(max_ex, 3)
                selected_examples = select_relevant_examples(
                    intent_result=intent_result or {"primary_intent": "unknown", "secondary_intents": []},
                    detected_entities=detected_entities,
                    all_examples=all_examples,
                    max_examples=max_ex,
                )
                pipeline_log.append(f"Stage 3: Selected {len(selected_examples)}/{len(all_examples)} examples")
                pipeline_trace["stage_3_examples"] = {"selected": len(selected_examples), "total": len(all_examples)}
            except Exception as e:
                logger.warning(f"NL-to-SQL Stage 3 (Examples) failed (non-fatal): {e}")

        # ===== SMART CONTEXT FILTERING =====
        filtered_knowledge = knowledge
        if knowledge:
            try:
                filtered_knowledge = self._filter_knowledge_for_prompt(
                    knowledge, schema_linking_result, intent_result, normalized_query,
                )
                pipeline_log.append("Context filtering: applied")
            except Exception as e:
                logger.warning(f"NL-to-SQL context filtering failed (non-fatal): {e}")
                filtered_knowledge = knowledge

        # ===== STAGE 4.25: Template Matching =====
        template_sql = None
        if knowledge and getattr(self, "enable_template_matching", True) and intent_result:
            try:
                template_sql = self._try_template_match(intent_result, schema_linking_result, knowledge, provider)
                if template_sql:
                    pipeline_log.append("Stage 4.25: Template match SUCCESS — skipping LLM")
                    pipeline_trace["stage_4_25_template"] = {"matched": True, "sql": template_sql}
            except Exception as e:
                logger.warning(f"NL-to-SQL Stage 4.25 (Template) failed (non-fatal): {e}")

        # ===== STAGE 4: SQL Generation (LLM — skipped if template matched) =====
        if template_sql:
            sql = template_sql
        else:
            prompt = self._build_sql_generation_prompt(
                schema_ddl, normalized_query, db_config,
                schema_linking_result=schema_linking_result,
                intent_result=intent_result,
                selected_examples=selected_examples,
                knowledge=filtered_knowledge,
            )

            try:
                generated_sql = self._invoke_llm_sync(prompt)
            except Exception as e:
                self._cached_result = {
                    "error": True,
                    "message": f"LLM SQL generation failed: {e!s}",
                    "pipeline_log": pipeline_log,
                }
                return self._cached_result

            sql = generated_sql.strip()
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            sql = sql.strip()
            pipeline_trace["stage_4_sql_gen"] = {"source": "llm"}

        # ===== STAGE 4.5: Anti-Pattern Fix =====
        anti_pattern_fixes = []
        if knowledge and getattr(self, "enable_anti_pattern_check", True):
            try:
                sql, anti_pattern_fixes = self._apply_anti_patterns(sql, knowledge)
                if anti_pattern_fixes:
                    pipeline_log.append(f"Stage 4.5: Applied {len(anti_pattern_fixes)} anti-pattern fix(es)")
                    pipeline_trace["stage_4_5_anti_patterns"] = anti_pattern_fixes
            except Exception as e:
                logger.warning(f"NL-to-SQL Stage 4.5 (Anti-Pattern) failed (non-fatal): {e}")

        # ===== Built-in Post-Processing (always runs) =====
        sql, pp_fixes = _post_process_sql(sql, provider, self.max_rows)
        if pp_fixes:
            anti_pattern_fixes.extend(pp_fixes)
            pipeline_log.append(f"Post-process: {len(pp_fixes)} fix(es) — {'; '.join(pp_fixes)}")
            pipeline_trace["post_process_fixes"] = pp_fixes

        # ===== Mandatory Date Filter (Oracle safety net) =====
        if provider == "oracle":
            sql, date_injected = _enforce_oracle_date_filter(sql)
            if date_injected:
                pipeline_log.append("Safety: Injected mandatory INVOICE_DATE > DATE '2024-04-01' filter")
                pipeline_trace["date_filter_injected"] = True
                logger.info("NL-to-SQL: Injected mandatory Oracle date filter")

        # ===== STAGE 5: Validation =====
        is_valid, validation_msg = _validate_sql(sql, provider)
        if not is_valid:
            self._cached_result = {
                "error": True,
                "message": f"SQL validation failed: {validation_msg}\n\nGenerated SQL:\n```sql\n{sql}\n```",
                "generated_sql": sql,
                "pipeline_log": pipeline_log,
                "pipeline_trace": pipeline_trace,
            }
            return self._cached_result

        ontology_warnings = []
        if knowledge and self.enable_ontology_validation:
            try:
                ont_valid, ont_msg, ontology_warnings = self._validate_against_ontology(sql, knowledge)
                if not ont_valid:
                    pipeline_log.append(f"Ontology validation: {ont_msg}")
                if ontology_warnings:
                    pipeline_log.extend(f"Warning: {w}" for w in ontology_warnings)
            except Exception as e:
                logger.warning(f"NL-to-SQL Stage 5 (Ontology) failed (non-fatal): {e}")

        pipeline_log.append("Stage 5: SQL validated successfully")
        pipeline_trace["stage_5_validation"] = {"ontology_warnings": ontology_warnings}

        # ===== STAGE 6: SQL Execution =====
        try:
            start_time = time.time()
            columns, rows = self._execute_sql(db_config, sql)
            exec_time = round((time.time() - start_time) * 1000, 2)
        except Exception as e:
            self._cached_result = {
                "error": True,
                "message": f"Query execution failed: {e!s}\n\nGenerated SQL:\n```sql\n{sql}\n```",
                "generated_sql": sql,
                "pipeline_log": pipeline_log,
                "pipeline_trace": pipeline_trace,
            }
            return self._cached_result

        pipeline_log.append(f"Stage 6: Executed — {len(rows)} rows in {exec_time}ms")
        pipeline_trace["stage_6_execution"] = {"rows": len(rows), "time_ms": exec_time}

        # ===== STAGE 7: Post-Result Validation =====
        post_warnings = []
        try:
            post_warnings = self._post_result_validation(columns, rows, sql, knowledge)
            if post_warnings:
                pipeline_log.extend(f"Stage 7: {w}" for w in post_warnings)
            else:
                pipeline_log.append("Stage 7: Post-result validation passed")
        except Exception as e:
            logger.warning(f"NL-to-SQL Stage 7 (Post-Result Validation) failed (non-fatal): {e}")

        pipeline_trace["stage_7_post_validation"] = {"warnings": post_warnings}

        self._cached_result = {
            "error": False,
            "generated_sql": sql,
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "execution_time_ms": exec_time,
            "user_query": user_query,
            "normalized_query": normalized_query,
            "pipeline_log": pipeline_log,
            "pipeline_trace": pipeline_trace,
            "ontology_warnings": ontology_warnings,
            "post_result_warnings": post_warnings,
            "anti_pattern_fixes": anti_pattern_fixes,
        }
        return self._cached_result

    def run_query(self) -> Message:
        """Return formatted markdown result."""
        result = self._run_nl_to_sql()

        if result.get("error"):
            self.status = "Error"
            return Message(text=result["message"])

        # Build response
        parts = []
        parts.append(f"**Query Results** ({result['row_count']} rows, {result['execution_time_ms']}ms)\n")

        if self.include_sql:
            parts.append(f"**Generated SQL:**\n```sql\n{result['generated_sql']}\n```\n")

        # Markdown table
        md_table = _format_results_as_markdown(result["columns"], [tuple(r) for r in result["rows"]])
        parts.append(md_table)

        # Anti-pattern fixes applied
        anti_fixes = result.get("anti_pattern_fixes", [])
        if anti_fixes:
            parts.append("\n**Auto-Fixes Applied:**")
            for fix in anti_fixes:
                parts.append(f"- {fix}")

        # Post-result validation warnings
        post_warnings = result.get("post_result_warnings", [])
        if post_warnings:
            parts.append("\n**Data Quality Notes:**")
            for w in post_warnings:
                parts.append(f"- {w}")

        # Pipeline trace — visible step-by-step breakdown
        trace = result.get("pipeline_trace", {})
        if trace:
            parts.append(self._format_pipeline_trace(
                trace,
                result.get("normalized_query", result.get("user_query", "")),
                result.get("user_query", ""),
                result.get("pipeline_log", []),
            ))

        self.status = f"{result['row_count']} rows returned"
        return Message(text="\n".join(parts))

    def run_query_raw(self) -> Data:
        """Return raw structured data for visualization."""
        result = self._run_nl_to_sql()

        if result.get("error"):
            self.status = "Error"
            return Data(data={"error": True, "message": result["message"]})

        self.status = f"{result['row_count']} rows (raw)"
        return Data(data={
            "columns": result["columns"],
            "rows": result["rows"],
            "row_count": result["row_count"],
            "execution_time_ms": result["execution_time_ms"],
            "generated_sql": result["generated_sql"],
            "user_query": result["user_query"],
        })