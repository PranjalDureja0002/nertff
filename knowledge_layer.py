"""Knowledge Layer component for the Agent Builder.

Ingests knowledge artifacts and produces a unified, structured knowledge context
for downstream NL-to-SQL components.  Files are auto-detected by their content
structure — no fixed naming conventions required.

Recognised artifact types (all optional):
  - Knowledge graph  (entities, relationships, hierarchies)
  - Ontology         (valid combinations, constraints, drill-down hierarchies)
  - Semantic layer   (column metadata, cardinality, entity mappings)
  - Context graph    (question patterns → SQL patterns)
  - Synonyms         (column ↔ natural-language term mappings)
  - Business rules   (metrics, exclusions, time filters, DB-specific syntax)
  - Few-shot examples(question → SQL pairs)
  - Domain terms     (translations, abbreviations, domain-specific jargon)
  - Schema columns   (column definitions with types and descriptions)
  - SQL templates    (parameterized SQL for common query patterns)
  - Anti-patterns    (regex-based SQL error detection with auto-fix rules)
  - Column values    (actual value distributions with frequencies and tiers)
  - Entity aliases   (region, country, business concept, OEM, commodity aliases)

This is the "Fusion Layer" — combining mechanized knowledge creation with
business validation into a single, reusable knowledge context.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from agentcore.custom.custom_node.node import Node
from agentcore.inputs.inputs import (
    HandleInput,
    MultilineInput,
    TableInput,
)
from agentcore.schema.data import Data
from agentcore.schema.message import Message
from agentcore.template.field.base import Output
from agentcore.logging import logger


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

# Filename substring → slot  (fallback when content detection is ambiguous)
_FILENAME_HINTS = {
    "knowledge_graph": "knowledge_graph_file",
    "ontology": "ontology_file",
    "semantic_layer": "semantic_layer_file",
    "context_graph": "context_graph_file",
    "synonym": "synonyms_file",
    "business_rule": "business_rules_file",
    "example": "examples_file",
    "term": "domain_terms_file",       # generic — matches german_terms, domain_terms, etc.
    "german": "domain_terms_file",
    "column": "schema_columns_file",
    # Phase 1 new types
    "sql_template": "sql_templates_file",
    "template": "sql_templates_file",
    "anti_pattern": "anti_patterns_file",
    "column_value": "column_values_file",
    "histogram": "column_values_file",
    "entities_alias": "entities_aliases_file",
    "alias": "entities_aliases_file",
}


def _detect_artifact_type(parsed: Any, filename: str = "") -> Optional[str]:
    """Detect knowledge artifact type from content structure, with filename fallback.

    Returns the internal slot name (e.g. ``'ontology_file'``) or ``None``.
    """
    # --- Content-based detection (preferred — works regardless of filename) ---
    if isinstance(parsed, dict):
        keys = set(parsed.keys())

        # Knowledge graph: entities + relationships
        if "entities" in keys and ("relationships" in keys or "relations" in keys):
            return "knowledge_graph_file"

        # Ontology: hierarchies + (valid_combinations or constraints)
        if "hierarchies" in keys and ("valid_combinations" in keys or "constraints" in keys):
            return "ontology_file"

        # Semantic layer: columns + (entity_mappings or cardinality_summary)
        if "columns" in keys and ("entity_mappings" in keys or "cardinality_summary" in keys):
            return "semantic_layer_file"

        # Context graph: question_types or query_templates
        if "question_types" in keys:
            return "context_graph_file"

        # Synonyms: column_synonyms
        if "column_synonyms" in keys:
            return "synonyms_file"

        # Business rules: metrics + (exclusion_rules or time_filters or oracle_syntax)
        if "metrics" in keys and (
            "exclusion_rules" in keys or "exclusions" in keys
            or "time_filters" in keys or "oracle_syntax" in keys
            or "oracle_specific" in keys
        ):
            return "business_rules_file"

        # Few-shot examples: has "examples" list with question/sql dicts
        if "examples" in keys and isinstance(parsed.get("examples"), list):
            ex_list = parsed["examples"]
            if ex_list and isinstance(ex_list[0], dict):
                if "question" in ex_list[0] or "input" in ex_list[0] or "sql" in ex_list[0]:
                    return "examples_file"

        # Domain terms / translations: column_mappings with translation info
        if "column_mappings" in keys:
            return "domain_terms_file"

        # SQL templates: has "templates" key with items containing template/sql_template
        if "templates" in keys:
            templates_val = parsed.get("templates", {})
            if isinstance(templates_val, dict):
                sample = next(iter(templates_val.values()), None)
                if isinstance(sample, dict) and ("template" in sample or "sql_template" in sample):
                    return "sql_templates_file"

        # Anti-patterns: has "anti_patterns" key with items containing pattern/severity
        if "anti_patterns" in keys:
            ap_val = parsed.get("anti_patterns", [])
            if isinstance(ap_val, list):
                return "anti_patterns_file"

        # Column values: has "columns" key where values have values/cardinality subkeys
        if "columns" in keys and "columns_by_tier" in keys:
            return "column_values_file"
        if "columns" in keys:
            cols_val = parsed.get("columns", {})
            if isinstance(cols_val, dict):
                sample = next(iter(cols_val.values()), None)
                if isinstance(sample, dict) and ("cardinality" in sample or "values" in sample or "tier" in sample):
                    return "column_values_file"

        # Entity aliases: has region_aliases or business_concepts or oem_aliases
        if any(k in keys for k in ("region_aliases", "business_concepts", "oem_aliases", "commodity_aliases", "country_aliases")):
            return "entities_aliases_file"

        # Schema columns: all top-level values are dicts with type/category/description
        if keys:
            sample_values = [parsed[k] for k in list(keys)[:5] if isinstance(parsed[k], dict)]
            if len(sample_values) >= 3:
                sample = sample_values[0]
                if any(k in sample for k in ("type", "data_type", "category", "description")):
                    return "schema_columns_file"

    # List of examples (bare list format)
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        if "question" in parsed[0] or "input" in parsed[0]:
            return "examples_file"

    # --- Filename fallback ---
    fname_lower = Path(filename).stem.lower() if filename else ""
    for hint, slot in _FILENAME_HINTS.items():
        if hint in fname_lower:
            return slot

    return None


def _safe_parse(raw: Any) -> Optional[Union[dict, list]]:
    """Parse raw file content into dict/list. Handles Data, dict, str (YAML/JSON)."""
    if raw is None:
        return None
    if isinstance(raw, Data):
        inner = raw.data
        if isinstance(inner, dict):
            # If Data wraps file content under a "text" or "content" key, extract it
            text = inner.get("text") or inner.get("content") or inner.get("file_content")
            if text and isinstance(text, str):
                return _parse_text(text)
            return inner
        if isinstance(inner, str):
            return _parse_text(inner)
        return inner
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return _parse_text(raw)
    return None


def _parse_text(text: str) -> Optional[Union[dict, list]]:
    """Try parsing as JSON first, then YAML, then flat-YAML fallback."""
    text = text.strip()
    if not text:
        return None
    # JSON
    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # YAML
    try:
        result = yaml.safe_load(text)
        if isinstance(result, (dict, list)):
            return result
    except yaml.YAMLError:
        pass
    # Fallback: parse flat YAML (files with zero indentation)
    try:
        return _parse_flat_yaml(text)
    except Exception:
        pass
    return None


def _parse_flat_yaml(text: str) -> Optional[dict]:
    """Parse YAML files that have no indentation (all lines at column 0).

    These files have a consistent pattern where hierarchy is implied by:
    - Lines ending with ':' alone are section headers
    - Lines with 'key: value' are properties
    - Lines starting with '- ' are list items
    - Lines with 'key: |' start block scalars (SQL etc.)
    - Blank lines separate sibling items within a section

    Returns a flat dict with section names as top-level keys and their
    content preserved as-is for downstream consumption.
    """
    lines = text.replace('\r\n', '\n').split('\n')
    result = {}
    current_section = None
    current_item = None
    current_item_key = None
    items_in_section = {}
    in_block_scalar = False
    block_lines = []
    block_key = None
    in_list = False
    list_key = None
    list_items = []

    def _flush_block():
        nonlocal in_block_scalar, block_lines, block_key
        if in_block_scalar and block_key and current_item is not None:
            current_item[block_key] = '\n'.join(block_lines)
        in_block_scalar = False
        block_lines = []
        block_key = None

    def _flush_list():
        nonlocal in_list, list_items, list_key
        if in_list and list_key and current_item is not None:
            current_item[list_key] = list_items
        in_list = False
        list_items = []
        list_key = None

    def _flush_item():
        nonlocal current_item, current_item_key
        _flush_block()
        _flush_list()
        if current_item_key and current_item is not None:
            items_in_section[current_item_key] = current_item
        current_item = None
        current_item_key = None

    def _flush_section():
        nonlocal current_section, items_in_section
        _flush_item()
        if current_section and items_in_section:
            result[current_section] = items_in_section
        items_in_section = {}

    for raw_line in lines:
        stripped = raw_line.strip()

        # Empty line
        if not stripped:
            _flush_block()
            _flush_list()
            # Blank line between items signals a new sibling item
            if current_item_key and current_item is not None:
                _flush_item()
            continue

        # Comment
        if stripped.startswith('#'):
            continue

        # Block scalar content
        if in_block_scalar:
            # Check if this looks like a new YAML key (not block content)
            if re.match(r'^[A-Za-z_"\'][A-Za-z0-9_ .()"\'/]*:\s', stripped) or \
               (stripped.endswith(':') and not stripped.startswith(('SELECT', 'FROM', 'WHERE'))):
                _flush_block()
                # Fall through to process as normal line
            else:
                block_lines.append(stripped)
                continue

        # List item
        if stripped.startswith('- '):
            item_val = stripped[2:].strip()
            if not in_list and current_item is None and current_section:
                # List directly under a section (like examples: followed by - id: ...)
                # Parse as list of dicts
                if current_section not in result:
                    result[current_section] = []
                if isinstance(result.get(current_section), list):
                    # Parse the list item as a key:value if it contains ':'
                    if ': ' in item_val or item_val.endswith(':'):
                        current_item = {}
                        current_item_key = '__list_item__'
                        k, _, v = item_val.partition(':')
                        current_item[k.strip()] = v.strip() if v.strip() else None
                        continue
                    else:
                        result[current_section].append(item_val.strip('"').strip("'"))
                        continue

            if in_list:
                # Add to current list
                list_items.append(item_val.strip('"').strip("'"))
            elif current_item is not None:
                # Start a new list
                in_list = True
                # Find what key this list belongs to - it's the last key-only entry
                # Actually, just store with the last key
                list_items = [item_val.strip('"').strip("'")]
            continue

        # Key: Value or Key:
        colon_idx = stripped.find(':')
        if colon_idx > 0:
            key = stripped[:colon_idx].strip().strip('"').strip("'")
            value = stripped[colon_idx + 1:].strip()

            # Handle list items that look like key:value in list context
            _flush_list()

            if not value:
                # Section or sub-section header
                if current_section is None or key in _LIKELY_TOP_KEYS:
                    _flush_section()
                    current_section = key
                elif current_item is None:
                    # Item within current section
                    current_item = {}
                    current_item_key = key
                else:
                    # Sub-section within an item (like patterns:, levels:, etc.)
                    # Store as a key that will be populated by subsequent list/properties
                    in_list = True
                    list_key = key
                    list_items = []
            elif value in ('|', '>'):
                # Block scalar
                if current_item is None:
                    current_item = {}
                    current_item_key = key
                    in_block_scalar = True
                    block_key = '__content__'
                    block_lines = []
                else:
                    in_block_scalar = True
                    block_key = key
                    block_lines = []
            else:
                # Regular property
                if current_item is None:
                    if current_section:
                        current_item = {}
                        current_item_key = '__direct__'
                    else:
                        result[key] = _yaml_value(value)
                        continue
                current_item[key] = _yaml_value(value)
            continue

    # Flush remaining
    _flush_section()

    # Post-process: merge __direct__ items into section
    for section_key, section_val in list(result.items()):
        if isinstance(section_val, dict) and '__direct__' in section_val:
            direct = section_val.pop('__direct__')
            if isinstance(direct, dict):
                section_val.update(direct)

    # Handle sections that were stored as lists with __list_item__ entries
    for section_key, section_val in list(result.items()):
        if isinstance(section_val, dict):
            list_entries = []
            regular_entries = {}
            for k, v in section_val.items():
                if k == '__list_item__' and isinstance(v, dict):
                    list_entries.append(v)
                else:
                    regular_entries[k] = v
            if list_entries and not regular_entries:
                result[section_key] = list_entries

    return result if result else None


# Keys commonly found at the top level of knowledge files
_LIKELY_TOP_KEYS = {
    'metadata', 'metrics', 'time_filters', 'exclusion_rules', 'oracle_syntax',
    'query_templates', 'question_types', 'columns', 'column_synonyms',
    'column_mappings', 'concepts', 'hierarchies', 'valid_combinations',
    'constraints', 'entity_mappings', 'cardinality_summary', 'filter_hints',
    'time_hierarchies', 'examples', 'entities', 'relationships', 'relations',
    'view_name',
}


def _yaml_value(val_str: str):
    """Convert a YAML value string to a Python type."""
    if not val_str:
        return None
    # Quoted string
    if (val_str.startswith('"') and val_str.endswith('"')) or \
       (val_str.startswith("'") and val_str.endswith("'")):
        return val_str[1:-1]
    # List
    if val_str.startswith('[') and val_str.endswith(']'):
        inner = val_str[1:-1].strip()
        if not inner:
            return []
        items = []
        for item in re.split(r',\s*', inner):
            item = item.strip().strip('"').strip("'")
            if item:
                items.append(item)
        return items
    # Boolean
    if val_str.lower() in ('true', 'yes'):
        return True
    if val_str.lower() in ('false', 'no'):
        return False
    # Null
    if val_str.lower() in ('null', 'none', '~'):
        return None
    # Number
    try:
        if '.' in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        pass
    return val_str


# ---------------------------------------------------------------------------
# Intent classification helpers
# ---------------------------------------------------------------------------

def _build_intent_index(context_graph: dict) -> dict:
    """Build a token-based index from context graph question_types for fast matching."""
    index = {}
    question_types = context_graph.get("question_types", {})
    for intent_name, intent_def in question_types.items():
        patterns = intent_def.get("patterns", [])
        tokens = set()
        for p in patterns:
            for word in re.split(r"\s+", p.lower()):
                word = word.strip(".,?!'\"")
                if len(word) > 1:
                    tokens.add(word)
        index[intent_name] = {
            "tokens": tokens,
            "definition": intent_def,
        }
    return index


# ---------------------------------------------------------------------------
# Query normalizer
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "ytd": "year to date",
    "yoy": "year over year",
    "mom": "month over month",
    "qty": "quantity",
    "amt": "amount",
    "avg": "average",
    "mfg": "manufacturing",
    "mgmt": "management",
    "dept": "department",
    "org": "organization",
    "fy": "fiscal year",
    "q1": "quarter 1",
    "q2": "quarter 2",
    "q3": "quarter 3",
    "q4": "quarter 4",
}

_FILLER_PATTERNS = re.compile(
    r"\b(please|can you|could you|show me|i want to see|i need|"
    r"i would like|tell me|give me|display|find me|help me)\b",
    re.IGNORECASE,
)


def normalize_query(raw_query: str, knowledge: dict) -> dict:
    """Normalize a user query before pipeline processing.

    Steps:
    1. Expand abbreviations (ytd, yoy, qty, etc.)
    2. Resolve entity aliases (germany -> Germany and Eastern Europe)
    3. Remove filler words
    4. Extract numbers (for TOP N patterns)

    Returns dict with:
        normalized_query, expansions, alias_resolutions, extracted_numbers
    """
    text = raw_query.strip()
    expansions = []
    alias_resolutions = []

    # 1. Expand abbreviations
    for abbr, full in _ABBREVIATIONS.items():
        pattern = re.compile(r"\b" + re.escape(abbr) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(full, text)
            expansions.append({"from": abbr, "to": full})

    # 2. Resolve entity aliases
    entity_aliases = knowledge.get("entity_aliases", {})
    if entity_aliases:
        text_lower = text.lower()
        # Sort by length descending to match longest aliases first
        sorted_aliases = sorted(entity_aliases.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            if alias in text_lower:
                info = entity_aliases[alias]
                alias_resolutions.append({
                    "alias": alias,
                    "type": info.get("type", ""),
                    "canonical_value": info.get("canonical_value", ""),
                    "sql_filter": info.get("sql_filter", ""),
                })

    # 3. Extract numbers (for TOP N)
    extracted_numbers = [int(m) for m in re.findall(r"\b(\d+)\b", text)]

    # 4. Remove filler words (but keep the cleaned query readable)
    cleaned = _FILLER_PATTERNS.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # If cleaning removed everything meaningful, fall back to original
    if len(cleaned) < 3:
        cleaned = text

    return {
        "normalized_query": cleaned,
        "original_query": raw_query,
        "expansions": expansions,
        "alias_resolutions": alias_resolutions,
        "extracted_numbers": extracted_numbers,
    }


# ---------------------------------------------------------------------------
# Intent classification (upgraded: 15 intent types, confidence levels)
# ---------------------------------------------------------------------------

# Specificity hierarchy: more specific intents override less specific ones
_INTENT_SPECIFICITY = {
    "top_n": 10,
    "bottom_n": 10,
    "top_n_by_partition": 10,
    "percentile_threshold": 9,
    "cumulative_sum": 9,
    "comparison": 8,
    "trend": 8,
    "time_series": 7,
    "average": 6,
    "count": 6,
    "enumerate": 6,
    "aggregation_grouped": 5,
    "aggregation": 4,
    "filter": 3,
    "off_topic": 0,
}

# Built-in pattern matching for core intents (supplements context graph)
_BUILTIN_INTENT_PATTERNS = {
    "enumerate": [r"\blist\b", r"\bshow all\b", r"\bwhat are\b", r"\bwhich\b", r"\bdistinct\b"],
    "top_n": [r"\btop\s+\d+\b", r"\bhighest\s+\d+\b", r"\blargest\s+\d+\b", r"\bbiggest\s+\d+\b"],
    "bottom_n": [r"\bbottom\s+\d+\b", r"\blowest\s+\d+\b", r"\bsmallest\s+\d+\b"],
    "count": [r"\bhow many\b", r"\bnumber of\b", r"\bcount\b"],
    "average": [r"\baverage\b", r"\bavg\b", r"\bmean\b"],
    "time_series": [r"\bmonthly\b", r"\bquarterly\b", r"\bweekly\b", r"\bdaily\b", r"\bby month\b", r"\bby quarter\b"],
    "trend": [r"\btrend\b", r"\bover time\b", r"\byear over year\b", r"\bgrowth\b", r"\byoy\b"],
    "comparison": [r"\bcompare\b", r"\bvs\b", r"\bversus\b", r"\bdifference\b", r"\bvariance\b"],
    "aggregation_grouped": [r"\bby\s+\w+\b", r"\bper\s+\w+\b", r"\bfor each\b"],
    "aggregation": [r"\btotal\b", r"\bsum\b", r"\boverall\b", r"\bspend\b"],
    "filter": [r"\bexcluding\b", r"\bexcept\b", r"\bwithout\b", r"\babove\b", r"\bbelow\b"],
}


def classify_intent(user_query: str, intent_index: dict) -> dict:
    """Classify user query intent using pattern matching + context graph.

    Supports 15 intent types with specificity-based priority and confidence levels.
    Returns dict with primary_intent, secondary_intents, matched_template,
    confidence, confidence_level, matched_phrases.
    """
    query_lower = user_query.lower()
    query_tokens = set()
    for word in re.split(r"\s+", query_lower):
        word = word.strip(".,?!'\"")
        if len(word) > 1:
            query_tokens.add(word)

    scores = []
    matched_phrases = []

    # 1. Score from context graph index (token overlap)
    for intent_name, intent_data in intent_index.items():
        overlap = len(query_tokens & intent_data["tokens"])
        if overlap > 0:
            token_count = max(len(intent_data["tokens"]), 1)
            score = overlap / token_count
            scores.append((intent_name, score, intent_data["definition"]))

    # 2. Score from built-in patterns (regex)
    for intent_name, patterns in _BUILTIN_INTENT_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, query_lower)
            if m:
                matched_phrases.append({"intent": intent_name, "phrase": m.group()})
                # Check if this intent already has a score; boost it
                found = False
                for i, (name, score, defn) in enumerate(scores):
                    if name == intent_name:
                        scores[i] = (name, min(score + 0.3, 1.0), defn)
                        found = True
                        break
                if not found:
                    scores.append((intent_name, 0.4, {}))
                break  # One match per intent is enough

    # 3. Apply specificity: if both generic and specific match, prefer specific
    if len(scores) > 1:
        for i, (name, score, defn) in enumerate(scores):
            specificity = _INTENT_SPECIFICITY.get(name, 0)
            # Normalize specificity to 0-0.3 bonus
            scores[i] = (name, score + specificity * 0.03, defn)

    scores.sort(key=lambda x: x[1], reverse=True)

    if not scores:
        return {
            "primary_intent": "unknown",
            "secondary_intents": [],
            "matched_template": None,
            "confidence": 0.0,
            "confidence_level": "low",
            "matched_phrases": [],
        }

    primary = scores[0]
    secondary = [s[0] for s in scores[1:4]]
    confidence = round(min(primary[1], 1.0), 3)

    # Determine confidence level
    if confidence >= 0.6:
        confidence_level = "high"
    elif confidence >= 0.3:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    return {
        "primary_intent": primary[0],
        "secondary_intents": secondary,
        "matched_template": primary[2].get("sql_template") if isinstance(primary[2], dict) else None,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "matched_phrases": matched_phrases,
    }


def select_relevant_examples(
    intent_result: dict,
    detected_entities: list,
    all_examples: list,
    max_examples: int = 10,
) -> list:
    """Select the most relevant few-shot examples based on intent and entity overlap."""
    if not all_examples:
        return []
    if max_examples <= 0 or max_examples >= len(all_examples):
        return all_examples

    primary_intent = intent_result.get("primary_intent", "")
    secondary_intents = set(intent_result.get("secondary_intents", []))
    entity_set = {e.lower() for e in detected_entities}

    scored = []
    for ex in all_examples:
        score = 0.0
        q_lower = (ex.get("question") or ex.get("input", "")).lower()
        sql_lower = (ex.get("sql") or ex.get("output", "")).lower()
        tags = {t.lower() for t in ex.get("tags", [])}
        category = (ex.get("category", "") or "").lower()

        # Intent matching
        if primary_intent and primary_intent.lower() in (category or tags or q_lower):
            score += 3
        for si in secondary_intents:
            if si.lower() in q_lower or si.lower() in tags:
                score += 1

        # Entity matching
        for entity in entity_set:
            if entity in q_lower or entity in sql_lower:
                score += 2

        # Complexity matching (prefer similar complexity)
        complexity = ex.get("complexity", 1)
        if isinstance(complexity, (int, float)):
            score += max(0, 1 - abs(complexity - 2) * 0.3)

        scored.append((score, ex))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:max_examples]]


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

class KnowledgeLayerComponent(Node):
    """Ingests knowledge artifacts and produces a unified knowledge context.

    Connect a single Knowledge Base containing all your knowledge files.
    Each file is auto-detected by its content structure — no fixed naming
    conventions required.  The output is a structured Data object that
    downstream NL-to-SQL components consume for high-accuracy SQL generation.
    """

    display_name = "Knowledge Layer"
    description = (
        "Ingests knowledge files (any combination of knowledge graph, ontology, "
        "semantic layer, synonyms, business rules, examples, domain terms, etc.) "
        "and produces a unified knowledge context for NL-to-SQL components. "
        "Files are auto-detected by content — no naming conventions needed."
    )
    icon = "brain"
    name = "KnowledgeLayer"

    inputs = [
        # === PRIMARY INPUT — single Knowledge Base with all files ===
        HandleInput(
            name="knowledge_files",
            display_name="Knowledge Files",
            input_types=["Message"],
            info=(
                "Connect a Knowledge Base component containing your knowledge files. "
                "Each file is auto-detected by its content structure (not filename). "
                "Recognised types: knowledge graph, ontology, semantic layer, context graph, "
                "synonyms, business rules, few-shot examples, domain terms, schema columns."
            ),
            required=False,
        ),

        # === INDIVIDUAL FILE INPUTS (advanced — for overriding specific slots) ===
        HandleInput(
            name="knowledge_graph_file",
            display_name="Knowledge Graph",
            input_types=["Data"],
            info="Knowledge Graph file — entities, relationships, hierarchies",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="ontology_file",
            display_name="Ontology",
            input_types=["Data"],
            info="Ontology file — hierarchies, valid combinations, constraints",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="semantic_layer_file",
            display_name="Semantic Layer",
            input_types=["Data"],
            info="Semantic Layer file — column metadata, cardinality, entity mappings",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="context_graph_file",
            display_name="Context Graph",
            input_types=["Data"],
            info="Context Graph file — question patterns → SQL patterns",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="synonyms_file",
            display_name="Synonyms Dictionary",
            input_types=["Data"],
            info="Synonyms file — column ↔ natural language term mappings",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="business_rules_file",
            display_name="Business Rules",
            input_types=["Data"],
            info="Business Rules file — metrics, exclusions, time filters, DB syntax",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="examples_file",
            display_name="Few-Shot Examples",
            input_types=["Data"],
            info="Examples file — question → SQL pairs for in-context learning",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="domain_terms_file",
            display_name="Domain Terms",
            input_types=["Data"],
            info="Domain-specific term translations (e.g. German/SAP terms, abbreviations)",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="schema_columns_file",
            display_name="Schema Columns",
            input_types=["Data"],
            info="Column definitions with types, categories, descriptions",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="sql_templates_file",
            display_name="SQL Templates",
            input_types=["Data"],
            info="Parameterized SQL templates for common query patterns (top_n, time_series, etc.)",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="anti_patterns_file",
            display_name="Anti-Patterns",
            input_types=["Data"],
            info="SQL anti-patterns with regex detection and auto-fix rules",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="column_values_file",
            display_name="Column Values",
            input_types=["Data"],
            info="Actual column value distributions with frequencies and tiers",
            required=False,
            advanced=True,
        ),
        HandleInput(
            name="entities_aliases_file",
            display_name="Entity Aliases",
            input_types=["Data"],
            info="Entity value aliases — region, country, business concept, OEM, commodity mappings",
            required=False,
            advanced=True,
        ),

        # === MANUAL OVERRIDES ===
        MultilineInput(
            name="additional_business_rules",
            display_name="Additional Business Rules",
            info="Extra business rules to append (free text).",
            value="",
            advanced=True,
        ),
        MultilineInput(
            name="additional_domain_context",
            display_name="Additional Domain Context",
            info="Extra domain context to append.",
            value="",
            advanced=True,
        ),
        TableInput(
            name="additional_synonyms",
            display_name="Additional Synonyms",
            info="Extra synonym mappings to append.",
            table_schema=[
                {"name": "column_name", "display_name": "Column Name", "type": "str"},
                {"name": "synonyms", "display_name": "Synonyms (comma-separated)", "type": "str"},
            ],
            value=[],
            advanced=True,
        ),
        TableInput(
            name="additional_examples",
            display_name="Additional Examples",
            info="Extra few-shot examples to append.",
            table_schema=[
                {"name": "question", "display_name": "Question", "type": "str"},
                {"name": "sql", "display_name": "SQL Query", "type": "str"},
            ],
            value=[],
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Knowledge Context",
            name="knowledge_context",
            method="build_knowledge_context",
            types=["Data"],
        ),
        Output(
            display_name="Summary",
            name="summary",
            method="build_summary",
            types=["Message"],
        ),
    ]

    def _parse_input(self, attr_name: str) -> Optional[Union[dict, list]]:
        """Parse a file input into structured data."""
        raw = getattr(self, attr_name, None)
        if raw is None:
            return None
        parsed = _safe_parse(raw)
        if parsed is None:
            logger.debug(f"KnowledgeLayer: could not parse input '{attr_name}'")
        return parsed

    def _load_bundle(self) -> None:
        """Auto-route files from the bundled Knowledge Base input.

        The Knowledge Base component has two output modes:
        1. "Knowledge Base" output — Message with file paths (one per line)
        2. "Raw Content" output — Message with file contents concatenated

        This method handles BOTH modes:
        - If lines look like file paths, read and parse each file
        - Otherwise, treat the entire text as concatenated file contents and
          split by detecting document boundaries (YAML front-matter, JSON objects)

        Only populates slots that are not already connected via individual
        HandleInputs.
        """
        bundle = getattr(self, "knowledge_files", None)
        if bundle is None:
            logger.warning("KnowledgeLayer bundle: 'knowledge_files' input is None — not connected?")
            return

        # Log what we received
        logger.info(f"KnowledgeLayer bundle: received type={type(bundle).__name__}, repr={repr(bundle)[:200]}")

        # Extract text from the Message / Data / str / list
        text = ""
        if isinstance(bundle, Message):
            text = bundle.text or ""
        elif isinstance(bundle, str):
            text = bundle
        elif isinstance(bundle, Data) and isinstance(bundle.data, dict):
            text = bundle.data.get("text", "") or ""
        elif isinstance(bundle, list):
            # Knowledge Base may send a list of Data or Message objects
            logger.info(f"KnowledgeLayer bundle: received list of {len(bundle)} items")
            parts = []
            for item in bundle:
                if isinstance(item, Message):
                    parts.append(item.text or "")
                elif isinstance(item, Data):
                    if isinstance(item.data, dict):
                        parts.append(item.data.get("text", "") or item.data.get("content", "") or item.data.get("file_content", "") or "")
                    elif isinstance(item.data, str):
                        parts.append(item.data)
                elif isinstance(item, str):
                    parts.append(item)
            text = "\n\n".join(p for p in parts if p)
        else:
            logger.warning(f"KnowledgeLayer bundle: unexpected type {type(bundle).__name__}: {repr(bundle)[:200]}")

        if not text.strip():
            logger.warning("KnowledgeLayer bundle: received empty text from Knowledge Base")
            return

        logger.info(f"KnowledgeLayer bundle: received text of length {len(text)}")

        lines = [p.strip() for p in text.strip().splitlines() if p.strip()]

        # Heuristic: if first few non-empty lines look like file paths, use path mode
        sample_lines = lines[:3]
        looks_like_paths = all(
            (Path(line).suffix.lower() in ('.yaml', '.yml', '.json', '.txt', '.csv')
             and ('/' in line or '\\' in line))
            for line in sample_lines
        ) if sample_lines else False

        if looks_like_paths:
            self._load_bundle_from_paths(lines)
        else:
            self._load_bundle_from_content(text)

    def _load_bundle_from_paths(self, paths: list) -> None:
        """Load knowledge files by reading file paths."""
        routed = 0
        for file_path_str in paths:
            file_path = Path(file_path_str)

            # Read and parse the file
            try:
                raw_text = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"KnowledgeLayer bundle: failed to read '{file_path_str}': {e}")
                continue

            parsed = _parse_text(raw_text)
            if parsed is None:
                logger.warning(f"KnowledgeLayer bundle: could not parse '{file_path.name}'")
                continue

            # Detect artifact type by content first, filename as fallback
            target_slot = _detect_artifact_type(parsed, file_path.name)
            logger.info(
                f"KnowledgeLayer bundle: '{file_path.name}' detected as "
                f"'{target_slot}', parsed type={type(parsed).__name__}, "
                f"keys={list(parsed.keys())[:8] if isinstance(parsed, dict) else 'N/A'}"
            )

            if target_slot is None:
                logger.warning(f"KnowledgeLayer bundle: unrecognised file '{file_path.name}' — skipped")
                continue

            # Skip if the individual input already has real data
            # (framework initialises unconnected HandleInputs to "" so check for truthy)
            existing = getattr(self, target_slot, None)
            if existing is not None and existing != "" and existing != []:
                logger.warning(
                    f"KnowledgeLayer bundle: slot '{target_slot}' already has data "
                    f"(type={type(existing).__name__}), skipping '{file_path.name}'"
                )
                continue

            setattr(self, target_slot, Data(data={"text": raw_text}))
            routed += 1
            logger.info(f"KnowledgeLayer bundle: '{file_path.name}' → {target_slot}")

        if routed:
            logger.info(f"KnowledgeLayer bundle: auto-routed {routed}/{len(paths)} files from paths")
        else:
            logger.warning(f"KnowledgeLayer bundle: could not route any of {len(paths)} file paths")

    def _load_bundle_from_content(self, text: str) -> None:
        """Load knowledge files from concatenated raw content.

        The Knowledge Base 'Raw Content' output concatenates all file contents
        with a double-newline separator.  We split the blob back into individual
        documents by detecting JSON object boundaries and YAML document markers,
        then auto-detect each chunk's artifact type.
        """
        # Split concatenated content into individual documents
        chunks = self._split_concatenated_content(text)
        logger.info(f"KnowledgeLayer bundle: split content into {len(chunks)} chunks")

        routed = 0
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue

            parsed = _parse_text(chunk)
            if parsed is None:
                logger.warning(f"KnowledgeLayer bundle: could not parse chunk {i + 1}")
                continue

            target_slot = _detect_artifact_type(parsed, "")
            if target_slot is None:
                logger.debug(f"KnowledgeLayer bundle: unrecognised chunk {i + 1} — skipped")
                continue

            existing = getattr(self, target_slot, None)
            if existing is not None and existing != "" and existing != []:
                logger.warning(f"KnowledgeLayer bundle: slot '{target_slot}' already filled, skipping chunk {i + 1}")
                continue

            setattr(self, target_slot, Data(data={"text": chunk}))
            routed += 1
            logger.info(f"KnowledgeLayer bundle: chunk {i + 1} → {target_slot}")

        if routed:
            logger.info(f"KnowledgeLayer bundle: auto-routed {routed}/{len(chunks)} chunks from content")
        else:
            logger.warning(f"KnowledgeLayer bundle: could not route any of {len(chunks)} chunks")

    def _split_concatenated_content(self, text: str) -> list:
        """Split concatenated file contents back into individual documents.

        Handles JSON objects (start with '{') and YAML documents (separated
        by blank lines where a new top-level key begins, or by '---' markers).
        """
        chunks = []

        # First, try splitting by double-newline separator (the Knowledge Base default)
        # But we need smarter splitting because YAML docs may contain blank lines

        # Strategy: detect JSON blobs first (they start with '{' at the start of a line)
        remaining = text

        # Extract JSON objects first
        json_pattern = re.compile(r'^\{', re.MULTILINE)
        json_starts = [m.start() for m in json_pattern.finditer(remaining)]

        if json_starts:
            # Find matching closing brace for each JSON start
            prev_end = 0
            for js in json_starts:
                # Add any YAML content before this JSON
                if js > prev_end:
                    yaml_part = remaining[prev_end:js].strip()
                    if yaml_part:
                        chunks.extend(self._split_yaml_documents(yaml_part))

                # Find the end of this JSON object by counting braces
                depth = 0
                json_end = js
                for ci, ch in enumerate(remaining[js:], start=js):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            json_end = ci + 1
                            break
                chunks.append(remaining[js:json_end])
                prev_end = json_end

            # Any remaining content after last JSON
            if prev_end < len(remaining):
                yaml_part = remaining[prev_end:].strip()
                if yaml_part:
                    chunks.extend(self._split_yaml_documents(yaml_part))
        else:
            # All YAML content
            chunks.extend(self._split_yaml_documents(remaining))

        return chunks

    def _split_yaml_documents(self, text: str) -> list:
        """Split YAML text into individual documents.

        Uses '---' markers and blank-line + top-level-key patterns.
        """
        # Try explicit YAML document markers first
        if '\n---\n' in text or text.startswith('---\n'):
            docs = re.split(r'\n---\n|^---\n', text)
            return [d.strip() for d in docs if d.strip()]

        # Try splitting by double-newline followed by a comment or top-level key
        # This handles the Knowledge Base's default \n\n separator
        parts = re.split(r'\n\n(?=#\s|[A-Za-z_][A-Za-z0-9_]*:\s*\n)', text)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]

        # Can't split — return as single document
        return [text] if text.strip() else []

    def _build_synonym_map(self, synonyms_data: Optional[dict], semantic_layer: Optional[dict],
                           domain_terms: Optional[dict]) -> dict:
        """Build unified synonym map from all sources."""
        synonym_map = {}

        # 1. From synonyms file — column_synonyms section
        if synonyms_data:
            col_syns = synonyms_data.get("column_synonyms", synonyms_data)
            if isinstance(col_syns, dict):
                for col_name, syn_info in col_syns.items():
                    if isinstance(syn_info, list):
                        synonyms_list = syn_info
                    elif isinstance(syn_info, dict):
                        synonyms_list = syn_info.get("synonyms", [])
                    else:
                        continue
                    for s in synonyms_list:
                        s_lower = str(s).lower().strip()
                        if s_lower:
                            synonym_map[s_lower] = {"column": col_name, "source": "synonyms_file"}

        # 2. From semantic layer — per-column synonyms
        if semantic_layer:
            columns = semantic_layer.get("columns", {})
            for col_name, col_info in columns.items():
                if isinstance(col_info, dict):
                    for s in col_info.get("synonyms", []):
                        s_lower = str(s).lower().strip()
                        if s_lower and s_lower not in synonym_map:
                            synonym_map[s_lower] = {
                                "column": col_name,
                                "entity": col_info.get("entity", ""),
                                "type": col_info.get("type", ""),
                                "source": "semantic_layer",
                            }

        # 3. From domain terms (German/SAP terms, abbreviations, translations, etc.)
        if domain_terms:
            terms = domain_terms.get("column_mappings", domain_terms)
            if isinstance(terms, dict):
                for col_name, term_info in terms.items():
                    if isinstance(term_info, dict):
                        # Add all translations
                        for s in term_info.get("translations", []):
                            s_lower = str(s).lower().strip()
                            if s_lower and s_lower not in synonym_map:
                                synonym_map[s_lower] = {"column": col_name, "source": "domain_terms"}
                        # Add the native term itself (e.g. German, French, etc.)
                        for key in ("german", "native", "term", "abbreviation"):
                            native = term_info.get(key, "")
                            if native:
                                synonym_map[native.lower().strip()] = {"column": col_name, "source": "domain_terms"}
                    elif isinstance(term_info, str):
                        synonym_map[term_info.lower().strip()] = {"column": col_name, "source": "domain_terms"}

        # 4. Manual overrides
        if self.additional_synonyms:
            for row in self.additional_synonyms:
                if isinstance(row, dict) and row.get("column_name") and row.get("synonyms"):
                    for s in row["synonyms"].split(","):
                        s_lower = s.strip().lower()
                        if s_lower:
                            synonym_map[s_lower] = {"column": row["column_name"], "source": "manual"}

        return synonym_map

    def _build_entities(self, knowledge_graph: Optional[dict], entities_data: Optional[dict],
                        semantic_layer: Optional[dict]) -> dict:
        """Build unified entity definitions from knowledge graph, entities.yaml, and semantic layer."""
        entities = {}

        # From knowledge_graph.json
        if knowledge_graph:
            kg_entities = knowledge_graph.get("entities", {})
            for name, info in kg_entities.items():
                if isinstance(info, dict):
                    entities[name] = {
                        "type": info.get("type", "dimension"),
                        "primary_key": info.get("primary_key", ""),
                        "display_column": info.get("display_column", ""),
                        "columns": info.get("columns", []),
                        "measures": info.get("measures", []),
                        "description": info.get("description", ""),
                    }

        # Enrich from entities.yaml
        if entities_data:
            ent_defs = entities_data.get("entities", entities_data)
            if isinstance(ent_defs, dict):
                for name, info in ent_defs.items():
                    if isinstance(info, dict):
                        if name not in entities:
                            entities[name] = {}
                        entities[name]["synonyms"] = info.get("synonyms", [])
                        entities[name]["usage_rules"] = info.get("usage_rules", [])
                        cols = info.get("columns", {})
                        if isinstance(cols, dict):
                            entities[name]["column_roles"] = cols

        # Enrich from semantic_layer entity_mappings
        if semantic_layer:
            mappings = semantic_layer.get("entity_mappings", {})
            for name, mapping in mappings.items():
                if isinstance(mapping, dict) and name in entities:
                    entities[name]["id_column"] = mapping.get("id_column", "")
                    entities[name]["display_column"] = mapping.get("display_column", entities[name].get("display_column", ""))

        return entities

    def _build_hierarchies(self, ontology: Optional[dict], knowledge_graph: Optional[dict]) -> dict:
        """Build hierarchy definitions from ontology and knowledge graph."""
        hierarchies = {}

        if ontology:
            ont_hierarchies = ontology.get("hierarchies", {})
            for name, h_info in ont_hierarchies.items():
                if isinstance(h_info, dict):
                    levels = []
                    for level in h_info.get("levels", []):
                        if isinstance(level, dict):
                            levels.append({
                                "level": level.get("level", 0),
                                "name": level.get("name", ""),
                                "column": level.get("column", ""),
                                "description_column": level.get("description_column", ""),
                            })
                    hierarchies[name] = {
                        "name": h_info.get("name", name),
                        "description": h_info.get("description", ""),
                        "levels": levels,
                        "drill_down": h_info.get("drill_down", True),
                        "roll_up": h_info.get("roll_up", True),
                    }

        return hierarchies

    def _build_business_rules(self, rules_data: Optional[dict]) -> dict:
        """Build business rules from business_rules.yaml."""
        rules = {
            "metrics": {},
            "exclusion_rules": [],
            "time_filters": {},
            "classification_rules": {},
            "oracle_syntax": {},
        }

        if not rules_data:
            return rules

        # Metrics
        metrics = rules_data.get("metrics", {})
        if isinstance(metrics, dict):
            for name, info in metrics.items():
                if isinstance(info, dict):
                    rules["metrics"][name] = info.get("sql", info.get("expression", str(info)))
                elif isinstance(info, str):
                    rules["metrics"][name] = info

        # Exclusion rules
        exclusions = rules_data.get("exclusion_rules", rules_data.get("exclusions", []))
        if isinstance(exclusions, dict):
            for name, info in exclusions.items():
                if isinstance(info, dict):
                    rules["exclusion_rules"].append(f"{name}: {info.get('condition', info.get('sql', ''))}")
                elif isinstance(info, str):
                    rules["exclusion_rules"].append(f"{name}: {info}")
        elif isinstance(exclusions, list):
            rules["exclusion_rules"] = [str(e) for e in exclusions]

        # Time filters
        time_filters = rules_data.get("time_filters", {})
        if isinstance(time_filters, dict):
            rules["time_filters"] = {k: (v.get("sql", str(v)) if isinstance(v, dict) else str(v))
                                     for k, v in time_filters.items()}

        # Oracle syntax
        oracle = rules_data.get("oracle_syntax", rules_data.get("oracle_specific", {}))
        if isinstance(oracle, dict):
            rules["oracle_syntax"] = {k: str(v) for k, v in oracle.items()}

        return rules

    def _build_column_value_hints(self, semantic_layer: Optional[dict]) -> dict:
        """Build column value hints from semantic layer cardinality info."""
        hints = {}

        if not semantic_layer:
            return hints

        columns = semantic_layer.get("columns", {})
        cardinality_summary = semantic_layer.get("cardinality_summary", {})

        # From per-column cardinality_info
        for col_name, col_info in columns.items():
            if isinstance(col_info, dict):
                card_info = col_info.get("cardinality_info", {})
                if isinstance(card_info, dict) and card_info.get("unique_values"):
                    unique = card_info.get("unique_values", 0)
                    if unique < 200:  # Only include low/medium cardinality
                        hints[col_name] = {
                            "cardinality": "low" if unique < 50 else "medium",
                            "unique_values": unique,
                            "examples": card_info.get("examples", []),
                            "description": col_info.get("description", ""),
                        }

        # From filter_hints section
        filter_hints = semantic_layer.get("filter_hints", {})
        if isinstance(filter_hints, dict):
            for col_name, hint_text in filter_hints.items():
                if col_name in hints:
                    hints[col_name]["filter_hint"] = str(hint_text)
                elif isinstance(hint_text, str):
                    hints[col_name] = {"filter_hint": hint_text}

        return hints

    def _build_column_metadata(self, schema_columns: Optional[dict], semantic_layer: Optional[dict]) -> dict:
        """Build column metadata from schema columns and semantic layer."""
        columns = {}

        if schema_columns:
            col_defs = schema_columns.get("columns", schema_columns)
            if isinstance(col_defs, dict):
                for col_name, info in col_defs.items():
                    if isinstance(info, dict):
                        columns[col_name] = {
                            "type": info.get("type", ""),
                            "category": info.get("category", ""),
                            "description": info.get("description", ""),
                            "nullable": info.get("nullable", True),
                        }

        # Enrich with semantic layer
        if semantic_layer:
            sl_columns = semantic_layer.get("columns", {})
            for col_name, info in sl_columns.items():
                if isinstance(info, dict):
                    if col_name not in columns:
                        columns[col_name] = {}
                    columns[col_name]["entity"] = info.get("entity", "")
                    columns[col_name]["semantic_type"] = info.get("type", "")
                    columns[col_name]["synonyms"] = info.get("synonyms", [])
                    if not columns[col_name].get("description"):
                        columns[col_name]["description"] = info.get("description", "")

        return columns

    def _build_sql_templates(self, templates_data: Optional[dict]) -> dict:
        """Build normalized SQL templates from sql_templates file."""
        templates = {}
        if not templates_data:
            return templates

        raw_templates = templates_data.get("templates", templates_data)
        if not isinstance(raw_templates, dict):
            return templates

        for name, info in raw_templates.items():
            if not isinstance(info, dict):
                continue
            templates[name] = {
                "patterns": info.get("patterns", []),
                "template": info.get("template", info.get("sql_template", "")),
                "description": info.get("description", ""),
                "parameters": info.get("parameters", []),
            }

        return templates

    def _build_anti_patterns(self, anti_patterns_data: Optional[dict]) -> list:
        """Build anti-pattern rules with pre-compiled regexes."""
        patterns = []
        if not anti_patterns_data:
            return patterns

        raw_patterns = anti_patterns_data.get("anti_patterns", [])
        if not isinstance(raw_patterns, list):
            return patterns

        for ap in raw_patterns:
            if not isinstance(ap, dict):
                continue
            regex_str = ap.get("pattern", "")
            compiled = None
            if regex_str:
                try:
                    compiled = re.compile(regex_str, re.IGNORECASE)
                except re.error:
                    logger.warning(f"KnowledgeLayer: invalid anti-pattern regex: {regex_str}")
            patterns.append({
                "id": ap.get("id", ""),
                "name": ap.get("name", ""),
                "pattern": regex_str,
                "compiled": compiled,
                "severity": ap.get("severity", "warning"),
                "description": ap.get("description", ""),
                "fix": ap.get("fix", ""),
                "example_bad": ap.get("example_bad", ""),
                "example_good": ap.get("example_good", ""),
            })

        # Also include validation rules if present
        validation_rules = anti_patterns_data.get("validation_rules", [])
        if isinstance(validation_rules, list):
            for vr in validation_rules:
                if isinstance(vr, dict):
                    regex_str = vr.get("pattern", "")
                    compiled = None
                    if regex_str:
                        try:
                            compiled = re.compile(regex_str, re.IGNORECASE)
                        except re.error:
                            pass
                    patterns.append({
                        "id": vr.get("id", ""),
                        "name": vr.get("name", ""),
                        "pattern": regex_str,
                        "compiled": compiled,
                        "severity": "error" if vr.get("required") else ("error" if vr.get("forbidden") else "info"),
                        "description": vr.get("description", ""),
                        "required": vr.get("required", False),
                        "forbidden": vr.get("forbidden", False),
                        "fix": "",
                        "example_bad": "",
                        "example_good": "",
                    })

        return patterns

    def _build_column_values(self, column_values_data: Optional[dict]) -> dict:
        """Build detailed column value distributions from column_values/histograms file."""
        result = {}
        if not column_values_data:
            return result

        columns = column_values_data.get("columns", {})
        if not isinstance(columns, dict):
            return result

        # Tier info from metadata
        tiers = column_values_data.get("columns_by_tier", {})
        tier_lookup = {}
        if isinstance(tiers, dict):
            for tier_name, cols in tiers.items():
                if isinstance(cols, list):
                    for col in cols:
                        tier_lookup[str(col)] = tier_name

        for col_name, col_info in columns.items():
            if not isinstance(col_info, dict):
                continue

            cardinality = col_info.get("cardinality", 0)
            tier = col_info.get("tier", tier_lookup.get(col_name, "UNKNOWN"))
            values = []

            raw_values = col_info.get("values", [])
            if isinstance(raw_values, list):
                for v in raw_values:
                    if isinstance(v, dict):
                        values.append({
                            "value": str(v.get("value", "")),
                            "frequency": v.get("frequency", 0),
                            "pct": v.get("pct_of_total", v.get("pct", 0)),
                        })

            result[col_name] = {
                "cardinality": cardinality,
                "tier": tier,
                "values": values,
            }

        return result

    def _build_entity_aliases(self, aliases_data: Optional[dict]) -> dict:
        """Build flattened entity alias lookup from entities_aliases file.

        Returns a dict mapping alias_string (lowercased) -> {type, canonical_value, sql_filter}.
        """
        aliases = {}
        if not aliases_data:
            return aliases

        # Region aliases: alias -> region name (filter on REGION column)
        region_aliases = aliases_data.get("region_aliases", {})
        if isinstance(region_aliases, dict):
            for alias, canonical in region_aliases.items():
                aliases[str(alias).lower().strip()] = {
                    "type": "region",
                    "canonical_value": str(canonical),
                    "sql_filter": f"REGION = '{canonical}'",
                }

        # Country aliases: alias -> country code (filter on COUNTRY column)
        country_aliases = aliases_data.get("country_aliases", {})
        if isinstance(country_aliases, dict):
            for alias, canonical in country_aliases.items():
                key = str(alias).lower().strip()
                if key not in aliases:  # don't overwrite region alias
                    aliases[key] = {
                        "type": "country",
                        "canonical_value": str(canonical),
                        "sql_filter": f"COUNTRY = '{canonical}'",
                    }

        # Business concepts: alias -> SQL filter expression
        business_concepts = aliases_data.get("business_concepts", {})
        if isinstance(business_concepts, dict):
            for alias, sql_filter in business_concepts.items():
                aliases[str(alias).lower().strip()] = {
                    "type": "business_concept",
                    "canonical_value": str(alias),
                    "sql_filter": str(sql_filter),
                }

        # OEM aliases: alias -> OEM value (filter on CUSTOMER column)
        oem_aliases = aliases_data.get("oem_aliases", {})
        if isinstance(oem_aliases, dict):
            for alias, canonical in oem_aliases.items():
                aliases[str(alias).lower().strip()] = {
                    "type": "oem",
                    "canonical_value": str(canonical),
                    "sql_filter": f"CUSTOMER = '{canonical}'",
                }

        # Commodity aliases: alias -> commodity name
        commodity_aliases = aliases_data.get("commodity_aliases", {})
        if isinstance(commodity_aliases, dict):
            for alias, canonical in commodity_aliases.items():
                aliases[str(alias).lower().strip()] = {
                    "type": "commodity",
                    "canonical_value": str(canonical),
                    "sql_filter": f"COMMODITY_DESCRIPTION = '{canonical}'",
                }

        return aliases

    def _index_examples(self, examples_data: Optional[Union[dict, list]]) -> list:
        """Parse and normalize few-shot examples."""
        examples = []

        if examples_data:
            # Handle list format
            if isinstance(examples_data, list):
                for ex in examples_data:
                    if isinstance(ex, dict):
                        examples.append({
                            "question": ex.get("question") or ex.get("input", ""),
                            "sql": ex.get("sql") or ex.get("output", ""),
                            "category": ex.get("category", ""),
                            "complexity": ex.get("complexity", 1),
                            "tags": ex.get("tags", []),
                        })
            # Handle dict format with examples list
            elif isinstance(examples_data, dict):
                ex_list = examples_data.get("examples", [])
                if isinstance(ex_list, list):
                    for ex in ex_list:
                        if isinstance(ex, dict):
                            examples.append({
                                "question": ex.get("question") or ex.get("input", ""),
                                "sql": ex.get("sql") or ex.get("output", ""),
                                "category": ex.get("category", ""),
                                "complexity": ex.get("complexity", 1),
                                "tags": ex.get("tags", []),
                            })

        # Add manual examples
        if self.additional_examples:
            for row in self.additional_examples:
                if isinstance(row, dict) and row.get("question") and row.get("sql"):
                    examples.append({
                        "question": row["question"],
                        "sql": row["sql"],
                        "category": "manual",
                        "complexity": 1,
                        "tags": [],
                    })

        return examples

    def _build_intent_patterns(self, context_graph: Optional[dict]) -> dict:
        """Build intent patterns from context graph."""
        if not context_graph:
            return {}

        patterns = {}
        question_types = context_graph.get("question_types", {})
        for intent_name, intent_def in question_types.items():
            if isinstance(intent_def, dict):
                patterns[intent_name] = {
                    "patterns": intent_def.get("patterns", []),
                    "measure": intent_def.get("measure"),
                    "aggregation": intent_def.get("aggregation"),
                    "group_by": intent_def.get("group_by"),
                    "filter_column": intent_def.get("filter_column"),
                    "order": intent_def.get("order"),
                }

        # Include query templates
        templates = context_graph.get("query_templates", {})
        return {
            "question_types": patterns,
            "query_templates": templates,
        }

    def _build_valid_combinations(self, ontology: Optional[dict]) -> dict:
        """Extract valid measure-dimension combinations from ontology."""
        if not ontology:
            return {}

        combos = ontology.get("valid_combinations", {})
        constraints = ontology.get("constraints", {})

        return {
            "valid_combinations": combos if isinstance(combos, dict) else {},
            "constraints": constraints if isinstance(constraints, dict) else {},
        }

    def build_knowledge_context(self) -> Data:
        """Build the unified knowledge context from all input files.

        FUSION LAYER: Combines mechanized + business knowledge into a unified context.
        """
        # Auto-route bundled Knowledge Base files (if connected)
        self._load_bundle()

        # Debug: log what inputs are populated
        input_slots = [
            "knowledge_graph_file", "ontology_file", "semantic_layer_file",
            "context_graph_file", "synonyms_file", "business_rules_file",
            "examples_file", "domain_terms_file", "schema_columns_file",
            "sql_templates_file", "anti_patterns_file", "column_values_file",
            "entities_aliases_file",
        ]
        for slot in input_slots:
            val = getattr(self, slot, None)
            if val is not None:
                val_type = type(val).__name__
                val_preview = str(val)[:80] if val else "empty"
                logger.info(f"KnowledgeLayer: slot '{slot}' has {val_type}: {val_preview}")
            else:
                logger.debug(f"KnowledgeLayer: slot '{slot}' is None")

        # Parse all inputs
        knowledge_graph = self._parse_input("knowledge_graph_file")
        ontology = self._parse_input("ontology_file")
        semantic_layer = self._parse_input("semantic_layer_file")
        context_graph = self._parse_input("context_graph_file")
        synonyms_data = self._parse_input("synonyms_file")
        business_rules_data = self._parse_input("business_rules_file")
        examples_data = self._parse_input("examples_file")
        domain_terms = self._parse_input("domain_terms_file")
        schema_columns = self._parse_input("schema_columns_file")
        sql_templates_data = self._parse_input("sql_templates_file")
        anti_patterns_data = self._parse_input("anti_patterns_file")
        column_values_data = self._parse_input("column_values_file")
        entities_aliases_data = self._parse_input("entities_aliases_file")

        # Count what we loaded
        all_inputs = [
            knowledge_graph, ontology, semantic_layer, context_graph,
            synonyms_data, business_rules_data, examples_data,
            domain_terms, schema_columns, sql_templates_data,
            anti_patterns_data, column_values_data, entities_aliases_data,
        ]
        loaded = sum(1 for x in all_inputs if x is not None)
        logger.info(f"KnowledgeLayer: loaded {loaded}/13 knowledge files")

        # Build unified structures
        synonym_map = self._build_synonym_map(synonyms_data, semantic_layer, domain_terms)
        entities = self._build_entities(knowledge_graph, None, semantic_layer)
        hierarchies = self._build_hierarchies(ontology, knowledge_graph)
        business_rules = self._build_business_rules(business_rules_data)
        column_value_hints = self._build_column_value_hints(semantic_layer)
        column_metadata = self._build_column_metadata(schema_columns, semantic_layer)
        examples = self._index_examples(examples_data)
        intent_patterns = self._build_intent_patterns(context_graph)
        valid_combinations = self._build_valid_combinations(ontology)
        sql_templates = self._build_sql_templates(sql_templates_data)
        anti_patterns = self._build_anti_patterns(anti_patterns_data)
        column_values_detailed = self._build_column_values(column_values_data)
        entity_aliases = self._build_entity_aliases(entities_aliases_data)

        # Build intent index for classification
        intent_index = {}
        if context_graph and isinstance(context_graph, dict):
            intent_index = _build_intent_index(context_graph)

        # Merge additional manual context
        additional_rules = self.additional_business_rules.strip() if self.additional_business_rules else ""
        additional_context = self.additional_domain_context.strip() if self.additional_domain_context else ""

        context = {
            "synonym_map": synonym_map,
            "entities": entities,
            "hierarchies": hierarchies,
            "business_rules": business_rules,
            "column_value_hints": column_value_hints,
            "column_metadata": column_metadata,
            "examples": examples,
            "intent_patterns": intent_patterns,
            "valid_combinations": valid_combinations,
            "intent_index": intent_index,
            "sql_templates": sql_templates,
            "anti_patterns": anti_patterns,
            "column_values_detailed": column_values_detailed,
            "entity_aliases": entity_aliases,
            "additional_business_rules": additional_rules,
            "additional_domain_context": additional_context,
            # Metadata
            "knowledge_files_loaded": loaded,
            "total_knowledge_slots": 13,
            "synonym_count": len(synonym_map),
            "entity_count": len(entities),
            "hierarchy_count": len(hierarchies),
            "example_count": len(examples),
            "column_count": len(column_metadata),
            "sql_template_count": len(sql_templates),
            "anti_pattern_count": len(anti_patterns),
            "column_values_count": len(column_values_detailed),
            "entity_alias_count": len(entity_aliases),
        }

        self.status = (
            f"Loaded: {loaded}/13 files, {len(synonym_map)} synonyms, "
            f"{len(entities)} entities, {len(examples)} examples, "
            f"{len(sql_templates)} templates, {len(anti_patterns)} anti-patterns, "
            f"{len(entity_aliases)} aliases"
        )

        return Data(data=context)

    def build_summary(self) -> Message:
        """Return a human-readable summary of the loaded knowledge."""
        ctx = self.build_knowledge_context()
        data = ctx.data

        parts = [
            "**Knowledge Layer Summary**\n",
            f"- Files loaded: **{data['knowledge_files_loaded']}** / {data.get('total_knowledge_slots', 13)}",
            f"- Synonyms: **{data['synonym_count']}** mappings",
            f"- Entities: **{data['entity_count']}** defined",
            f"- Hierarchies: **{data['hierarchy_count']}** defined",
            f"- Few-shot examples: **{data['example_count']}**",
            f"- Column metadata: **{data['column_count']}** columns",
            f"- SQL templates: **{data.get('sql_template_count', 0)}**",
            f"- Anti-patterns: **{data.get('anti_pattern_count', 0)}** rules",
            f"- Column value distributions: **{data.get('column_values_count', 0)}** columns",
            f"- Entity aliases: **{data.get('entity_alias_count', 0)}** mappings",
        ]

        # List entities
        if data["entities"]:
            parts.append("\n**Entities:**")
            for name, info in data["entities"].items():
                etype = info.get("type", "?")
                pk = info.get("primary_key", "?")
                display = info.get("display_column", "?")
                parts.append(f"  - {name} ({etype}): PK={pk}, Display={display}")

        # List hierarchies
        if data["hierarchies"]:
            parts.append("\n**Hierarchies:**")
            for name, info in data["hierarchies"].items():
                levels = " → ".join(l.get("name", l.get("column", "?")) for l in info.get("levels", []))
                parts.append(f"  - {info.get('name', name)}: {levels}")

        # Business rules summary
        rules = data.get("business_rules", {})
        if rules.get("metrics"):
            parts.append(f"\n**KPI Definitions:** {len(rules['metrics'])} metrics")
        if rules.get("exclusion_rules"):
            parts.append(f"**Exclusion Rules:** {len(rules['exclusion_rules'])} rules")
        if rules.get("oracle_syntax"):
            parts.append(f"**Oracle Syntax Rules:** {len(rules['oracle_syntax'])} rules")

        return Message(text="\n".join(parts))
