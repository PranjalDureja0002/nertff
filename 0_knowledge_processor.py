# Paste this into a Custom Code component's Code tab
# Knowledge Processor — ingests 13 artifact types, outputs unified knowledge dict
# Replaces backend Knowledge Layer for Custom Code flows

from agentcore.custom import Node
import json
import re

# ─── Artifact Detection ───────────────────────────────────────────────────────

def _detect_artifact_type(parsed, filename=""):
    """Detect knowledge artifact type from content structure."""
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

        # Context graph: question_types
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

        # Domain terms / translations: column_mappings
        if "column_mappings" in keys:
            return "domain_terms_file"

        # SQL templates: has "templates" key with template/sql_template items
        if "templates" in keys:
            templates_val = parsed.get("templates", {})
            if isinstance(templates_val, dict):
                sample = next(iter(templates_val.values()), None)
                if isinstance(sample, dict) and ("template" in sample or "sql_template" in sample):
                    return "sql_templates_file"

        # Anti-patterns: has "anti_patterns" list
        if "anti_patterns" in keys:
            ap_val = parsed.get("anti_patterns", [])
            if isinstance(ap_val, list):
                return "anti_patterns_file"

        # Column values: columns + columns_by_tier, or columns with cardinality/values
        if "columns" in keys and "columns_by_tier" in keys:
            return "column_values_file"
        if "columns" in keys:
            cols_val = parsed.get("columns", {})
            if isinstance(cols_val, dict):
                sample = next(iter(cols_val.values()), None)
                if isinstance(sample, dict) and ("cardinality" in sample or "values" in sample or "tier" in sample):
                    return "column_values_file"

        # Entity aliases: region_aliases, business_concepts, oem_aliases, etc.
        if any(k in keys for k in ("region_aliases", "business_concepts", "oem_aliases", "commodity_aliases", "country_aliases")):
            return "entities_aliases_file"

        # Schema columns: top-level values are dicts with type/category/description
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

    # Filename fallback
    _FILENAME_HINTS = {
        "knowledge_graph": "knowledge_graph_file", "ontology": "ontology_file",
        "semantic_layer": "semantic_layer_file", "context_graph": "context_graph_file",
        "synonym": "synonyms_file", "business_rule": "business_rules_file",
        "example": "examples_file", "term": "domain_terms_file",
        "sql_template": "sql_templates_file", "template": "sql_templates_file",
        "anti_pattern": "anti_patterns_file", "column_value": "column_values_file",
        "histogram": "column_values_file", "entities_alias": "entities_aliases_file",
        "alias": "entities_aliases_file",
    }
    fname_lower = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0].lower() if filename else ""
    for hint, slot in _FILENAME_HINTS.items():
        if hint in fname_lower:
            return slot

    return None


# ─── Parsing Helpers ──────────────────────────────────────────────────────────

def _parse_text(text):
    """Try parsing as JSON first, then YAML fallback."""
    text = text.strip()
    if not text:
        return None
    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # YAML fallback
    try:
        import yaml
        result = yaml.safe_load(text)
        if isinstance(result, (dict, list)):
            return result
    except Exception:
        pass
    # Try JSON even without { prefix (some files have whitespace)
    try:
        return json.loads(text)
    except Exception:
        pass
    return None


def _safe_parse(raw):
    """Parse raw file content into dict/list. Handles Message, Data, dict, str."""
    if raw is None:
        return None
    # Message objects have .text attribute (from Knowledge Base)
    if hasattr(raw, "text") and isinstance(raw.text, str):
        return _parse_text(raw.text)
    # Data objects have .data attribute
    if hasattr(raw, "data"):
        inner = raw.data
        if isinstance(inner, dict):
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


# ─── Build Methods ────────────────────────────────────────────────────────────

def _build_synonym_map(synonyms_data, semantic_layer, domain_terms):
    synonym_map = {}

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

    if domain_terms:
        terms = domain_terms.get("column_mappings", domain_terms)
        if isinstance(terms, dict):
            for col_name, term_info in terms.items():
                if isinstance(term_info, dict):
                    for s in term_info.get("translations", []):
                        s_lower = str(s).lower().strip()
                        if s_lower and s_lower not in synonym_map:
                            synonym_map[s_lower] = {"column": col_name, "source": "domain_terms"}
                    for key in ("german", "native", "term", "abbreviation"):
                        native = term_info.get(key, "")
                        if native:
                            synonym_map[native.lower().strip()] = {"column": col_name, "source": "domain_terms"}
                elif isinstance(term_info, str):
                    synonym_map[term_info.lower().strip()] = {"column": col_name, "source": "domain_terms"}

    return synonym_map


def _build_entities(knowledge_graph, semantic_layer):
    entities = {}

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

    if semantic_layer:
        mappings = semantic_layer.get("entity_mappings", {})
        for name, mapping in mappings.items():
            if isinstance(mapping, dict) and name in entities:
                entities[name]["id_column"] = mapping.get("id_column", "")
                entities[name]["display_column"] = mapping.get("display_column", entities[name].get("display_column", ""))

    return entities


def _build_hierarchies(ontology):
    hierarchies = {}
    if not ontology:
        return hierarchies

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


def _build_business_rules(rules_data):
    rules = {
        "metrics": {},
        "exclusion_rules": [],
        "time_filters": {},
        "classification_rules": {},
        "oracle_syntax": {},
    }
    if not rules_data:
        return rules

    metrics = rules_data.get("metrics", {})
    if isinstance(metrics, dict):
        for name, info in metrics.items():
            if isinstance(info, dict):
                rules["metrics"][name] = info.get("sql", info.get("expression", str(info)))
            elif isinstance(info, str):
                rules["metrics"][name] = info

    exclusions = rules_data.get("exclusion_rules", rules_data.get("exclusions", []))
    if isinstance(exclusions, dict):
        for name, info in exclusions.items():
            if isinstance(info, dict):
                rules["exclusion_rules"].append(f"{name}: {info.get('condition', info.get('sql', ''))}")
            elif isinstance(info, str):
                rules["exclusion_rules"].append(f"{name}: {info}")
    elif isinstance(exclusions, list):
        rules["exclusion_rules"] = [str(e) for e in exclusions]

    time_filters = rules_data.get("time_filters", {})
    if isinstance(time_filters, dict):
        rules["time_filters"] = {
            k: (v.get("sql", str(v)) if isinstance(v, dict) else str(v))
            for k, v in time_filters.items()
        }

    oracle = rules_data.get("oracle_syntax", rules_data.get("oracle_specific", {}))
    if isinstance(oracle, dict):
        rules["oracle_syntax"] = {k: str(v) for k, v in oracle.items()}

    return rules


def _build_column_value_hints(semantic_layer):
    hints = {}
    if not semantic_layer:
        return hints

    columns = semantic_layer.get("columns", {})
    for col_name, col_info in columns.items():
        if isinstance(col_info, dict):
            card_info = col_info.get("cardinality_info", {})
            if isinstance(card_info, dict) and card_info.get("unique_values"):
                unique = card_info.get("unique_values", 0)
                if unique < 200:
                    hints[col_name] = {
                        "cardinality": "low" if unique < 50 else "medium",
                        "unique_values": unique,
                        "examples": card_info.get("examples", []),
                        "description": col_info.get("description", ""),
                    }

    filter_hints = semantic_layer.get("filter_hints", {})
    if isinstance(filter_hints, dict):
        for col_name, hint_text in filter_hints.items():
            if col_name in hints:
                hints[col_name]["filter_hint"] = str(hint_text)
            elif isinstance(hint_text, str):
                hints[col_name] = {"filter_hint": hint_text}

    return hints


def _build_column_metadata(schema_columns, semantic_layer):
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


def _build_sql_templates(templates_data):
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


def _build_anti_patterns(anti_patterns_data):
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
                pass
        patterns.append({
            "id": ap.get("id", ""),
            "name": ap.get("name", ""),
            "pattern": regex_str,
            "compiled": compiled,
            "severity": ap.get("severity", "warning"),
            "description": ap.get("description", ""),
            "fix": ap.get("fix", ""),
        })

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
                    "severity": "error" if vr.get("required") or vr.get("forbidden") else "info",
                    "description": vr.get("description", ""),
                    "required": vr.get("required", False),
                    "forbidden": vr.get("forbidden", False),
                    "fix": "",
                })

    return patterns


def _build_column_values(column_values_data):
    result = {}
    if not column_values_data:
        return result

    columns = column_values_data.get("columns", {})
    if not isinstance(columns, dict):
        return result

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
        result[col_name] = {"cardinality": cardinality, "tier": tier, "values": values}

    return result


def _build_entity_aliases(aliases_data):
    aliases = {}
    if not aliases_data:
        return aliases

    for alias_type, col_filter in [
        ("region_aliases", "REGION"), ("country_aliases", "COUNTRY"),
        ("oem_aliases", "CUSTOMER"), ("commodity_aliases", "COMMODITY_DESCRIPTION"),
    ]:
        type_data = aliases_data.get(alias_type, {})
        if isinstance(type_data, dict):
            type_name = alias_type.replace("_aliases", "")
            for alias, canonical in type_data.items():
                key = str(alias).lower().strip()
                if key not in aliases:
                    aliases[key] = {
                        "type": type_name,
                        "canonical_value": str(canonical),
                        "sql_filter": f"{col_filter} = '{canonical}'",
                    }

    business_concepts = aliases_data.get("business_concepts", {})
    if isinstance(business_concepts, dict):
        for alias, sql_filter in business_concepts.items():
            aliases[str(alias).lower().strip()] = {
                "type": "business_concept",
                "canonical_value": str(alias),
                "sql_filter": str(sql_filter),
            }

    return aliases


def _index_examples(examples_data):
    examples = []
    if not examples_data:
        return examples

    if isinstance(examples_data, list):
        ex_list = examples_data
    elif isinstance(examples_data, dict):
        ex_list = examples_data.get("examples", [])
    else:
        return examples

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

    return examples


def _build_intent_index(context_graph):
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
        index[intent_name] = {"tokens": tokens, "definition": intent_def}
    return index


def _build_valid_combinations(ontology):
    if not ontology:
        return {}
    combos = ontology.get("valid_combinations", {})
    constraints = ontology.get("constraints", {})
    return {
        "valid_combinations": combos if isinstance(combos, dict) else {},
        "constraints": constraints if isinstance(constraints, dict) else {},
    }


# ─── Component ────────────────────────────────────────────────────────────────

class CodeEditorNode(Node):
    display_name = "Knowledge Processor"
    description = "Processes 13 knowledge artifact types into unified context. Pure code — no LLM cost."
    icon = "brain"
    name = "KnowledgeProcessor"

    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="Raw Knowledge Text",
            info="Paste raw JSON/YAML text here for quick testing.",
            tool_mode=False,
        ),
        HandleInput(
            name="knowledge_base",
            display_name="Knowledge Base",
            input_types=["Message"],
            info="From Knowledge Base component (auto-detects artifact types from uploaded files).",
            required=False,
        ),
        MultilineInput(
            name="additional_rules",
            display_name="Additional Business Rules",
            value="",
            info="Extra business rules (free text, appended to context).",
        ),
        MultilineInput(
            name="additional_context",
            display_name="Additional Domain Context",
            value="",
            info="Extra domain context (free text).",
        ),
    ]

    outputs = [
        Output(display_name="Knowledge Context", name="output", method="build_output"),
    ]

    def build_output(self) -> Data:
        # Collect all raw inputs
        raw_inputs = []

        # From Knowledge Base (list of Data objects from uploaded files)
        kb = self.knowledge_base
        if kb and kb != "":
            if isinstance(kb, list):
                for item in kb:
                    raw_inputs.append(item)
            else:
                raw_inputs.append(kb)

        # From text input (for quick testing)
        text_in = self.input_value
        if text_in and text_in.strip():
            if text_in.startswith('"') and text_in.endswith('"'):
                text_in = text_in[1:-1]
            raw_inputs.append(text_in)

        if not raw_inputs:
            self.status = "No knowledge inputs provided"
            return Data(data={"error": True, "message": "No knowledge inputs. Connect Knowledge Base or paste text."})

        # Parse and auto-detect each input
        slots = {}
        files_detected = []

        for raw in raw_inputs:
            parsed = _safe_parse(raw)
            if parsed is None:
                continue

            # Try to get filename for fallback detection
            filename = ""
            if hasattr(raw, "data") and isinstance(raw.data, dict):
                filename = raw.data.get("file_name", raw.data.get("source", ""))

            slot = _detect_artifact_type(parsed, filename)
            if slot:
                slots[slot] = parsed
                name_map = {
                    "knowledge_graph_file": "Knowledge Graph",
                    "ontology_file": "Ontology",
                    "semantic_layer_file": "Semantic Layer",
                    "context_graph_file": "Context Graph",
                    "synonyms_file": "Synonyms",
                    "business_rules_file": "Business Rules",
                    "examples_file": "Few-Shot Examples",
                    "domain_terms_file": "Domain Terms",
                    "schema_columns_file": "Schema Columns",
                    "sql_templates_file": "SQL Templates",
                    "anti_patterns_file": "Anti-Patterns",
                    "column_values_file": "Column Values",
                    "entities_aliases_file": "Entity Aliases",
                }
                display_name = name_map.get(slot, slot)
                item_count = len(parsed) if isinstance(parsed, (dict, list)) else 1
                files_detected.append({"name": display_name, "items": item_count})

        # Extract parsed data by slot
        knowledge_graph = slots.get("knowledge_graph_file")
        ontology = slots.get("ontology_file")
        semantic_layer = slots.get("semantic_layer_file")
        context_graph = slots.get("context_graph_file")
        synonyms_data = slots.get("synonyms_file")
        business_rules_data = slots.get("business_rules_file")
        examples_data = slots.get("examples_file")
        domain_terms = slots.get("domain_terms_file")
        schema_columns = slots.get("schema_columns_file")
        sql_templates_data = slots.get("sql_templates_file")
        anti_patterns_data = slots.get("anti_patterns_file")
        column_values_data = slots.get("column_values_file")
        entities_aliases_data = slots.get("entities_aliases_file")

        loaded = len(files_detected)

        # Build unified structures
        synonym_map = _build_synonym_map(synonyms_data, semantic_layer, domain_terms)
        entities = _build_entities(knowledge_graph, semantic_layer)
        hierarchies = _build_hierarchies(ontology)
        business_rules = _build_business_rules(business_rules_data)
        column_value_hints = _build_column_value_hints(semantic_layer)
        column_metadata = _build_column_metadata(schema_columns, semantic_layer)
        examples = _index_examples(examples_data)
        sql_templates = _build_sql_templates(sql_templates_data)
        anti_patterns = _build_anti_patterns(anti_patterns_data)
        column_values_detailed = _build_column_values(column_values_data)
        entity_aliases = _build_entity_aliases(entities_aliases_data)
        valid_combinations = _build_valid_combinations(ontology)

        # Build intent index
        intent_index = {}
        if context_graph and isinstance(context_graph, dict):
            intent_index = _build_intent_index(context_graph)

        additional_rules = (self.additional_rules or "").strip()
        additional_context = (self.additional_context or "").strip()

        context = {
            "synonym_map": synonym_map,
            "entities": entities,
            "hierarchies": hierarchies,
            "business_rules": business_rules,
            "column_value_hints": column_value_hints,
            "column_metadata": column_metadata,
            "examples": examples,
            "intent_patterns": {},
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
            "files_detected": files_detected,
        }

        self.status = (
            f"Loaded: {loaded}/13 | {len(synonym_map)} synonyms, "
            f"{len(entities)} entities, {len(examples)} examples, "
            f"{len(sql_templates)} templates, {len(anti_patterns)} anti-patterns, "
            f"{len(entity_aliases)} aliases"
        )

        # Format readable summary for Chat Output
        parts = [
            f"**Knowledge Processor** (pure code -- no LLM)\n",
            f"**Files Detected:** {loaded}/13",
        ]
        for fd in files_detected:
            parts.append(f"  - {fd['name']} ({fd['items']} items)")

        counts = [
            ("Synonyms", len(synonym_map)),
            ("Entities", len(entities)),
            ("Hierarchies", len(hierarchies)),
            ("Examples", len(examples)),
            ("Columns", len(column_metadata)),
            ("SQL Templates", len(sql_templates)),
            ("Anti-Patterns", len(anti_patterns)),
            ("Column Values", len(column_values_detailed)),
            ("Entity Aliases", len(entity_aliases)),
        ]
        parts.append("\n**Knowledge Summary:**")
        for name, count in counts:
            if count > 0:
                parts.append(f"  - {name}: {count}")

        if additional_rules:
            parts.append(f"\n**Additional Rules:** {additional_rules[:200]}...")
        if additional_context:
            parts.append(f"**Additional Context:** {additional_context[:200]}...")

        # Return Data with the text representation for display
        result = Data(data=context)
        result.text_key = "summary"
        context["summary"] = "\n".join(parts)
        return result
