# Paste this into a Custom Code component's Code tab
# Knowledge Processor — ingests knowledge artifacts, outputs unified knowledge dict
# Handles both file PATHS and raw CONTENT from Knowledge Base component

from agentcore.custom import Node
import json
import re
import os


# ─── File Reading & Parsing ──────────────────────────────────────────────────

def _read_file(path):
    """Read a file and return its text content."""
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _parse_content(text, filename=""):
    """Parse file content — tries JSON first, then YAML, then flat YAML."""
    text = text.strip()
    if not text:
        return None

    # JSON
    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # YAML (standard indented)
    try:
        import yaml
        result = yaml.safe_load(text)
        if isinstance(result, (dict, list)):
            # Validate: yaml.safe_load on flat YAML returns dicts with None values
            # Check if most values are None — if so, it's flat YAML parsed wrong
            if isinstance(result, dict) and len(result) > 3:
                none_count = sum(1 for v in result.values() if v is None)
                if none_count > len(result) * 0.5:
                    pass  # Fall through to flat YAML parser
                else:
                    return result
            else:
                return result
    except Exception:
        pass

    # JSON (retry without prefix check)
    try:
        return json.loads(text)
    except Exception:
        pass

    # Flat YAML (zero-indent files used in knowledge artifacts)
    result = _parse_flat_yaml(text)
    if result and isinstance(result, dict) and len(result) > 0:
        return result

    return None


# ─── Filename-Based Type Detection ───────────────────────────────────────────

# Keywords mapped to artifact types — matched against normalized filename
# Order matters: longer/more-specific keys first to avoid false positives
_FILENAME_KEYWORDS = [
    (["knowledgegraph", "knowledge_graph", "kg"],          "knowledge_graph_file"),
    (["contextgraph", "context_graph"],                     "context_graph_file"),
    (["semanticlayer", "semantic_layer", "semantic"],        "semantic_layer_file"),
    (["ontology"],                                          "ontology_file"),
    (["synonym", "synonymn", "synonyms"],                   "synonyms_file"),
    (["businessrule", "business_rule", "business_rules"],   "business_rules_file"),
    (["example", "examples", "fewshot", "few_shot"],        "examples_file"),
    (["germanterm", "german_term", "german_column",
     "domainterm", "domain_term"],                          "domain_terms_file"),
    (["columnvalue", "column_value", "histogram"],          "column_values_file"),
    (["entityalias", "entities_alias", "alias"],            "entities_aliases_file"),
    (["antipattern", "anti_pattern"],                       "anti_patterns_file"),
    (["sqltemplate", "sql_template", "template"],           "sql_templates_file"),
    (["columns", "schema_columns"],                         "schema_columns_file"),
]


def _normalize_filename(filename):
    """Normalize filename for flexible matching.

    'Context Graph (1).yaml' -> 'contextgraph'
    'business_rules.YAML'    -> 'businessrules'
    'SemanticLayer-v2.json'  -> 'semanticlayerv2'
    """
    base = os.path.basename(filename).rsplit(".", 1)[0]  # strip extension
    base = base.lower()
    base = re.sub(r'\(\d+\)', '', base)       # remove (1), (2) etc.
    base = re.sub(r'[^a-z0-9]', '', base)     # strip all non-alphanumeric
    return base


def _detect_type_by_filename(filename):
    """Detect artifact type from filename — flexible matching.

    Handles: case variations, spaces/underscores/hyphens, (1) suffixes,
    CamelCase, version numbers, any extension.
    """
    if not filename:
        return None
    normalized = _normalize_filename(filename)
    if not normalized:
        return None
    for keywords, slot in _FILENAME_KEYWORDS:
        for kw in keywords:
            if kw in normalized:
                return slot
    return None


def _detect_type_by_content(parsed, filename=""):
    """Detect artifact type from content structure."""
    # Try filename first
    slot = _detect_type_by_filename(filename)
    if slot:
        return slot

    if not isinstance(parsed, dict):
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            if "question" in parsed[0] or "sql" in parsed[0]:
                return "examples_file"
        return None

    keys = set(parsed.keys())

    if "entities" in keys and ("relationships" in keys or "relations" in keys):
        return "knowledge_graph_file"
    if "hierarchies" in keys and ("valid_combinations" in keys or "constraints" in keys):
        return "ontology_file"
    if "columns" in keys and ("entity_mappings" in keys or "cardinality_summary" in keys):
        return "semantic_layer_file"
    if "question_types" in keys:
        return "context_graph_file"
    if "column_synonyms" in keys:
        return "synonyms_file"
    if "columns" in keys and ("aggregations" in keys or "patterns" in keys or "phrases" in keys):
        return "synonyms_file"
    if "metrics" in keys and ("exclusion_rules" in keys or "time_filters" in keys or "oracle_syntax" in keys):
        return "business_rules_file"
    if "examples" in keys:
        ex = parsed.get("examples")
        if isinstance(ex, list) and ex and isinstance(ex[0], dict):
            return "examples_file"
    if "column_mappings" in keys or "german_columns" in keys:
        return "domain_terms_file"
    if "view_name" in keys and "columns" in keys:
        return "schema_columns_file"
    if "templates" in keys:
        return "sql_templates_file"
    if "anti_patterns" in keys:
        return "anti_patterns_file"
    if any(k in keys for k in ("region_aliases", "oem_aliases", "business_concepts")):
        return "entities_aliases_file"

    return None


# ─── Flat YAML Parser (for zero-indent knowledge files) ─────────────────────

_LIKELY_TOP_KEYS = {
    'metadata', 'metrics', 'time_filters', 'exclusion_rules', 'oracle_syntax',
    'query_templates', 'question_types', 'columns', 'column_synonyms',
    'column_mappings', 'german_columns', 'concepts', 'hierarchies',
    'valid_combinations', 'constraints', 'entity_mappings', 'cardinality_summary',
    'filter_hints', 'examples', 'entities', 'relationships', 'aggregations',
    'patterns', 'phrases', 'templates', 'anti_patterns', 'view_name',
    'extracted_at', 'total_columns', 'unique_columns', 'duplicate_count',
    'region_aliases', 'country_aliases', 'oem_aliases', 'commodity_aliases',
    'business_concepts', 'exclusions', 'properties', 'columns_by_tier',
}


def _yaml_value(val_str):
    """Convert a YAML value string to a Python type."""
    if not val_str:
        return None
    if (val_str.startswith('"') and val_str.endswith('"')) or \
       (val_str.startswith("'") and val_str.endswith("'")):
        return val_str[1:-1]
    if val_str.startswith('[') and val_str.endswith(']'):
        inner = val_str[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip('"').strip("'") for item in re.split(r',\s*', inner) if item.strip()]
    if val_str.lower() in ('true', 'yes'):
        return True
    if val_str.lower() in ('false', 'no'):
        return False
    if val_str.lower() in ('null', 'none', '~'):
        return None
    try:
        return float(val_str) if '.' in val_str else int(val_str)
    except ValueError:
        return val_str


def _parse_flat_yaml(text):
    """Parse flat YAML (zero indentation) into a structured dict."""
    lines = text.replace('\r\n', '\n').split('\n')
    result = {}
    current_section = None
    current_item = None
    current_item_key = None
    items_in_section = {}
    section_list_items = []
    in_block_scalar = False
    block_lines = []
    block_key = None
    in_list = False
    list_key = None
    list_items = []
    section_list_data = {}

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
            if current_item_key == '__list_item__':
                section_list_items.append(current_item)
            else:
                items_in_section[current_item_key] = current_item
        current_item = None
        current_item_key = None

    def _flush_section():
        nonlocal current_section, items_in_section, section_list_items
        _flush_item()
        if current_section:
            if section_list_items and not items_in_section:
                result[current_section] = list(section_list_items)
            elif items_in_section:
                section_data = dict(items_in_section)
                if '__direct__' in section_data:
                    direct = section_data.pop('__direct__')
                    if isinstance(direct, dict):
                        section_data.update(direct)
                result[current_section] = section_data
            elif current_section in section_list_data:
                result[current_section] = section_list_data[current_section]
        items_in_section = {}
        section_list_items = []

    for raw_line in lines:
        stripped = raw_line.strip()

        if not stripped:
            _flush_block()
            _flush_list()
            if current_item_key and current_item is not None:
                _flush_item()
            continue

        if stripped.startswith('#'):
            continue

        if in_block_scalar:
            if re.match(r'^[A-Za-z_"\'][A-Za-z0-9_ .()"\'/]*:\s', stripped) or \
               (stripped.endswith(':') and not stripped.startswith(('SELECT', 'FROM', 'WHERE'))):
                _flush_block()
            else:
                block_lines.append(stripped)
                continue

        if stripped.startswith('- '):
            item_val = stripped[2:].strip()
            if not in_list and current_item is None and current_section:
                if current_section not in section_list_data:
                    section_list_data[current_section] = []
                if ': ' in item_val or item_val.endswith(':'):
                    current_item = {}
                    current_item_key = '__list_item__'
                    k, _, v = item_val.partition(':')
                    current_item[k.strip()] = v.strip() if v.strip() else None
                    continue
                else:
                    section_list_data[current_section].append(item_val.strip('"').strip("'"))
                    continue
            if in_list:
                list_items.append(item_val.strip('"').strip("'"))
            elif current_item is not None:
                in_list = True
                list_items = [item_val.strip('"').strip("'")]
            continue

        colon_idx = stripped.find(':')
        if colon_idx > 0:
            key = stripped[:colon_idx].strip().strip('"').strip("'")
            value = stripped[colon_idx + 1:].strip()
            _flush_list()

            if not value:
                if current_section is None or (key in _LIKELY_TOP_KEYS and (current_item is None or current_item_key == '__direct__')):
                    _flush_section()
                    current_section = key
                elif current_item is None:
                    current_item = {}
                    current_item_key = key
                else:
                    in_list = True
                    list_key = key
                    list_items = []
            elif value in ('|', '>'):
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
                if key in _LIKELY_TOP_KEYS and (current_item is None or current_item_key == '__direct__'):
                    _flush_section()
                    result[key] = _yaml_value(value)
                    continue
                if current_item is None:
                    if current_section:
                        current_item = {}
                        current_item_key = '__direct__'
                    else:
                        result[key] = _yaml_value(value)
                        continue
                current_item[key] = _yaml_value(value)
            continue

    _flush_section()
    return result if result else None


# ─── Build Methods ────────────────────────────────────────────────────────────

def _build_synonym_map(synonyms_data, semantic_layer, domain_terms):
    synonym_map = {}
    if synonyms_data:
        col_syns = synonyms_data.get("column_synonyms") or synonyms_data.get("columns") or synonyms_data
        if isinstance(col_syns, dict):
            for col_name, syn_info in col_syns.items():
                syns = syn_info if isinstance(syn_info, list) else (syn_info.get("synonyms", []) if isinstance(syn_info, dict) else [])
                for s in syns:
                    s_lower = str(s).lower().strip()
                    if s_lower:
                        synonym_map[s_lower] = {"column": col_name, "source": "synonyms_file"}
        for section_key in ("aggregations", "patterns"):
            section = synonyms_data.get(section_key, {})
            if isinstance(section, dict):
                for name, info in section.items():
                    if isinstance(info, dict):
                        for s in info.get("synonyms", []):
                            s_lower = str(s).lower().strip()
                            if s_lower and s_lower not in synonym_map:
                                synonym_map[s_lower] = {"column": name, "source": f"synonyms_{section_key}"}
        phrases = synonyms_data.get("phrases", {})
        if isinstance(phrases, dict):
            for phrase, info in phrases.items():
                p_lower = str(phrase).lower().strip()
                if p_lower and p_lower not in synonym_map:
                    sql = info.get("sql", "") if isinstance(info, dict) else str(info)
                    synonym_map[p_lower] = {"column": sql, "source": "synonyms_phrases"}
    if semantic_layer:
        for col_name, col_info in semantic_layer.get("columns", {}).items():
            if isinstance(col_info, dict):
                for s in col_info.get("synonyms", []):
                    s_lower = str(s).lower().strip()
                    if s_lower and s_lower not in synonym_map:
                        synonym_map[s_lower] = {"column": col_name, "entity": col_info.get("entity", ""), "source": "semantic_layer"}
    if domain_terms:
        terms = domain_terms.get("column_mappings") or domain_terms.get("german_columns") or domain_terms
        if isinstance(terms, dict):
            for col_name, info in terms.items():
                if isinstance(info, dict):
                    for s in info.get("translations", []):
                        s_lower = str(s).lower().strip()
                        if s_lower and s_lower not in synonym_map:
                            synonym_map[s_lower] = {"column": col_name, "source": "domain_terms"}
                    for k in ("german", "native", "term", "abbreviation"):
                        v = info.get(k, "")
                        if v:
                            synonym_map[v.lower().strip()] = {"column": col_name, "source": "domain_terms"}
    return synonym_map


def _build_entities(kg, sl):
    entities = {}
    if kg:
        for name, info in kg.get("entities", {}).items():
            if isinstance(info, dict):
                entities[name] = {
                    "type": info.get("type", "dimension"),
                    "primary_key": info.get("primary_key", ""),
                    "display_column": info.get("display_column", ""),
                    "columns": info.get("columns", []),
                    "measures": info.get("measures", []),
                    "description": info.get("description", ""),
                }
    if sl:
        for name, mapping in sl.get("entity_mappings", {}).items():
            if isinstance(mapping, dict) and name in entities:
                entities[name]["id_column"] = mapping.get("id_column", "")
    return entities


def _build_hierarchies(ontology):
    hierarchies = {}
    if not ontology:
        return hierarchies
    for name, h in ontology.get("hierarchies", {}).items():
        if isinstance(h, dict):
            levels = [{"level": l.get("level", 0), "name": l.get("name", ""), "column": l.get("column", "")}
                      for l in h.get("levels", []) if isinstance(l, dict)]
            hierarchies[name] = {"name": h.get("name", name), "levels": levels, "description": h.get("description", "")}
    return hierarchies


def _build_business_rules(rules_data):
    rules = {"metrics": {}, "exclusion_rules": [], "time_filters": {}, "oracle_syntax": {}}
    if not rules_data:
        return rules
    for name, info in rules_data.get("metrics", {}).items():
        rules["metrics"][name] = info.get("sql", info.get("expression", str(info))) if isinstance(info, dict) else str(info)
    exclusions = rules_data.get("exclusion_rules", rules_data.get("exclusions", []))
    if isinstance(exclusions, dict):
        for name, info in exclusions.items():
            rules["exclusion_rules"].append(f"{name}: {info.get('condition', str(info)) if isinstance(info, dict) else info}")
    elif isinstance(exclusions, list):
        rules["exclusion_rules"] = [str(e) for e in exclusions]
    tf = rules_data.get("time_filters", {})
    if isinstance(tf, dict):
        rules["time_filters"] = {k: (v.get("sql", str(v)) if isinstance(v, dict) else str(v)) for k, v in tf.items()}
    oracle = rules_data.get("oracle_syntax", rules_data.get("oracle_specific", {}))
    if isinstance(oracle, dict):
        rules["oracle_syntax"] = {k: str(v) for k, v in oracle.items()}
    return rules


def _build_column_metadata(schema_columns, semantic_layer):
    columns = {}
    if schema_columns:
        col_defs = schema_columns.get("columns", schema_columns)
        if isinstance(col_defs, dict):
            for name, info in col_defs.items():
                if isinstance(info, dict):
                    columns[name] = {"type": info.get("type", ""), "category": info.get("category", ""),
                                     "description": info.get("description", ""), "nullable": info.get("nullable", True)}
    if semantic_layer:
        for name, info in semantic_layer.get("columns", {}).items():
            if isinstance(info, dict):
                if name not in columns:
                    columns[name] = {}
                columns[name]["entity"] = info.get("entity", "")
                columns[name]["synonyms"] = info.get("synonyms", [])
                if not columns[name].get("description"):
                    columns[name]["description"] = info.get("description", "")
    return columns


def _index_examples(examples_data):
    examples = []
    if not examples_data:
        return examples
    ex_list = examples_data if isinstance(examples_data, list) else examples_data.get("examples", [])
    for ex in (ex_list if isinstance(ex_list, list) else []):
        if isinstance(ex, dict):
            examples.append({"question": ex.get("question") or ex.get("input", ""),
                             "sql": ex.get("sql") or ex.get("output", ""),
                             "category": ex.get("category", ""), "complexity": ex.get("complexity", 1)})
    return examples


def _build_intent_index(context_graph):
    index = {}
    if not context_graph:
        return index
    for intent_name, intent_def in context_graph.get("question_types", {}).items():
        tokens = set()
        for p in intent_def.get("patterns", []):
            for word in re.split(r"\s+", p.lower()):
                word = word.strip(".,?!'\"")
                if len(word) > 1:
                    tokens.add(word)
        index[intent_name] = {"tokens": tokens, "definition": intent_def}
    return index


def _build_entity_aliases(aliases_data):
    aliases = {}
    if not aliases_data:
        return aliases
    for alias_type, col in [("region_aliases", "REGION"), ("country_aliases", "COUNTRY"),
                            ("oem_aliases", "CUSTOMER"), ("commodity_aliases", "COMMODITY_DESCRIPTION")]:
        for alias, canonical in aliases_data.get(alias_type, {}).items():
            aliases[str(alias).lower()] = {"type": alias_type.replace("_aliases", ""),
                                           "canonical_value": str(canonical), "sql_filter": f"{col} = '{canonical}'"}
    for alias, sql in aliases_data.get("business_concepts", {}).items():
        aliases[str(alias).lower()] = {"type": "business_concept", "sql_filter": str(sql)}
    return aliases


# ─── Component ────────────────────────────────────────────────────────────────

class CodeEditorNode(Node):
    display_name = "Knowledge Processor"
    description = "Processes knowledge artifacts (YAML/JSON) into unified context. Pure code."
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
            input_types=["Data", "Message"],
            info="From Knowledge Base component.",
            required=False,
        ),
        MultilineInput(name="additional_rules", display_name="Additional Business Rules", value=""),
        MultilineInput(name="additional_context", display_name="Additional Domain Context", value=""),
    ]

    outputs = [
        Output(display_name="Knowledge Context", name="output", method="build_output"),
    ]

    def _extract_files_from_kb(self, kb):
        """Extract individual files from Knowledge Base output.

        Returns list of (filename, parsed_content) tuples.
        Handles: file paths, concatenated text, Data objects, Message objects.
        """
        files = []

        # Get raw text from the KB output
        text = None
        if hasattr(kb, "text") and isinstance(kb.text, str):
            text = kb.text.strip()
        elif hasattr(kb, "data"):
            inner = kb.data
            if isinstance(inner, str):
                text = inner.strip()
            elif isinstance(inner, dict):
                text = (inner.get("text") or inner.get("content") or inner.get("file_content") or "").strip()
        elif isinstance(kb, str):
            text = kb.strip()

        if not text:
            return files

        # Check if text contains file paths (lines ending in .yaml, .json, etc.)
        text_lines = [l.strip() for l in text.replace('\r\n', '\n').split('\n') if l.strip()]
        sample = text_lines[:min(5, len(text_lines))]
        looks_like_paths = len(sample) > 0 and all(
            any(l.lower().endswith(ext) for ext in ('.yaml', '.yml', '.json', '.txt', '.md', '.csv'))
            and ('/' in l or '\\' in l)
            for l in sample
        )

        if looks_like_paths:
            # Input is file paths — read each file
            for line in text_lines:
                path = line.strip().strip('"').strip("'")
                if not path:
                    continue
                content = _read_file(path)
                if content:
                    filename = os.path.basename(path)
                    parsed = _parse_content(content, filename)
                    if parsed is not None:
                        files.append((filename, parsed))
        else:
            # Input is raw content — treat as single document or concatenated
            parsed = _parse_content(text)
            if parsed is not None:
                files.append(("", parsed))

        return files

    def build_output(self) -> Data:
        all_files = []  # list of (filename, parsed_content)
        debug_info = {}

        # From Knowledge Base
        kb = self.knowledge_base
        if kb and kb != "":
            debug_info["kb_type"] = str(type(kb).__name__)
            debug_info["kb_has_text"] = hasattr(kb, "text")
            debug_info["kb_has_data"] = hasattr(kb, "data")
            if hasattr(kb, "text"):
                txt = str(kb.text) if kb.text else ""
                debug_info["kb_text_length"] = len(txt)
                debug_info["kb_text_preview"] = txt[:300]
            if hasattr(kb, "data"):
                d = kb.data
                debug_info["kb_data_type"] = str(type(d).__name__)
                if isinstance(d, dict):
                    debug_info["kb_data_keys"] = list(d.keys())[:10]
                elif isinstance(d, str):
                    debug_info["kb_data_preview"] = d[:200]

            extracted = self._extract_files_from_kb(kb)
            all_files.extend(extracted)
            debug_info["kb_files_extracted"] = len(extracted)
            debug_info["kb_file_names"] = [f[0] for f in extracted]

        # From text input (for quick testing)
        text_in = self.input_value
        if text_in and isinstance(text_in, str) and text_in.strip():
            parsed = _parse_content(text_in.strip())
            if parsed is not None:
                all_files.append(("text_input", parsed))

        if not all_files:
            self.status = "No knowledge inputs parsed"
            return Data(data={"error": True, "message": "No knowledge inputs parsed.", "_debug": debug_info})

        # Detect artifact types and assign to slots
        slots = {}
        files_detected = []
        name_map = {
            "knowledge_graph_file": "Knowledge Graph", "ontology_file": "Ontology",
            "semantic_layer_file": "Semantic Layer", "context_graph_file": "Context Graph",
            "synonyms_file": "Synonyms", "business_rules_file": "Business Rules",
            "examples_file": "Few-Shot Examples", "domain_terms_file": "Domain Terms",
            "schema_columns_file": "Schema Columns", "sql_templates_file": "SQL Templates",
            "anti_patterns_file": "Anti-Patterns", "column_values_file": "Column Values",
            "entities_aliases_file": "Entity Aliases",
        }

        for filename, parsed in all_files:
            slot = _detect_type_by_content(parsed, filename)
            if slot:
                slots[slot] = parsed
                display = name_map.get(slot, slot)
                count = len(parsed) if isinstance(parsed, (dict, list)) else 1
                files_detected.append({"name": display, "items": count, "filename": filename})

        loaded = len(files_detected)

        # Build unified structures
        synonym_map = _build_synonym_map(slots.get("synonyms_file"), slots.get("semantic_layer_file"), slots.get("domain_terms_file"))
        entities = _build_entities(slots.get("knowledge_graph_file"), slots.get("semantic_layer_file"))
        hierarchies = _build_hierarchies(slots.get("ontology_file"))
        business_rules = _build_business_rules(slots.get("business_rules_file"))
        column_metadata = _build_column_metadata(slots.get("schema_columns_file"), slots.get("semantic_layer_file"))
        examples = _index_examples(slots.get("examples_file"))
        intent_index = _build_intent_index(slots.get("context_graph_file"))
        entity_aliases = _build_entity_aliases(slots.get("entities_aliases_file"))

        valid_combos = {}
        ont = slots.get("ontology_file")
        if ont:
            valid_combos = {"valid_combinations": ont.get("valid_combinations", {}),
                            "constraints": ont.get("constraints", {})}

        additional_rules = (self.additional_rules or "").strip()
        additional_context = (self.additional_context or "").strip()

        context = {
            "synonym_map": synonym_map,
            "entities": entities,
            "hierarchies": hierarchies,
            "business_rules": business_rules,
            "column_value_hints": {},
            "column_metadata": column_metadata,
            "examples": examples,
            "intent_patterns": {},
            "valid_combinations": valid_combos,
            "intent_index": intent_index,
            "sql_templates": {},
            "anti_patterns": [],
            "column_values_detailed": {},
            "entity_aliases": entity_aliases,
            "additional_business_rules": additional_rules,
            "additional_domain_context": additional_context,
            "knowledge_files_loaded": loaded,
            "total_knowledge_slots": 13,
            "synonym_count": len(synonym_map),
            "entity_count": len(entities),
            "hierarchy_count": len(hierarchies),
            "example_count": len(examples),
            "column_count": len(column_metadata),
            "entity_alias_count": len(entity_aliases),
            "files_detected": files_detected,
            "_debug": debug_info,
        }

        self.status = f"Loaded: {loaded}/13 | {len(synonym_map)} synonyms, {len(entities)} entities, {len(examples)} examples"

        # Summary text
        parts = [f"**Knowledge Processor** (pure code -- no LLM)\n", f"**Files Detected:** {loaded}/13"]
        for fd in files_detected:
            parts.append(f"  - {fd['name']} ({fd['items']} items) [{fd['filename']}]")
        parts.append("\n**Knowledge Summary:**")
        for label, count in [("Synonyms", len(synonym_map)), ("Entities", len(entities)),
                             ("Hierarchies", len(hierarchies)), ("Examples", len(examples)),
                             ("Columns", len(column_metadata)), ("Entity Aliases", len(entity_aliases))]:
            if count > 0:
                parts.append(f"  - {label}: {count}")

        result = Data(data=context)
        result.text_key = "summary"
        context["summary"] = "\n".join(parts)
        return result
