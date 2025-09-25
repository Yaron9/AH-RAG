import json
import re
import hashlib
from pydantic import ValidationError, TypeAdapter
from typing import List
from ah_rag.extract.hypergraph_schema import HypergraphExtraction, ExtractionResponse, Entity
from ah_rag.utils.llm_client import LLMModule, create_chat_completion, is_llm_enabled
import os

class HypergraphExtractor:
    """
    Extracts structured knowledge hypergraphs from text using an OpenAI-compatible API.
    """
    CANONICAL_TYPES = {
        "person": {"aliases": {"human", "individual", "artist", "actor", "director", "author"}},
        "organization": {"aliases": {"company", "agency", "institution", "team", "studio"}},
        "position": {"aliases": {"role", "office", "title", "job", "occupation"}},
        "location": {"aliases": {"place", "city", "country", "region", "state", "province", "neighborhood"}},
        "event": {"aliases": {"conference", "war", "summit", "ceremony"}},
        "work": {"aliases": {"film", "movie", "book", "novel", "song", "album", "series"}},
        "concept": {"aliases": {"idea", "theory", "technology", "process"}},
        "date": {"aliases": {"year", "time", "era"}},
    }

    def __init__(self, granularity: str = 'fine'):
        """
        Initializes the extractor using unified LLM configuration.
        """
        self.granularity = granularity
        self.prompt_template = self._build_prompt_template()

    def _build_prompt_template(self) -> str:
        """Builds the prompt template for the LLM."""
        return f"""
You are a precision JSON generator. Read the TEXT and return EXACTLY one JSON object.
Do NOT add commentary, code fences, or explanations.

GLOBAL RULES
- Maximum 8 extractions. Each extraction describes one atomic fact/event.
- Every extraction MUST include: hyperedge (short verb phrase), relation_type (CamelCase),
  entities (list of objects containing the keys ["name", "type", "description"]), confidence_score (1-10).
- Entity types must belong to this controlled set: person, organization, position, location,
  work, event, concept, date. If unsure, pick the closest type and explain rationale in the
  description.
- Descriptions must consolidate all key attributes mentioned in the text: nationality, role,
  dates, numeric facts, aliases, relationships, etc.
- If the text mentions nationality, citizenship, or residence for a PERSON, explicitly include
  it in the description (e.g., "Italian American director based in Greenwich Village, New York City").
- If the text states an official title or position (e.g., "Chief of Protocol"), ensure a POSITION
  entity captures that title verbatim and link it to the relevant PERSON via the same extraction.
- Keep descriptions concise (<= 160 characters) and avoid repeating the same sentence across
  multiple entities.
- Cover distinct facts. Do not waste extractions repeating similar statements about the same
  film or organization if there are untouched people, locations, or positions available.
- When the text gives multiple facets of the same surface form (e.g. a person vs. a film of
  the same name) emit separate entities with distinct types.
- Preserve factual grounding. If the source text states a role like "Chief of Protocol", the
  position entity must contain that title verbatim in its description.

WORKFLOW
1. Skim the TEXT and identify the strongest 1-8 knowledge fragments.
2. For each fragment, define a hyperedge that captures the core relation/action.
3. Attach every participating entity, filling type/description with the richest details.
4. Set confidence_score to reflect extraction certainty (integer or float 1-10).
5. Return JSON of the exact shape {{{{"extractions": [...]}}}}.

TEXT:
{{text_chunk}}
"""

    def extract(self, text_chunk: str) -> List[HypergraphExtraction]:
        """
        Extracts hypergraphs from a single chunk of text.
        """
        prompt = self.prompt_template.format(text_chunk=text_chunk)

        raw_json: str | None = None

        # Check if LLM is enabled for knowledge extraction
        if not is_llm_enabled(LLMModule.KNOWLEDGE_EXTRACTION):
            print("LLM knowledge extraction disabled, using fallback")
            return self._fallback_extract(text_chunk)

        try:
            response = create_chat_completion(
                LLMModule.KNOWLEDGE_EXTRACTION,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            raw_json = response.choices[0].message.content
        except Exception as e:
            print(f"--- AN API ERROR OCCURRED ---")
            import traceback
            traceback.print_exc()
            print("-----------------------------")

        if not raw_json:
            return self._fallback_extract(text_chunk)

        # Try multiple parsing strategies
        candidates: List[str] = []
        # 1) fenced ```json ... ```
        for m in re.finditer(r"```json\s*([\s\S]*?)```", raw_json):
            candidates.append(m.group(1))
        # 2) outermost braces slice
        l = raw_json.find('{')
        r = raw_json.rfind('}')
        if l != -1 and r != -1 and r > l:
            candidates.append(raw_json[l:r+1])
        # 3) extractions array slice
        ex_pos = raw_json.find('"extractions"')
        if ex_pos != -1:
            lb = raw_json.find('[', ex_pos)
            rb = raw_json.rfind(']')
            if lb != -1 and rb != -1 and rb > lb:
                arr = raw_json[lb:rb+1]
                candidates.append('{"extractions": ' + arr + '}')

        def coerce_conf(v):
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                s = v.strip()
                mapping = {"高": 9.0, "中": 6.0, "低": 3.0}
                if s in mapping:
                    return mapping[s]
                try:
                    return float(s)
                except Exception:
                    return 6.0
            return 6.0

        # 4) salvage: extract individual JSON objects and assemble
        def salvage_objects(text: str) -> List[dict]:
            objs: List[dict] = []
            buf = []
            depth = 0
            for ch in text:
                if ch == '{':
                    depth += 1
                if depth > 0:
                    buf.append(ch)
                if ch == '}':
                    depth -= 1
                    if depth == 0 and buf:
                        chunk = ''.join(buf)
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict):
                                objs.append(obj)
                        except Exception:
                            pass
                        buf = []
            return objs

        for cand in candidates:
            try:
                data = json.loads(cand)
                if isinstance(data, dict) and isinstance(data.get("extractions"), list):
                    for item in data["extractions"]:
                        if "confidence_score" in item:
                            item["confidence_score"] = coerce_conf(item.get("confidence_score"))
                    adapter = TypeAdapter(ExtractionResponse)
                    validated = adapter.validate_python(data)
                    ex = self._postprocess_extractions(validated.extractions, text_chunk)
                    return ex[:8] if isinstance(ex, list) else ex
            except Exception:
                # try salvage path on candidate text
                try:
                    objs = salvage_objects(cand)
                    if objs:
                        for it in objs:
                            if "confidence_score" in it:
                                it["confidence_score"] = coerce_conf(it.get("confidence_score"))
                        adapter = TypeAdapter(ExtractionResponse)
                        validated = adapter.validate_python({"extractions": objs})
                        ex = self._postprocess_extractions(validated.extractions, text_chunk)
                        return ex[:8] if isinstance(ex, list) else ex
                    partial = self._recover_partial_extractions(cand)
                    if partial:
                        return partial[:8]
                except Exception:
                    pass
                continue
        print("Failed to validate or decode response from LLM: no valid JSON candidate")
        print(f"Raw response was: {raw_json}")
        return self._fallback_extract(text_chunk)

    def _postprocess_extractions(self, extractions: List[HypergraphExtraction], text_chunk: str) -> List[HypergraphExtraction]:
        """Normalize entity types/descriptions to improve downstream robustness."""
        context_lower = text_chunk.lower()
        for extraction in extractions:
            # Clamp confidence to [1,10]
            extraction.confidence_score = max(1.0, min(10.0, float(extraction.confidence_score or 6.0)))
            normalized_entities: List[Entity] = []
            for ent in extraction.entities:
                ent_type = self._normalize_entity_type(ent.type, ent.name, ent.description, context_lower)
                raw_desc = ent.description.strip() if ent.description else ""
                snippet = self._extract_snippet(ent.name, text_chunk)
                if raw_desc:
                    if snippet and snippet.lower() not in raw_desc.lower():
                        candidate = f"{raw_desc} | {snippet}"
                    else:
                        candidate = raw_desc
                else:
                    candidate = snippet or raw_desc
                description = candidate[:160]
                if len(candidate) > 160:
                    description = candidate[:157] + "..."
                normalized_entities.append(Entity(name=ent.name.strip(), type=ent_type, description=description))
            extraction.entities = normalized_entities
        return extractions

    def _fallback_extract(self, text_chunk: str) -> List[HypergraphExtraction]:
        """Heuristic extraction used when the LLM endpoint is unavailable."""
        sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s+", text_chunk) if s.strip()]
        extractions: List[HypergraphExtraction] = []

        for sent in sentences:
            entities = self._fallback_entities(sent)
            if not entities:
                continue
            relation = "CoOccurrence" if len(entities) > 1 else "Mention"
            extractions.append(HypergraphExtraction(
                hyperedge=sent[:240],
                relation_type=relation,
                entities=entities,
                confidence_score=5.0,
            ))

        if not extractions:
            doc_entity = Entity(
                name="Document",
                type="text",
                description=text_chunk[:240]
            )
            extractions.append(HypergraphExtraction(
                hyperedge=text_chunk[:240],
                relation_type="DocumentSummary",
                entities=[doc_entity],
                confidence_score=3.0,
            ))

        print(f"[fallback] Generated {len(extractions)} heuristic extractions")
        return extractions

    def _fallback_entities(self, sentence: str) -> List[Entity]:
        name_pattern = r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z][a-z]+)"
        raw_matches = re.findall(name_pattern, sentence)
        if not raw_matches:
            return []

        seen = set()
        names: List[str] = []
        for m in raw_matches:
            if m not in seen:
                seen.add(m)
                names.append(m)

        entities: List[Entity] = []
        for name in names:
            ent_type = self._infer_entity_type(name, sentence)
            entities.append(Entity(name=name, type=ent_type, description=sentence[:240]))
        return entities

    def _normalize_entity_type(self, raw_type: str | None, name: str, description: str | None, context_lower: str) -> str:
        candidate = (raw_type or "").strip().lower()
        if candidate in self.CANONICAL_TYPES:
            return candidate
        for canonical, payload in self.CANONICAL_TYPES.items():
            aliases = payload.get("aliases", set())
            if candidate in aliases:
                return canonical
        # Heuristic based on name/description keywords
        text = f"{name} {(description or '')}".lower()
        if any(tok in text for tok in [" governor", "president", "minister", "protocol", "ambassador", "chief", "captain"]):
            return "position"
        if any(tok in text for tok in [" university", " company", " studio", " society", " committee", " agency", " government", " department", " network"]):
            return "organization"
        if any(tok in text for tok in [" city", " village", " town", " district", " county", " province", " state", " country", " mosque", " mansion", " valley", " river"]):
            return "location"
        if any(tok in text for tok in [" film", " movie", " novel", " book", " series", " drama", " song", " album", " comic"]):
            return "work"
        if any(tok in text for tok in [" battle", " summit", " war", " ceremony", " festival"]):
            return "event"
        if any(tok in text for tok in [" born", " died", " 19", " 20", " century", " 18"]):
            return "person"
        if any(tok in text for tok in [" theory", " concept", " system", " process", " technology"]):
            return "concept"
        if re.fullmatch(r"\d{4}", name.strip()):
            return "date"
        # default fallback
        if "person" in context_lower or name.istitle():
            return "person"
        return "concept"

    def _extract_snippet(self, name: str, text_chunk: str) -> str:
        pattern = re.compile(r"[^.!?。]*" + re.escape(name) + r"[^.!?。]*(?:[.!?。]|$)", re.IGNORECASE)
        match = pattern.search(text_chunk)
        if match:
            snippet = match.group(0).strip()
            return snippet[:160] if len(snippet) > 160 else snippet
        idx = text_chunk.lower().find(name.lower())
        if idx != -1:
            start = max(0, idx - 80)
            end = min(len(text_chunk), idx + 120)
            snippet = text_chunk[start:end].strip()
            return snippet[:160] if len(snippet) > 160 else snippet
        return text_chunk[:160].strip()

    def _recover_partial_extractions(self, raw_json: str) -> List[HypergraphExtraction]:
        pattern = re.compile(r"\{\s*\"hyperedge\"[\s\S]*?\}\s*(?=,\s*\{\s*\"hyperedge\"|\s*\]\s*\}|$)")
        matches = pattern.findall(raw_json)
        if not matches:
            return []
        stitched = '{"extractions": [' + ','.join(matches) + ']}'
        try:
            data = json.loads(stitched)
            adapter = TypeAdapter(ExtractionResponse)
            validated = adapter.validate_python(data)
            return self._postprocess_extractions(validated.extractions, raw_json)
        except Exception:
            return []

    def _infer_entity_type(self, name: str, context: str) -> str:
        lowered = context.lower()
        if any(tok in lowered for tok in [" film", " movie", " drama", "comedy"]):
            if name.lower() in lowered:
                return "film"
        if any(tok in lowered for tok in [" director", " actor", "actress", "singer", "writer", "woman", "man", "person"]):
            return "person"
        if any(tok in lowered for tok in [" city", "neighborhood", "country", "state", "province"]):
            return "location"
        if any(tok in lowered for tok in [" government", "office", "position", "chief"]):
            return "role"
        return "entity"

if __name__ == '__main__':
    extractor = HypergraphExtractor()
    sample_document = """
    In Q4 2023, the tech giant InnovateCorp, led by its visionary CEO Dr. Evelyn Reed,
    announced a groundbreaking new product, the 'QuantumLeap Processor'. 
    This announcement, made during the annual TechSummit in Geneva,
    promises to revolutionize the field of quantum computing.
    The processor's development was a collaborative effort with the
    MIT Department of Physics.
    """

    try:
        extractions = extractor.extract(sample_document)
        if extractions:
            print(json.dumps([e.model_dump() for e in extractions], indent=2))
            # Export hyperedges artifact for downstream graph building
            os.makedirs('artifacts', exist_ok=True)
            out = []
            for idx, e in enumerate(extractions):
                text = e.hyperedge or ""
                uid = hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]
                out.append({
                    "id": f"h{idx}_{uid}",
                    "hyperedge": e.hyperedge,
                    "relation_type": e.relation_type,
                    "confidence_score": e.confidence_score,
                    "entities": [
                        {
                            "name": ent.name,
                            "type": ent.type,
                            "description": ent.description
                        } for ent in e.entities
                    ]
                })
            with open(os.path.join('artifacts', 'hyperedges.json'), 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"An error occurred: {e}")
