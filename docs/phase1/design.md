# Design (Phase 1)

## Architecture and Levels
- Levels: L0 (entities, hyperedges), L1 (summaries via BERTopic soft clustering), L2 (summaries via community roll-up), Lk (extensible).
- Node types: entity | hyperedge | summary; Edge types: participates_in | belongs_to | related_to.
- DAG: belongs_to edges form a DAG across layers; multi-parent allowed.

## Data Semantics
- entity: name, description, entity_type, l1_parents, embedding_ref.
- hyperedge: description (hyperedge sentence), relation_type, confidence_score, source_text_ref.
- summary: title, summary_text, top_words, members (children), centroid, confidence, judge_scores, level.

## Quality Controls
- LLM-as-a-Judge for sampled nodes/edges (consistency/accuracy/informativeness/overall).
- Escalation metrics for L2+: compression, coverage, judge improvement; stop if not met.
