"""
NLP Pipeline for object extraction and spatial relation parsing.

Implements the full pipeline:
  1. Normalize text (lowercase, lemmatization)
  2. Extract objects (nouns) + attributes (adjectives) via POS tagging
  3. Expand each object with WordNet synonyms, filter by embedding similarity
  4. Extract spatial relations with linguistic rules → relation(subj, type, obj)
  5. Build a symbolic goal representation
  6. Produce ranked candidate search phrases for RADIO/SigLIP scoring

Only standard NLP tools are used (spaCy + WordNet). No LLM/VLM.

Usage (standalone test)
-----------------------
    python test/nlp_pipeline.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
import spacy
from nltk.corpus import wordnet as wn

if TYPE_CHECKING:
    from src.features.extractor import FeatureExtractor

# Load once at import time
_nlp = spacy.load("en_core_web_sm")

# Spatial relation patterns (regex on lemmatized token sequences)
SPATIAL_PATTERNS = [
    # Directional
    (r"\bleft\s+of\b",           "left_of"),
    (r"\bright\s+of\b",          "right_of"),
    (r"\babove\b",                "above"),
    (r"\bbelow\b",                "below"),
    (r"\bunder(?:neath)?\b",      "below"),
    (r"\bon\s+top\s+of\b",        "above"),
    (r"\bon\b",                   "on"),
    (r"\bnext\s+to\b",            "next_to"),
    (r"\badjacent\s+to\b",        "next_to"),
    (r"\bbeside\b",               "next_to"),
    (r"\bnear(?:by)?\b",          "near"),
    (r"\bclose\s+to\b",           "near"),
    (r"\bwithin\b",               "near"),
    (r"\bin\s+front\s+of\b",      "in_front_of"),
    (r"\bbehind\b",               "behind"),
    (r"\bbetween\b",              "between"),
    (r"\binside\b",               "inside"),
    (r"\bopposite\b",             "opposite"),
    (r"\bstacked\s+on\b",         "above"),
    (r"\bmounted\s+on\b",         "on"),
    (r"\bhanging\s+on\b",         "on"),
]
_SPATIAL_RE = [(re.compile(p, re.I), label) for p, label in SPATIAL_PATTERNS]


# Spatial direction words that spaCy may parse as noun chunks (pobj of "to the right of")
# but are NOT real objects — filter them from landmarks
_SPATIAL_NOUNS = {
    "right", "left", "top", "bottom", "front", "back", "side",
    "middle", "center", "centre", "corner", "edge", "end",
}

# Abstract / non-visual words that spaCy extracts from instructions as noun chunks
# but have no meaningful visual representation for SigLIP — skip as landmarks.
_NON_VISUAL_WORDS = {
    # spatial/locative abstractions
    "area", "region", "vicinity", "zone", "location", "place", "spot", "space",
    # material / property nouns from descriptions
    "steel", "wood", "glass", "metal", "plastic", "fabric", "material",
    # appearance / style words
    "appearance", "style", "color", "colour", "size", "shape", "type", "kind",
    # generic scene words
    "scene", "object", "thing", "item", "piece", "part", "section",
    # structural/architectural words that appear generically
    "wall", "floor", "ceiling", "room", "area",
    # body/clothing parts often misextracted
    "thread", "arch", "robe", "fixture",
    # functional words
    "use", "purpose", "function", "view",
}

# Data structures

@dataclass
class ObjectMention:
    """A noun phrase extracted from the instruction."""
    head: str                          # lemmatized head noun, e.g. "bed"
    attributes: list[str]              # adjective modifiers, e.g. ["king size", "white"]
    span_text: str                     # original span text from the sentence
    is_target: bool = False            # True if this is the primary search target


@dataclass
class SpatialRelation:
    """A spatial relation between two object mentions."""
    subject: ObjectMention
    relation_type: str                 # e.g. "left_of", "near", "above"
    reference: Optional[ObjectMention] # the anchor object (can be None for "between")


@dataclass
class GoalRepresentation:
    """Symbolic goal: target object + optional spatial constraints."""
    target: ObjectMention
    relations: list[SpatialRelation] = field(default_factory=list)
    landmarks: list[ObjectMention] = field(default_factory=list)


# Core pipeline

class NLPPipeline:
    """
    Full NLP pipeline: instruction text → GoalRepresentation.

    Steps:
      1. Normalize (lowercase)
      2. spaCy parse → POS tags, dependency tree, lemmas
      3. Extract noun chunks (objects + attributes)
      4. Identify target object (first / most salient noun chunk)
      5. Extract spatial relations between objects
      6. Build GoalRepresentation
    """

    def __init__(self):
        pass

    def build(
        self,
        instruction: str,
        extractor: "FeatureExtractor",
        syn_sim_threshold: float = 0.60,
        syn_top_k: int = 5,
    ) -> tuple:
        """
        Full pipeline: instruction → (target_phrase, text_emb, nlp_object_embs).

        Parameters
        ----------
        instruction       : raw instruction string
        extractor         : FeatureExtractor for SigLIP text encoding
        syn_sim_threshold : min cosine sim to keep a WordNet synonym
        syn_top_k         : max synonyms per landmark

        Returns
        -------
        target_phrase   : str — e.g. "king size bed"
        text_emb        : np.ndarray (D,) — SigLIP embedding of target_phrase
        nlp_object_embs : dict[str, tuple[list[str], np.ndarray(N,D)]]
                          one entry per landmark: (phrases, embeddings)
        """
        goal_repr = self.parse(instruction)

        # Target phrase: use span_text (the original noun chunk text) stripped of
        # leading determiners. This preserves compound nouns like "washing machine",
        # "fire extinguisher", "bath towel" that spaCy correctly groups into one chunk
        # but whose head noun alone ("machine", "extinguisher") is too generic.
        # We fall back to head-only if span_text is long (likely an over-extended chunk
        # caused by a relative clause) or contains adjectives that hurt visual sim.
        t = goal_repr.target
        # Normalize hyphens so "washer-dryer" becomes two words
        span_words = t.span_text.lower().replace("-", " ").split()
        while span_words and span_words[0] in ("the", "a", "an"):
            span_words = span_words[1:]
        # Keep span only when it is 1-3 words with no coordinating conjunctions
        # (those usually indicate over-chunking like "bed, pillow, and curtain")
        if 1 < len(span_words) <= 3 and "and" not in span_words and "or" not in span_words:
            target_phrase = " ".join(span_words)
        else:
            target_phrase = t.head
        text_emb = extractor.encode_text(target_phrase)

        # Landmark embeddings: WordNet candidates filtered by SigLIP similarity.
        # Skip non-visual abstract words that spaCy extracts from the instruction
        # (e.g. "area", "vicinity", "steel", "region", "appearance") — they are
        # useless for visual matching and dilute the confirmation signal.
        nlp_object_embs = {}
        for lm in goal_repr.landmarks:
            if lm.head in nlp_object_embs:
                continue
            if lm.head in _NON_VISUAL_WORDS:
                continue
            seen: set[str] = {lm.head}
            candidates: list[str] = []
            for syn in wn.synsets(lm.head, pos=wn.NOUN):
                for ln in syn.lemma_names():
                    w = ln.replace("_", " ").lower()
                    if w not in seen:
                        seen.add(w)
                        candidates.append(w)
            q_emb = extractor.encode_text(lm.head)
            syns = [w for w in candidates
                    if float(q_emb @ extractor.encode_text(w)) >= syn_sim_threshold
                    ][:syn_top_k]
            phrases = [lm.head] + syns
            embs = np.stack([extractor.encode_text(p) for p in phrases])
            nlp_object_embs[lm.head] = (phrases, embs)

        return target_phrase, text_emb, nlp_object_embs

    def parse(self, instruction: str) -> GoalRepresentation:
        """Parse an instruction string into a GoalRepresentation."""
        text = instruction.lower().strip().rstrip(".")
        doc = _nlp(text)

        # Step 1: extract noun chunks
        objects = self._extract_objects(doc)
        if not objects:
            # Fallback: use the raw instruction as a single target
            fallback = ObjectMention(
                head=text, attributes=[], span_text=text, is_target=True)
            return GoalRepresentation(target=fallback)

        # Step 2: identify target -- first subject noun or first before spatial keyword
        target = self._identify_target(objects, doc)
        target.is_target = True

        # Step 3: extract spatial relations
        relations = self._extract_relations(doc, objects, target)

        # Step 4: collect all non-target objects as landmarks
        landmarks = [o for o in objects if o is not target]

        return GoalRepresentation(target=target, relations=relations, landmarks=landmarks)

    # Private helpers

    def _extract_objects(self, doc) -> list[ObjectMention]:
        """Extract noun chunks as ObjectMention instances."""
        objects = []
        seen_heads: set[str] = set()

        for chunk in doc.noun_chunks:
            head_lemma = chunk.root.lemma_
            # Skip pronouns and very short stopword-only chunks
            if chunk.root.pos_ not in ("NOUN", "PROPN"):
                continue
            if chunk.root.is_stop:
                continue
            # Skip spatial direction words that spaCy captures as pobj
            # e.g. "to the right of" → chunk "the right", head "right"
            if head_lemma in _SPATIAL_NOUNS:
                continue
            if head_lemma in seen_heads:
                continue
            seen_heads.add(head_lemma)

            # Collect adjective modifiers within the chunk, preserving conjunctions.
            # Example: "red and yellow cloth" → tokens [red(ADJ), and(cc), yellow(conj/ADJ)]
            # We want attrs = ["red and yellow"], not ["red", "yellow"].
            attrs = []
            i = 0
            toks = list(chunk)
            while i < len(toks):
                tok = toks[i]
                if tok.pos_ == "ADJ" and not tok.is_stop:
                    # Collect any coordinated adjectives via cc/conj
                    adj_parts = [tok.lemma_]
                    j = i + 1
                    while j < len(toks):
                        if toks[j].dep_ == "cc":          # "and" / "or"
                            adj_parts.append(toks[j].text)
                            j += 1
                        elif toks[j].dep_ == "conj" and toks[j].pos_ == "ADJ":
                            adj_parts.append(toks[j].lemma_)
                            j += 1
                        else:
                            break
                    attrs.append(" ".join(adj_parts))
                    i = j
                elif tok.dep_ == "compound" and tok.pos_ == "NOUN":
                    attrs.append(tok.lemma_)
                    i += 1
                else:
                    i += 1

            obj = ObjectMention(
                head=head_lemma,
                attributes=attrs,
                span_text=chunk.text,
            )
            objects.append(obj)

        return objects

    def _identify_target(self, objects: list[ObjectMention], doc) -> ObjectMention:
        """
        Identify the primary target object.
        Priority: syntactic subject > first object before a spatial keyword.
        """
        # Try syntactic subject
        for obj in objects:
            for tok in doc:
                if tok.lemma_ == obj.head and tok.dep_ in ("nsubj", "nsubjpass"):
                    return obj

        # Fall back: first object that appears before any spatial keyword
        spatial_tokens = {tok.i for tok in doc
                          if any(re.search(p, tok.text, re.I) for p, _ in SPATIAL_PATTERNS)}
        if spatial_tokens:
            cutoff = min(spatial_tokens)
            for obj in objects:
                for tok in doc:
                    if tok.lemma_ == obj.head and tok.i < cutoff:
                        return obj

        # Default: first object
        return objects[0]

    def _extract_relations(
        self,
        doc,
        objects: list[ObjectMention],
        target: ObjectMention,
    ) -> list[SpatialRelation]:
        """Extract spatial relations from the sentence text."""
        text = doc.text
        relations = []

        for regex, rel_type in _SPATIAL_RE:
            for m in regex.finditer(text):
                # Find which object comes after the spatial keyword
                after_text = text[m.end():].strip()
                reference = self._find_object_in_text(after_text, objects, exclude=target)
                relation = SpatialRelation(
                    subject=target,
                    relation_type=rel_type,
                    reference=reference,
                )
                relations.append(relation)
                break  # one match per relation type

        return relations

    def _find_object_in_text(
        self,
        text_fragment: str,
        objects: list[ObjectMention],
        exclude: ObjectMention,
    ) -> Optional[ObjectMention]:
        """Find the best-matching object whose head appears in text_fragment."""
        for obj in objects:
            if obj is exclude:
                continue
            if obj.head in text_fragment or obj.span_text in text_fragment:
                return obj
        return None


# Standalone test

if __name__ == "__main__":
    pipeline = NLPPipeline()

    test_cases = [
        # (task_id, object_category, instruction_text)
        ("X",  "red and yellow cloth",
         "red and yellow cloth located to the right of the picture."),
        (1,  "bed",
         "king size bed located near the chest drawer, painting, curtain, and pillow."),
        (9,  "oven and stove",
         "oven and stove, which are located in the kitchen area. the oven and stove are "
         "made of stainless steel. look for them in the vicinity of the kitchen cabinet "
         "and kitchen counter."),
        (8,  "kitchen lower cabinet",
         "kitchen lower cabinet, a white cabinet with a glass door. referencing other "
         "objects in the scene, locate the cabinet below the kitchen cabinet, to the "
         "right of the refrigerator cabinet, and to the left of the bench."),
        (10, "table",
         "table with a lamp on it. the table is located near the bed, pillow, and picture."),
        (3,  "tv",
         "flat screen tv that is located to the left of the cabinet."),
        (5,  "cabinet",
         "the cabinet with a mirror on it, which is located next to the sink and clutter."),
        (2,  "bed",
         "queen-sized bed located near the dresser and the book. look for a bed that is "
         "positioned slightly below the dresser and to the right of the book."),
    ]

    print("=" * 70)
    print("NLP PIPELINE — TEST OUTPUT")
    print("=" * 70)

    for task_id, category, instruction in test_cases:
        print(f"\n[task {task_id}]  category='{category}'")
        print(f"  instruction: '{instruction[:80]}...'")
        print()

        goal = pipeline.parse(instruction)
        t = goal.target
        phrase = ((" ".join(t.attributes) + " ") if t.attributes else "") + t.head
        print(f"  target_phrase : '{phrase}'")
        print(f"  landmarks     : {[l.head for l in goal.landmarks]}")
        print("-" * 70)
