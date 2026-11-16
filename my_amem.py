import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import httpx
import numpy as np
from ollama import Client, ResponseError
from sentence_transformers import SentenceTransformer


MODEL_TIMEOUT_SECONDS = 75
MODEL_MAX_TOKENS = 512
MODEL_RETRIES = 3
logger = logging.getLogger(__name__)


class ModelCallFailed(RuntimeError):
    def __init__(self, stage: str, attempts: list[str]):
        super().__init__(f"{stage} failed after {len(attempts)} attempts: {attempts[-1]}")
        self.details = {"stage": stage, "attempts": attempts}


@dataclass
class MemoryNote:
    id: int
    speaker: str
    text: str
    timestamp: str
    keywords: list[str]
    tags: list[str]
    description: str
    embedding: list[float]
    links: list[int] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)

    def prompt_context(self) -> dict:
        return {
            "id": self.id,
            "speaker": self.speaker,
            "content": self.text,
            "time": self.timestamp,
            "keywords": self.keywords,
            "tags": self.tags,
            "description": self.description,
            "linked_memories": self.links.copy(),
        }


class MemoryStore:
    def __init__(
        self, client: Client, model: str, encoder: SentenceTransformer, *,
        evolve: bool = True, evolution_workers: int = 1,
    ):
        self.client = client
        self.model = model
        self.encoder = encoder
        self.evolve = evolve
        self.evolution_workers = evolution_workers
        self.notes: dict[int, MemoryNote] = {}
        self.seen: set[tuple[str, str, str]] = set()
        self.link_candidates: dict[int, list[dict]] = {}
        self.evolution_updates: list[dict] = []
        self.evolution_failures: list[dict] = []

    def ask(
        self, instructions: str, data: dict, schema: dict, temperature: float = 0, *,
        stage: str = "answer",
    ) -> dict:
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ]
        failures = []
        for attempt in range(1, MODEL_RETRIES + 2):
            try:
                result = self.client.chat(
                    model=self.model,
                    messages=messages,
                    format=schema,
                    options={"temperature": temperature, "num_predict": MODEL_MAX_TOKENS},
                )
                if result.done_reason == "length":
                    raise ValueError(f"generation reached the {MODEL_MAX_TOKENS}-token limit")
                payload = json.loads(result.message.content)
                if not isinstance(payload, dict) or not set(schema["required"]).issubset(payload):
                    raise ValueError("response is missing required JSON fields")
                return payload
            except (httpx.TransportError, ConnectionError, ResponseError, ValueError) as error:
                reason = f"{type(error).__name__}: {str(error) or 'no response'}"
                failures.append(reason)
                logger.warning("%s attempt %d/%d: %s", stage, attempt, MODEL_RETRIES + 1, reason)
        raise ModelCallFailed(stage, failures)

    def analyze(self, speaker: str, text: str, timestamp: str) -> dict:
        return self.ask(
            "Create metadata for a memory note. Extract specific keywords and broad "
            "topic tags. Write one contextual description grounded in the utterance. "
            "Keep the speaker, named people, and concrete events distinct. "
            "The time field is when the speaker talked. In the description, copy "
            "any relative time phrase verbatim (such as yesterday or on Saturday). "
            "Do not add a calendar date, motive, or detail absent from the content.",
            {"speaker": speaker, "content": text, "time": timestamp},
            {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["keywords", "tags", "description"],
            },
            stage="note_metadata",
        )

    def encode(self, text: str) -> list[float]:
        return self.encoder.encode(text, normalize_embeddings=True).tolist()

    def encode_note(
        self, text: str, keywords: list[str], tags: list[str], description: str
    ) -> list[float]:
        # Equation (3): encode the concatenation of c_i, K_i, G_i and X_i.
        return self.encode("\n".join([text, ", ".join(keywords), ", ".join(tags), description]))

    def rank(self, embedding: list[float], k: int) -> list[tuple[MemoryNote, float]]:
        if k < 1:
            raise ValueError("k must be at least 1")
        # Both vectors are normalized, so their dot product is cosine similarity.
        scored = [(note, float(np.dot(embedding, note.embedding))) for note in self.notes.values()]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:k]

    def choose_links(self, note: MemoryNote, candidates: list[dict]) -> list[int]:
        if not candidates:
            return []

        result = self.ask(
            "Read the new note and each candidate's original content. Select a candidate "
            "only if both notes describe the same concrete event or support the same fact. "
            "Similarity scores are for ranking, not evidence. Shared dates, speakers, "
            "or topics are insufficient. Return an empty list when there is no match.",
            {"new_note": note.prompt_context(), "candidates": candidates},
            {
                "type": "object",
                "properties": {"link_ids": {"type": "array", "items": {"type": "integer"}}},
                "required": ["link_ids"],
            },
            stage="link_generation",
        )
        allowed = {item["id"] for item in candidates}
        return sorted({note_id for note_id in result["link_ids"] if note_id in allowed})

    def evolution_decision(self, new_note: MemoryNote, candidates: list[dict], candidate: dict) -> dict:
        target_id = candidate["id"]
        return self.ask(
            "Decide whether the new note adds a concrete fact about the same event "
            "or fact described by the target note. Other neighbors are context only. "
            "Shared people, dates, or broad topics are insufficient. If there is no "
            "supported addition, set should_update to false and repeat the target metadata. "
            "If updating, return the complete revised tags and description. "
            "Write the description as a single concise factual sentence, with no commentary "
            "about editing or linking. Preserve supported facts and keep relative time "
            "phrases verbatim. Never copy dates from the time metadata into the description. "
            "For example, if content says yesterday and time says 8 May, write yesterday.",
            {
                "new_note": new_note.prompt_context(),
                "other_neighbors": [item for item in candidates if item["id"] != target_id],
                "target_note": candidate,
            },
            {
                "type": "object",
                "properties": {
                    "should_update": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["should_update", "tags", "description"],
            },
            stage=f"evolution_target_{target_id}",
        )

    def evolve_neighbors(self, new_note: MemoryNote, candidates: list[dict]) -> dict[int, MemoryNote]:
        def decide(candidate):
            try:
                return self.evolution_decision(new_note, candidates, candidate)
            except ModelCallFailed as error:
                return error

        # Each request sees the same snapshot. Apply the decisions in candidate order.
        workers = min(self.evolution_workers, len(candidates))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                decisions = list(pool.map(decide, candidates))
        else:
            decisions = [decide(candidate) for candidate in candidates]

        replacements = {}
        for candidate, result in zip(candidates, decisions):
            target = self.notes[candidate["id"]]
            if isinstance(result, ModelCallFailed):
                self.evolution_failures.append({
                    "trigger_note_id": new_note.id,
                    "source_turn_ids": new_note.source_turn_ids.copy(),
                    "target_note_id": target.id,
                    **result.details,
                })
                continue
            if not result["should_update"]:
                continue

            tags = result["tags"] or target.tags
            description = result["description"].strip() or target.description
            if tags == target.tags and description == target.description:
                continue
            replacements[target.id] = replace(
                target, tags=tags, description=description,
                embedding=self.encode_note(target.text, target.keywords, tags, description),
            )
        return replacements

    def add_note(
        self, speaker: str, text: str, timestamp: str, link_k: int = 5, *,
        source_turn_id: str | None = None,
    ) -> MemoryNote | None:
        key = (speaker, text.strip(), timestamp)
        if not key[1]:
            return None
        if key in self.seen:
            if source_turn_id:
                for existing in self.notes.values():
                    if (existing.speaker, existing.text.strip(), existing.timestamp) == key:
                        if source_turn_id not in existing.source_turn_ids:
                            existing.source_turn_ids.append(source_turn_id)
                        break
            return None

        metadata = self.analyze(speaker, text, timestamp)
        note = MemoryNote(
            id=len(self.notes) + 1,
            speaker=speaker,
            text=text,
            timestamp=timestamp,
            **metadata,
            embedding=self.encode_note(text, **metadata),
            source_turn_ids=[source_turn_id] if source_turn_id else [],
        )
        candidates = [
            {**neighbor.prompt_context(), "similarity": score}
            for neighbor, score in self.rank(note.embedding, link_k)
        ]
        note.links = self.choose_links(note, candidates)
        replacements = self.evolve_neighbors(note, candidates) if self.evolve else {}

        for target_id, updated in replacements.items():
            previous = self.notes[target_id]
            self.evolution_updates.append({
                "trigger_id": note.id,
                "target_id": target_id,
                "before": previous.prompt_context(),
                "after": updated.prompt_context(),
                "embedding_changed": previous.embedding != updated.embedding,
            })
            self.notes[target_id] = updated
        self.notes[note.id] = note
        self.seen.add(key)
        self.link_candidates[note.id] = candidates
        for related_id in note.links:
            self.notes[related_id].links.append(note.id)
        return note

    def answer(self, question: str, k: int) -> dict:
        retrieved = [
            {**note.prompt_context(), "similarity": score}
            for note, score in self.rank(self.encode(question), k)
        ]
        result = self.ask(
            "Answer using the retrieved notes. If the relevant utterance says "
            "'yesterday', subtract one calendar day from its conversation date. "
            "Give a short answer, or say unknown if the evidence is insufficient.",
            {"question": question, "retrieved_notes": retrieved},
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        )
        return {"question": question, "retrieved": retrieved, "answer": result["answer"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--link-k", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # Use the local Ollama service even when a system proxy is configured.
    client = Client(host="http://127.0.0.1:11434", trust_env=False, timeout=MODEL_TIMEOUT_SECONDS)
    store = MemoryStore(client, args.model, SentenceTransformer(args.encoder))
    turns = [
        ("Caroline", "I went to a LGBTQ support group yesterday.", "8 May 2023"),
        ("Caroline", "At yesterday's LGBTQ support group, I met the organizer Alex.", "8 May 2023"),
        ("Melanie", "I bought a bicycle on Saturday.", "8 May 2023"),
    ]
    for turn in turns:
        store.add_note(*turn, link_k=args.link_k)
    duplicate_skipped = store.add_note(*turns[0], link_k=args.link_k) is None
    assert duplicate_skipped and len(store.notes) == len(turns)
    for note in store.notes.values():
        assert all(note.id in store.notes[neighbor].links for neighbor in note.links)

    snapshot = {
        "model": args.model,
        "encoder": args.encoder,
        "notes": [asdict(note) for note in store.notes.values()],
        "link_candidates": store.link_candidates,
        "evolution_updates": store.evolution_updates,
        "evolution_failures": store.evolution_failures,
        "qa": store.answer("When did Caroline go to the LGBTQ support group?", args.retrieve_k),
        "duplicate_skipped": duplicate_skipped,
    }
    output = json.dumps(snapshot, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
