import argparse
import hashlib
import io
import json
import random
import re
import sys
import unicodedata
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

from my_amem import (
    MODEL_MAX_TOKENS, MODEL_RETRIES, MODEL_TIMEOUT_SECONDS, Client, MemoryNote,
    MemoryStore, ModelCallFailed, SentenceTransformer,
)


# A-Mem Table 8, Qwen2.5-3B.
QA_K = {1: 10, 2: 10, 3: 50, 4: 10, 5: 10}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}
GROUNDED_SINGLE_HOP_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_note_id": {"type": "integer"},
        "evidence_quote": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["evidence_note_id", "evidence_quote", "answer"],
}
GROUNDED_SINGLE_HOP_INSTRUCTIONS = (
    "Identify the person, event, and any time or place constraints named in the question. "
    "Find the original content of one note that directly supports the requested fact. "
    "Use the note's speaker and time to check who spoke and when. "
    "Put its id in evidence_note_id and copy the shortest supporting passage verbatim "
    "from its content into evidence_quote. Then answer with just the requested name, "
    "frequency, activity, or other short phrase supported by that passage. "
    "If no original passage supports the fact, use -1 for evidence_note_id, "
    "an empty evidence_quote, and unknown for answer."
)
RETRIEVAL_KEYWORDS_SCHEMA = {
    "type": "object",
    "properties": {"keywords": {"type": "string"}},
    "required": ["keywords"],
}
OPEN_DOMAIN_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_items": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["answer_items"],
}
TEMPORAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "conversation_time": {"type": "string"},
        "time_expression": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["conversation_time", "time_expression", "answer"],
}
TEMPORAL_INSTRUCTIONS = (
    "Use the original content and conversation time from the same note. Its time field "
    "is when the sentence was spoken. Pay attention to time expressions in the sentence, "
    "including explicit dates and relative phrases such as yesterday or two days ago. "
    "Resolve relative time against the conversation date of the note containing that phrase. "
    "For example, if a note is dated 14 June 2022 and says the event happened yesterday, "
    "the event date is 13 June 2022. Use the conversation timestamp itself only if the "
    "question asks when the conversation took place or the evidence places the event then. "
    "Fill conversation_time with the relevant conversation time and time_expression "
    "with the exact time words from the original content (or none). If the question "
    "asks when, fill answer with the event time, using D Month YYYY for an exact date. "
    "If it asks who, what, or where at a given time, use the date to find the right "
    "event and answer with the requested person, fact, or place. Keep answer short. "
    "Use unknown if the evidence is insufficient."
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def retrieval_query(store, question):
    try:
        result = store.ask(
            "Extract search keywords from the question. Keep names, dates, negations, events, "
            "and relationship clues needed for every part of the question. "
            "Return a comma-separated list in keywords. Do not answer or add facts.",
            {"question": question}, RETRIEVAL_KEYWORDS_SCHEMA, stage="retrieval_keywords",
        )
    except ModelCallFailed as error:
        return question, error.details
    keywords = result["keywords"]
    if isinstance(keywords, str) and keywords.strip():
        return keywords.strip(), None
    return question, {"stage": "retrieval_keywords", "reason": "No usable keywords"}


def expand_linked_notes(store, ranked, limit):
    seen = {note.id for note, _ in ranked}
    if not limit:
        return []
    linked = []
    # Visit the first link of each ranked note before visiting their second links.
    for neighbors in zip_longest(*(note.links for note, _ in ranked)):
        for (source, _), neighbor_id in zip(ranked, neighbors):
            if neighbor_id in seen or neighbor_id not in store.notes:
                continue
            seen.add(neighbor_id)
            linked.append({**store.notes[neighbor_id].prompt_context(), "linked_from": source.id})
            if len(linked) == limit:
                return linked
    return linked


def answer_question(
    store, qa, rng, temperature_override=None, *, temporal_prompt="relative", linked_note_limit=10,
    ground_single_hop=False,
):
    k = QA_K[qa.category]
    query, query_failure = retrieval_query(store, qa.question)
    ranked = store.rank(store.encode(query), k)
    linked_by_source = {}
    for neighbor in expand_linked_notes(store, ranked, linked_note_limit):
        linked_by_source.setdefault(neighbor["linked_from"], []).append(neighbor)
    retrieved = []
    for note, score in ranked:
        retrieved.append({**note.prompt_context(), "similarity": score})
        retrieved.extend(linked_by_source.get(note.id, ()))
    response, failure = generate_answer(
        store, qa, rng, retrieved, temperature_override,
        temporal_prompt=temporal_prompt, ground_single_hop=ground_single_hop,
    )
    for record in retrieved:
        record["source_turn_ids"] = store.notes[record["id"]].source_turn_ids
    details = {
        "retrieval_query": query,
        "retrieval_query_failure": query_failure,
        "initial_retrieved_count": len(ranked),
        "linked_note_count": len(retrieved) - len(ranked),
    }
    if qa.category == 4 and ground_single_hop:
        details.update(citation_diagnostics(qa, response, retrieved))
    return response, k, retrieved, failure, details


def generate_answer(store, qa, rng, retrieved, temperature_override=None, *,
                    temporal_prompt="relative", ground_single_hop=False):
    data = {"question": qa.question, "retrieved_notes": retrieved}
    instructions = (
        "Answer the question using the retrieved conversation. Give a short phrase, "
        "using the exact words from the conversation whenever possible."
    )
    temperature = 0.7
    schema = ANSWER_SCHEMA
    if qa.category == 2:
        if temporal_prompt == "baseline":
            instructions += " Use the date of the conversation to answer with an approximate date."
        else:
            instructions = TEMPORAL_INSTRUCTIONS
            schema = TEMPORAL_ANSWER_SCHEMA
    elif qa.category == 3:
        instructions = (
            "Answer using the original utterances. Put one short answer phrase per item "
            "in answer_items. For one fact return exactly one item. For multiple facts "
            "return only their short phrases as separate items. Reuse the exact words "
            "from the utterances when possible. Omit introductions, explanations, and summaries."
        )
        schema = OPEN_DOMAIN_ANSWER_SCHEMA
    elif qa.category == 4 and ground_single_hop:
        instructions = GROUNDED_SINGLE_HOP_INSTRUCTIONS
        schema = GROUNDED_SINGLE_HOP_SCHEMA
    elif qa.category == 5:
        options = ["Not mentioned in the conversation", qa.final_answer]
        if rng.random() >= 0.5:
            options.reverse()
        data["options"] = options
        instructions = "Using the retrieved conversation, select one of the two answer options."
        temperature = 0.5
    if temperature_override is not None:
        temperature = temperature_override
    failure = None
    try:
        response = store.ask(instructions, data, schema, temperature=temperature)
        if qa.category == 3:
            items = response["answer_items"]
            if not isinstance(items, list) or not items or any(
                not isinstance(item, str) or not item.strip() for item in items
            ):
                raise ModelCallFailed("answer", ["Expected a nonempty list of answer phrases"])
            response["answer"] = ", ".join(item.strip() for item in items)
    except ModelCallFailed as error:
        response = {"answer": ""}
        failure = error.details
    return response, failure


def citation_diagnostics(qa, response, retrieved):
    citation = next((record for record in retrieved if record["id"] == response.get("evidence_note_id")), None)
    quote = response.get("evidence_quote")
    return {
        "citation_quote_found": bool(
            citation and isinstance(quote, str) and quote.strip() and quote.strip() in citation["content"]
        ),
        "citation_matches_gold_turn": bool(
            citation and set(qa.evidence).intersection(citation["source_turn_ids"])
        ),
    }


def parse_date(text):
    for pattern in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(text).strip().replace(",", ""), pattern).date()
        except ValueError:
            continue
    return None


def relaxed_answer_match(question, category, prediction, reference, strict_correct, metrics):
    """Recognize equivalent surface forms while retaining the original score."""
    if strict_correct:
        return "exact_or_date_match"
    if category == 5:
        return "adversarial_refusal" if metrics.get("adversarial_correct") else None
    if not prediction or not reference:
        return None

    def normalize(text):
        return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split()).rstrip(".!?")

    def list_items(text):
        return sorted(
            item.strip() for item in re.split(r"\s*,\s*(?:and\s+)?|\s*;\s*|\s+and\s+", text)
            if item.strip()
        )

    question = normalize(question)
    prediction = normalize(prediction)
    reference = normalize(reference)
    ordered = category == 2 or re.search(
        r"\b(first|second|last|before|after|order|sequence|chronological|respectively)\b", question
    )

    def equivalent(left, right):
        if left == right:
            return "surface_form"
        if not ordered:
            items = list_items(left)
            if len(items) > 1 and items == list_items(right):
                return "equivalent_list"
        return None

    result = equivalent(prediction, reference)
    if result:
        return result

    # Strip an answer wrapper only if it adds no facts. A subject/verb wrapper must
    # repeat the subject and predicate from the question.
    preamble = re.fullmatch(r"(?:the answer is |answer:\s*)(.+)", prediction)
    if preamble:
        return equivalent(preamble[1], reference)
    sentence = re.fullmatch(
        r"(?P<subject>[\w' -]+?)\s+(?:(?:has|have|had)\s+)?"
        r"(?P<verb>done|visited|tried|practiced|attended|played|learned|made|seen|bought|studied)"
        r"\s+(?P<answer>.+)",
        prediction,
    )
    if sentence:
        subject = sentence["subject"]
        verb = sentence["verb"]
        if (
            re.search(rf"(?<!\w){re.escape(subject)}\s+{re.escape(verb)}(?!\w)", question)
            and not re.search(r"\b(no|not|never)\b|n't\b", subject)
            and equivalent(sentence["answer"], reference)
        ):
            return "question_framing"
    return None


def compare_evidence_and_answer(qa, prediction, reference, retrieved, metrics):
    expected = set(qa.evidence)
    retrieved_ids = {source for note in retrieved for source in note["source_turn_ids"]}
    matched = sorted(expected & retrieved_ids)
    correct = bool(metrics["exact_match"])
    if qa.category == 2:
        predicted_date = parse_date(prediction)
        correct = correct or (predicted_date is not None and predicted_date == parse_date(reference))
    match_rule = relaxed_answer_match(qa.question, qa.category, prediction, reference, correct, metrics)
    return {
        "matched_evidence": matched,
        "evidence_hit": bool(matched) if expected else None,
        "evidence_recall": len(matched) / len(expected) if expected else None,
        "answer_correct": correct,
        "answer_correct_relaxed": match_rule is not None,
        "relaxed_match_rule": match_rule,
    }


def score_with_diagnostics(scoring, prediction, qa, legacy_c5):
    # The author utility logs some metric failures and returns zero. Keep valid scores
    # while excluding those fallback values from the aggregate.
    messages = io.StringIO()
    with redirect_stdout(messages):
        reference, metrics = scoring.score_prediction(prediction, qa, legacy_c5)
    warnings = messages.getvalue().splitlines()
    for metric_prefix, error_text in (
        ("bert_", "Error calculating BERTScore"),
        ("meteor", "Error calculating METEOR"),
        ("sbert_similarity", "Error calculating sentence similarity"),
    ):
        failed = any(error_text in message for message in warnings)
        if metric_prefix == "sbert_similarity" and scoring.sentence_model is None:
            failed = True
            warnings.append("SBERT scoring model unavailable")
        if failed:
            metrics = {name: value for name, value in metrics.items() if not name.startswith(metric_prefix)}
    return reference, metrics, warnings


def run_sample(args, sample, client, encoder, config, checkpoint, scoring):
    store = MemoryStore(
        client, args.model, encoder, evolve=not args.disable_evolution,
        evolution_workers=args.evolution_workers,
    )
    state = {
        "config": config, "next_turn": 0, "notes": [], "results": [],
        "evolved_notes": 0, "evolution_failures": [],
        "failed_turns": [], "note_write_failures": [],
    }
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["config"] != config:
            raise ValueError(f"Checkpoint configuration changed. Choose a new --output: {checkpoint}")
        for record in state["notes"]:
            note = MemoryNote(**record)
            store.notes[note.id] = note
            store.seen.add((note.speaker, note.text.strip(), note.timestamp))
        print(
            f"Sample {sample.sample_id}: resumed at turn {state['next_turn']}, "
            f"{len(state['results'])} questions scored",
            flush=True,
        )

    turns = [
        (turn.speaker, turn.text, session.date_time, turn.dia_id)
        for session in sorted(sample.conversation.sessions.values(), key=lambda session: session.session_id)
        for turn in session.turns
    ]
    turns = turns[:args.max_turns]
    questions = [(index, qa) for index, qa in enumerate(sample.qa) if qa.category in QA_K]
    questions = questions[:args.max_questions]

    def write_turn(index):
        speaker, text, timestamp, dia_id = turns[index]
        store.add_note(speaker, text, timestamp, link_k=args.link_k, source_turn_id=dia_id)

    def save_turn_progress():
        state["evolved_notes"] += len(store.evolution_updates)
        state["evolution_failures"].extend(store.evolution_failures)
        # Include earlier notes because a write can evolve them.
        state["notes"] = [asdict(note) for note in store.notes.values()]
        write_json(checkpoint, state)
        store.link_candidates.clear()
        store.evolution_updates.clear()
        store.evolution_failures.clear()

    for index in range(state["next_turn"], len(turns)):
        try:
            write_turn(index)
        except ModelCallFailed as error:
            state["note_write_failures"].append({
                "sample_id": sample.sample_id, "turn_index": index, "source_turn_id": turns[index][3],
                "initial_failure": error.details,
            })
            print(f"Sample {sample.sample_id}: turn {index + 1}/{len(turns)} deferred: {error}", flush=True)
        state["next_turn"] = index + 1
        save_turn_progress()
        print(f"Sample {sample.sample_id}: turn {index + 1}/{len(turns)} saved", flush=True)

    for record in state["note_write_failures"]:
        if "retry" in record:
            continue
        index = record["turn_index"]
        try:
            write_turn(index)
        except ModelCallFailed as error:
            record["retry"] = {"status": "failed", **error.details}
            state["failed_turns"].append({
                "sample_id": record["sample_id"], "turn_index": index,
                "source_turn_id": record["source_turn_id"], **error.details,
            })
            print(f"Sample {sample.sample_id}: turn {index + 1} still failed after deferred retry: {error}", flush=True)
        else:
            record["retry"] = {"status": "recovered"}
            print(f"Sample {sample.sample_id}: turn {index + 1} recovered on deferred retry", flush=True)
        save_turn_progress()

    if args.build_only:
        return state

    for position in range(len(state["results"]), len(questions)):
        question_index, qa = questions[position]
        rng = random.Random(f"{args.seed}:{sample.sample_id}:{question_index}")
        response, k, retrieved, failure, retrieval_details = answer_question(
            store, qa, rng, args.answer_temperature, temporal_prompt=args.temporal_prompt,
            linked_note_limit=args.linked_note_limit, ground_single_hop=args.ground_single_hop,
        )
        prediction = response["answer"]
        reference, metrics, warnings = score_with_diagnostics(scoring, prediction, qa, args.legacy_c5_reference)
        initial_evidence = compare_evidence_and_answer(
            qa, prediction, reference, [note for note in retrieved if "linked_from" not in note], metrics
        )
        state["results"].append({
            "sample_id": sample.sample_id,
            "question_index": question_index,
            "category": qa.category,
            "question": qa.question,
            "evidence": qa.evidence,
            "prediction": prediction,
            "answer_fields": response,
            "reference": reference,
            "retrieve_k": k,
            "retrieved_notes": retrieved,
            **retrieval_details,
            "initial_matched_evidence": initial_evidence["matched_evidence"],
            "initial_evidence_recall": initial_evidence["evidence_recall"],
            "metrics": metrics,
            **compare_evidence_and_answer(qa, prediction, reference, retrieved, metrics),
            "scoring_warnings": warnings,
            "model_failure": failure,
        })
        write_json(checkpoint, state)
        print(f"Sample {sample.sample_id}: question {position + 1}/{len(questions)} saved", flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description="Evaluate the independent A-Mem implementation on LoCoMo.")
    parser.add_argument("--author-dir", type=Path, default=Path("D:/paper/A-Mem/A-mem-main"))
    parser.add_argument("--dataset", type=Path, help="Defaults to the author's data/locomo10.json")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--evolution-workers", type=int, default=1)
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--samples", type=int, default=1, help="Number of conversations to process")
    parser.add_argument("--sample-start", type=int, default=1, help="First conversation, numbered from 1")
    parser.add_argument("--build-only", action="store_true", help="Save the memory checkpoint before answering")
    parser.add_argument("--max-turns", type=int, help="Limit turns per conversation for a smoke run")
    parser.add_argument("--max-questions", type=int, help="Limit questions per conversation for a smoke run")
    parser.add_argument("--link-k", type=int, default=5)
    parser.add_argument("--linked-note-limit", type=int, default=10, help="Maximum additional notes from one-hop links per question")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy-c5-reference", action="store_true")
    parser.add_argument("--disable-evolution", action="store_true")
    parser.add_argument("--answer-temperature", type=float, help="Override answer generation temperature")
    parser.add_argument(
        "--ground-single-hop", action="store_true",
        help="Have single-hop answers identify and quote the supporting original utterance",
    )
    parser.add_argument("--temporal-prompt", choices=("relative", "baseline"), default="relative")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value is not None and value < 1 for value in
           (args.samples, args.sample_start, args.link_k, args.max_turns, args.max_questions, args.evolution_workers)):
        parser.error("Sample, turn, question and link counts must be positive.")
    if args.linked_note_limit < 0:
        parser.error("--linked-note-limit must be nonnegative.")

    author_dir = args.author_dir.resolve()
    dataset = (args.dataset or author_dir / "data" / "locomo10.json").resolve()
    sys.path.insert(0, str(author_dir))
    from load_dataset import load_locomo_dataset
    import utils as scoring

    samples = load_locomo_dataset(dataset)
    sample_start = args.sample_start - 1
    sample_end = sample_start + args.samples
    if sample_end > len(samples):
        parser.error(f"Requested conversations {args.sample_start} to {sample_end}; the dataset has {len(samples)}.")
    sources = [Path(__file__), Path(__file__).with_name("my_amem.py"),
               author_dir / "load_dataset.py", author_dir / "utils.py"]
    config = {
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "model": args.model,
        "ollama_host": args.ollama_host,
        "evolution_workers": args.evolution_workers,
        "encoder": args.encoder,
        "samples": args.samples,
        "sample_start": args.sample_start,
        "max_turns": args.max_turns,
        "max_questions": args.max_questions,
        "link_k": args.link_k,
        "linked_note_limit": args.linked_note_limit,
        "seed": args.seed,
        "legacy_c5_reference": args.legacy_c5_reference,
        "evolution_enabled": not args.disable_evolution,
        "answer_temperature": args.answer_temperature,
        "ground_single_hop": args.ground_single_hop,
        "temporal_prompt": args.temporal_prompt,
        "qa_k": {str(category): k for category, k in QA_K.items()},
        "model_call": {"timeout_seconds": MODEL_TIMEOUT_SECONDS,
                       "max_generated_tokens": MODEL_MAX_TOKENS, "retries": MODEL_RETRIES},
        "deferred_note_retries": 1,
    }
    checkpoint_dir = args.output.with_suffix(".checkpoints")
    encoder = SentenceTransformer(args.encoder)
    client = Client(host=args.ollama_host, trust_env=False, timeout=MODEL_TIMEOUT_SECONDS)
    results = []
    evolved_notes = 0
    failed_turns = []
    evolution_failures = []
    note_write_failures = []
    resume_history = []
    memory_summary = []
    for sample in samples[sample_start:sample_end]:
        state = run_sample(
            args, sample, client, encoder, config, checkpoint_dir / f"sample-{sample.sample_id}.json", scoring,
        )
        results.extend(state["results"])
        evolved_notes += state["evolved_notes"]
        failed_turns.extend(state["failed_turns"])
        evolution_failures.extend(state["evolution_failures"])
        note_write_failures.extend(state["note_write_failures"])
        resume_history.extend(state.get("resume_history", []))
        memory_summary.append({
            "sample_id": sample.sample_id,
            "processed_turns": state["next_turn"],
            "notes": len(state["notes"]),
        })

    write_json(args.output, {
        "config": config,
        "mode": "build_only" if args.build_only else "evaluation",
        "memory_summary": memory_summary,
        "total_questions": len(results),
        "relaxed_answer_correct_count": sum(row["answer_correct_relaxed"] for row in results),
        "evolution_updates": evolved_notes,
        "failed_turns": failed_turns,
        "evolution_failures": evolution_failures,
        "note_write_failures": note_write_failures,
        "recovered_turn_count": sum(record["retry"]["status"] == "recovered" for record in note_write_failures),
        "failed_question_count": sum(row["model_failure"] is not None for row in results),
        "resume_history": resume_history,
        "questions_with_scoring_warnings": sum(bool(row["scoring_warnings"]) for row in results),
        "aggregate_metrics": scoring.aggregate_metrics(
            [row["metrics"] for row in results], [row["category"] for row in results],
        ),
        "individual_results": results,
    })
    print(f"Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
