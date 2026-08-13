#!/usr/bin/env python3
"""Run one independent blind LVLM annotator over portable Stage 3 tasks."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from runtime import (
    AgentConfig,
    BLIND_TASK_SCHEMA,
    ClientPool,
    INTERNAL_PREFIX,
    PROMPT_VERSION,
    PROTOCOL_VERSION,
    VOTE_SCHEMA,
    _to_float,
    append_jsonl,
    api_messages,
    evidence_regions,
    execute_inspect_region,
    find_crop,
    load_image,
    make_overview,
    is_gpt5_model,
    normalized_text,
    pending_crop_ids,
    read_api_config,
    read_jsonl,
    record_observation,
    refresh_context,
    save_data_url,
    stable_id,
    stable_shuffle_indices,
    to_data_url,
    tool_specs,
)


ANNOTATOR_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def pbar(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm is not None else iterable


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def submit_votes_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "submit_visual_votes",
            "description": "Submit one independent image-grounded vote for every question in this batch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question_id": {"type": "string"},
                                "selected_option": {
                                    "type": "integer",
                                    "description": "Displayed 0-based option index.",
                                },
                                "organ_visibility": {
                                    "type": "string",
                                    "enum": ["visible", "partially_visible", "not_visible"],
                                },
                                "trait_answerability": {
                                    "type": "string",
                                    "enum": ["answerable", "ambiguous", "not_observable"],
                                },
                                "confidence": {"type": "number"},
                                "observed_state": {"type": "string"},
                                "evidence": {"type": "string"},
                                "evidence_crop_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "question_id",
                                "selected_option",
                                "organ_visibility",
                                "trait_answerability",
                                "confidence",
                                "observed_state",
                                "evidence",
                                "evidence_crop_ids",
                            ],
                        },
                    }
                },
                "required": ["answers"],
            },
        },
    }


def annotation_tools() -> list[dict[str, Any]]:
    keep = {"inspect_region", "record_observation"}
    tools = [tool for tool in tool_specs() if tool["function"]["name"] in keep]
    tools.append(submit_votes_tool())
    return tools


def resolve_language(task: dict[str, Any], requested: str, kind: str) -> str:
    if requested != "original":
        return requested
    default = task.get(f"{kind}_language_default") or task.get("original_language") or "en"
    return "de" if default == "de" else "en"


def display_task(
    task: dict[str, Any],
    annotator_id: str,
    shuffle_seed: str,
    question_language: str,
    option_language: str,
) -> dict[str, Any]:
    q_language = resolve_language(task, question_language, "question")
    o_language = resolve_language(task, option_language, "option")
    question = task.get(f"question_{q_language}") or task.get("question_en") or task.get("question_de")
    if not question:
        raise ValueError(f"{task.get('candidate_id')}: missing question for language {q_language}")
    options = task.get("options") or []
    order = stable_shuffle_indices(
        len(options),
        {
            "protocol": PROTOCOL_VERSION,
            "annotator_id": annotator_id,
            "shuffle_seed": shuffle_seed,
            "candidate_id": task.get("candidate_id"),
        },
    )
    displayed: list[dict[str, Any]] = []
    for source_index in order:
        option = options[source_index]
        text = option.get(f"text_{o_language}") or option.get("text_en") or option.get("text_de")
        if not text:
            raise ValueError(
                f"{task.get('candidate_id')}: missing option text for language {o_language}"
            )
        displayed.append(
            {
                "option_id": str(option["option_id"]),
                "text": str(text),
                "source_index": source_index,
            }
        )
    cannot_id = str(task.get("cannot_determine_option_id") or "")
    cannot_display_index = next(
        (index for index, option in enumerate(displayed) if option["option_id"] == cannot_id),
        -1,
    )
    if cannot_display_index < 0:
        raise ValueError(f"{task.get('candidate_id')}: Cannot determine option missing")
    return {
        "task": task,
        "candidate_id": str(task["candidate_id"]),
        "question_id": str(task.get("question_id") or task["candidate_id"]),
        "question": str(question),
        "question_language": q_language,
        "option_language": o_language,
        "displayed_options": displayed,
        "cannot_determine_option_id": cannot_id,
        "cannot_determine_display_index": cannot_display_index,
    }


def build_batches(
    tasks: list[dict[str, Any]],
    annotator_id: str,
    shuffle_seed: str,
    question_language: str,
    option_language: str,
    max_questions: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        display = display_task(
            task,
            annotator_id,
            shuffle_seed,
            question_language,
            option_language,
        )
        organ = normalized_text(task.get("suborgan") or task.get("organ") or "unknown")
        grouped[(str(task["sample_id"]), organ, display["question_language"])].append(display)

    batches: list[dict[str, Any]] = []
    for (sample_id, organ, language), questions in grouped.items():
        questions.sort(key=lambda row: row["candidate_id"])
        for start in range(0, len(questions), max_questions):
            chunk = questions[start : start + max_questions]
            batch_id = make_batch_id(sample_id, organ, chunk)
            batches.append(
                {
                    "batch_id": batch_id,
                    "sample_id": sample_id,
                    "organ": organ,
                    "language": language,
                    "questions": chunk,
                }
            )
    batches.sort(key=lambda row: (row["sample_id"], row["organ"], row["batch_id"]))
    return batches


def make_batch_id(sample_id: str, organ: str, questions: list[dict[str, Any]]) -> str:
    return stable_id(
        "blindvisualbatch",
        {
            "protocol": PROTOCOL_VERSION,
            "sample_id": sample_id,
            "organ": organ,
            "candidate_ids": [row["candidate_id"] for row in questions],
        },
    )


def system_prompt() -> str:
    return (
        "You are an independent visual annotator of morphological traits in herbarium specimens. "
        "Judge every question from visible image evidence only. Use inspect_region to examine relevant "
        "organs at the original scan resolution, then call record_observation for every returned crop. "
        "For an answerable state, cite at least one useful or partial evidence crop. Set organ_visibility "
        "independently from trait_answerability: a visible organ can still have an unobservable character. "
        "Use ambiguous when the organ is present but the state cannot be distinguished. Use not_observable "
        "when the required organ, preservation, or resolution is absent. In either case choose the displayed "
        "Cannot determine option. Do not infer from taxonomy, labels, geography, or likely species identity. "
        "Finish by calling submit_visual_votes exactly once with every question_id."
    )


def user_prompt(batch: dict[str, Any]) -> str:
    lines = [
        f"Batch ID: {batch['batch_id']}",
        f"Shared organ focus: {batch['organ']}",
        "",
        "Independently annotate every question below. Displayed option indices are 0-based.",
    ]
    for item in batch["questions"]:
        lines.extend(["", f"Question ID: {item['question_id']}", item["question"], "Options:"])
        lines.extend(
            f"{index}. {option['text']}"
            for index, option in enumerate(item["displayed_options"])
        )
    lines.extend(
        [
            "",
            "Do not identify the specimen or infer an answer from its label.",
            "Inspect the original-resolution image where needed and submit one vote per Question ID.",
        ]
    )
    return "\n".join(lines)


def resolve_image_path(task: dict[str, Any], image_root: Path) -> Path:
    candidates = [
        image_root / str(task.get("image_relpath") or ""),
        image_root / str(task.get("image_filename") or ""),
        image_root / str(task.get("region") or "") / str(task.get("image_filename") or ""),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"image for {task.get('candidate_id')} not found; tried: {tried}")


def submission_errors(
    batch: dict[str, Any],
    answers: list[dict[str, Any]],
    crops: list[dict[str, Any]],
) -> list[str]:
    expected = {question["question_id"]: question for question in batch["questions"]}
    submitted = {
        str(answer.get("question_id")): answer
        for answer in answers
        if isinstance(answer, dict) and answer.get("question_id")
    }
    errors: list[str] = []
    missing = sorted(set(expected) - set(submitted))
    extra = sorted(set(submitted) - set(expected))
    if missing:
        errors.append(f"missing question_ids: {', '.join(missing)}")
    if extra:
        errors.append(f"unknown question_ids: {', '.join(extra)}")
    for question_id, answer in submitted.items():
        display = expected.get(question_id)
        if not display:
            continue
        selected = int(_to_float(answer.get("selected_option"), -1))
        answerability = str(answer.get("trait_answerability") or "")
        visibility = str(answer.get("organ_visibility") or "")
        cannot_index = display["cannot_determine_display_index"]
        if not 0 <= selected < len(display["displayed_options"]):
            errors.append(f"{question_id}: selected_option out of range")
            continue
        if visibility == "not_visible" and answerability != "not_observable":
            errors.append(f"{question_id}: not_visible requires not_observable")
        if answerability == "answerable":
            if selected == cannot_index:
                errors.append(f"{question_id}: answerable cannot select Cannot determine")
            crop_ids = [str(value) for value in (answer.get("evidence_crop_ids") or [])]
            usable = [
                find_crop(crops, crop_id)
                for crop_id in crop_ids
            ]
            usable = [
                crop for crop in usable
                if crop and (crop.get("observation") or {}).get("utility") in {"useful", "partial"}
            ]
            if not usable:
                errors.append(f"{question_id}: answerable state requires a recorded evidence crop")
        elif answerability in {"ambiguous", "not_observable"}:
            if selected != cannot_index:
                errors.append(f"{question_id}: unanswerable vote must select Cannot determine")
        else:
            errors.append(f"{question_id}: invalid trait_answerability")
    return errors


def derive_vote(
    display: dict[str, Any],
    raw_answer: dict[str, Any] | None,
    crops: list[dict[str, Any]],
    annotator_id: str,
    model: str,
    batch_id: str,
) -> dict[str, Any]:
    task = display["task"]
    base = {
        "schema_version": VOTE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "candidate_id": display["candidate_id"],
        "question_id": display["question_id"],
        "task_fingerprint": task["task_fingerprint"],
        "region": task["region"],
        "sample_id": task["sample_id"],
        "batch_id": batch_id,
        "annotator_id": annotator_id,
        "model": model,
        "visual_ground_truth_validated": False,
        "label_provenance": "independent_lvlm_vote",
    }
    if raw_answer is None:
        return {
            **base,
            "vote_valid": False,
            "quality_flags": ["missing_model_answer"],
            "organ_visibility": None,
            "trait_answerability": None,
            "selected_option_id": None,
            "selected_display_index": None,
            "selected_text": None,
            "observed_state": None,
            "confidence": 0.0,
            "evidence": "",
            "evidence_regions": [],
            "displayed_option_ids": [option["option_id"] for option in display["displayed_options"]],
            "raw_answer": None,
        }

    quality_flags: list[str] = []
    selected = int(_to_float(raw_answer.get("selected_option"), -1))
    if not 0 <= selected < len(display["displayed_options"]):
        quality_flags.append("selected_option_out_of_range")
        selected = display["cannot_determine_display_index"]
    chosen = display["displayed_options"][selected]
    visibility = str(raw_answer.get("organ_visibility") or "")
    if visibility not in {"visible", "partially_visible", "not_visible"}:
        quality_flags.append("invalid_organ_visibility")
        visibility = None
    answerability = str(raw_answer.get("trait_answerability") or "")
    if answerability not in {"answerable", "ambiguous", "not_observable"}:
        quality_flags.append("invalid_trait_answerability")
        answerability = None
    cannot_selected = chosen["option_id"] == display["cannot_determine_option_id"]
    if visibility == "not_visible" and answerability != "not_observable":
        quality_flags.append("visibility_answerability_conflict")
    if answerability == "answerable" and cannot_selected:
        quality_flags.append("answerable_selected_cannot_determine")
    if answerability in {"ambiguous", "not_observable"} and not cannot_selected:
        quality_flags.append("unanswerable_selected_state")

    crop_ids = [str(value) for value in (raw_answer.get("evidence_crop_ids") or [])]
    regions = evidence_regions(crops, crop_ids)
    usable = [
        region for region in regions
        if (region.get("observation") or {}).get("utility") in {"useful", "partial"}
    ]
    if answerability == "answerable" and not usable:
        quality_flags.append("answerable_without_localized_evidence")
    confidence = max(0.0, min(1.0, _to_float(raw_answer.get("confidence"), 0.0)))
    return {
        **base,
        "vote_valid": not quality_flags,
        "quality_flags": quality_flags,
        "organ_visibility": visibility,
        "trait_answerability": answerability,
        "selected_option_id": chosen["option_id"],
        "selected_display_index": selected,
        "selected_text": chosen["text"],
        "cannot_determine_selected": cannot_selected,
        "observed_state": str(raw_answer.get("observed_state") or ""),
        "confidence": confidence,
        "evidence": str(raw_answer.get("evidence") or ""),
        "evidence_regions": regions,
        "displayed_option_ids": [option["option_id"] for option in display["displayed_options"]],
        "raw_answer": raw_answer,
    }


def load_resume_crops(
    resume_root: Path,
    batch: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path]:
    report_path = resume_root / "batches" / batch["batch_id"] / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"resume report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("batch_id") != batch["batch_id"]:
        raise ValueError(f"resume report batch mismatch: {report_path}")
    if str(report.get("sample_id")) != str(batch["sample_id"]):
        raise ValueError(f"resume report sample mismatch: {report_path}")

    crops: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in report.get("crops") or []:
        if not isinstance(raw, dict):
            continue
        crop_id = str(raw.get("crop_id") or "")
        args = raw.get("args")
        if not crop_id or crop_id in seen or not isinstance(args, dict):
            continue
        observation = raw.get("observation")
        crops.append(
            {
                "crop_id": crop_id,
                "args": dict(args),
                "reason": str(raw.get("reason") or args.get("reason") or ""),
                "saved_path": raw.get("saved_path"),
                "observation": dict(observation) if isinstance(observation, dict) else None,
                "recovered_from": str(report_path),
            }
        )
        seen.add(crop_id)
    return crops, report_path


def crop_sequence(crops: list[dict[str, Any]]) -> int:
    values = []
    for crop in crops:
        match = re.fullmatch(r"crop_(\d+)", str(crop.get("crop_id") or ""))
        if match:
            values.append(int(match.group(1)))
    return max(values, default=0)


class BlindVisualRunner:
    def __init__(
        self,
        cfg: AgentConfig,
        annotator_id: str,
        image_root: Path,
        enable_thinking: bool,
        thinking_budget: int | None,
        reasoning_effort: str | None,
        completion_token_limit: int | None,
        resume_from_out_dir: Path | None,
    ):
        self.cfg = cfg
        self.annotator_id = annotator_id
        self.image_root = image_root
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.reasoning_effort = reasoning_effort
        self.completion_token_limit = completion_token_limit
        self.resume_from_out_dir = resume_from_out_dir
        hosts = list(dict.fromkeys([*cfg.hosts, cfg.host]))
        self.pool = ClientPool(
            hosts,
            api_key=cfg.api_key,
            model=cfg.model,
            max_retries=1,
            require_tools=True,
        )
        self.model = self.pool.model
        self.tools = annotation_tools()

    def run_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        cfg = self.cfg
        paths = {
            resolve_image_path(question["task"], self.image_root)
            for question in batch["questions"]
        }
        if len(paths) != 1:
            raise ValueError(f"batch {batch['batch_id']} resolved to {len(paths)} image paths")
        image_path = next(iter(paths))
        loaded = load_image(str(image_path))
        overview = make_overview(loaded, cfg.overview_max_side)
        out_root = Path(cfg.out_dir) / "batches" / batch["batch_id"]
        crop_dir = out_root / "crops"
        if cfg.save_crops:
            crop_dir.mkdir(parents=True, exist_ok=True)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt(batch)},
                    {"type": "image_url", "image_url": {"url": to_data_url(overview, cfg.jpeg_quality)}},
                ],
            },
        ]
        crops: list[dict[str, Any]] = []
        active_crop_ids: list[str] = []
        trace: list[dict[str, Any]] = []
        submitted: dict[str, dict[str, Any]] | None = None
        n_inspections = 0
        resume_report_path: Path | None = None
        if self.resume_from_out_dir is not None:
            crops, resume_report_path = load_resume_crops(
                self.resume_from_out_dir,
                batch,
            )
            n_inspections = crop_sequence(crops)
            active_crop_ids = [crop["crop_id"] for crop in crops[-cfg.max_active_crops :]]
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Resume this unfinished annotation. Recovered {len(crops)} prior crops and "
                        f"{sum(crop.get('observation') is not None for crop in crops)} recorded observations. "
                        "Use the recovered evidence memory below, continue inspecting only where needed, "
                        "and finish by submitting the vote."
                    ),
                }
            )
            for crop in crops[-cfg.max_active_crops :]:
                try:
                    _, data_url = execute_inspect_region(loaded, crop["args"], cfg)
                except Exception:
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Recovered {crop['crop_id']}: {crop['reason']}",
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                        f"{INTERNAL_PREFIX}kind": "crop_image",
                        f"{INTERNAL_PREFIX}crop_id": crop["crop_id"],
                    }
                )
        started = time.time()

        total_steps = max(1, cfg.max_steps - 1) + cfg.max_submit_attempts
        for step_index in range(total_steps):
            refresh_context(messages, crops, active_crop_ids, cfg.max_active_crops)
            force_submit = step_index >= max(0, cfg.max_steps - 1)
            final_attempt = step_index == total_steps - 1
            tool_choice: Any = (
                {"type": "function", "function": {"name": "submit_visual_votes"}}
                if force_submit else "auto"
            )
            available_tools = [submit_votes_tool()] if force_submit else self.tools
            kwargs: dict[str, Any] = {
                "messages": api_messages(messages),
                "tools": available_tools,
                "tool_choice": tool_choice,
                "timeout": cfg.request_timeout,
            }
            if is_gpt5_model(self.model):
                if self.completion_token_limit is not None:
                    kwargs["max_completion_tokens"] = self.completion_token_limit
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = cfg.temperature
                kwargs["max_tokens"] = cfg.max_tokens
            if self.enable_thinking and not is_gpt5_model(self.model):
                kwargs["extra_body"] = {"enable_thinking": True}
                if self.thinking_budget is not None:
                    kwargs["extra_body"]["thinking_budget"] = self.thinking_budget
            try:
                response = self.pool.chat(**kwargs)
            except Exception as exc:
                trace.append({"step": step_index, "error": f"{type(exc).__name__}: {exc}"})
                break
            message = response.choices[0].message
            tool_calls = (message.tool_calls or [])[: cfg.max_parallel_tools]
            assistant: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in tool_calls
                ]
            messages.append(assistant)
            step_trace: dict[str, Any] = {
                "step": step_index,
                "assistant_content": message.content or "",
                "tool_calls": [],
                "tool_results": [],
            }
            trace.append(step_trace)
            if not tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You must now call submit_visual_votes. Do not inspect or record more evidence."
                            if force_submit
                            else "Use inspect_region if needed, then finish with submit_visual_votes."
                        ),
                    }
                )
                continue

            finished = False
            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                step_trace["tool_calls"].append({"name": name, "args": args})

                if force_submit and name != "submit_visual_votes":
                    text = "Final submission phase: call submit_visual_votes now; no other tools are available."
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
                    step_trace["tool_results"].append(
                        {"name": name, "status": "rejected_final_phase", "error": text}
                    )
                    continue

                if name == "inspect_region":
                    try:
                        summary, data_url = execute_inspect_region(loaded, args, cfg)
                    except Exception as exc:
                        text = f"inspect_region rejected: {type(exc).__name__}: {exc}"
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
                        step_trace["tool_results"].append({"name": name, "status": "rejected", "error": text})
                        continue
                    n_inspections += 1
                    crop_id = f"crop_{n_inspections:03d}"
                    crop_path = crop_dir / f"{crop_id}.jpg" if cfg.save_crops else None
                    if crop_path:
                        save_data_url(data_url, crop_path)
                    crop = {
                        "crop_id": crop_id,
                        "args": args,
                        "reason": str(args.get("reason") or ""),
                        "saved_path": str(crop_path) if crop_path else None,
                        "observation": None,
                    }
                    crops.append(crop)
                    active_crop_ids.append(crop_id)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": f"{crop_id}: {summary} Record an observation for this crop next.",
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{crop_id}: {args.get('reason', '')}"},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                            f"{INTERNAL_PREFIX}kind": "crop_image",
                            f"{INTERNAL_PREFIX}crop_id": crop_id,
                        }
                    )
                    step_trace["tool_results"].append(
                        {"name": name, "status": "crop_returned", "crop_id": crop_id}
                    )
                    continue

                if name == "record_observation":
                    result = record_observation(crops, args)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result["message"]})
                    step_trace["tool_results"].append({"name": name, **result})
                    continue

                if name == "submit_visual_votes":
                    pending = pending_crop_ids(crops)
                    answers = args.get("answers") or []
                    errors = submission_errors(batch, answers, crops)
                    if pending:
                        errors.insert(0, f"record observations for pending crops: {', '.join(pending)}")
                    if errors and not final_attempt:
                        text = "Submission rejected:\n- " + "\n- ".join(errors)
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
                        step_trace["tool_results"].append(
                            {"name": name, "status": "rejected", "errors": errors}
                        )
                        continue
                    submitted = {
                        str(answer.get("question_id")): answer
                        for answer in answers
                        if isinstance(answer, dict) and answer.get("question_id")
                    }
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": "Votes accepted."})
                    step_trace["tool_results"].append(
                        {
                            "name": name,
                            "status": "accepted_forced" if errors else "accepted",
                            "errors": errors,
                            "n_answers": len(submitted),
                        }
                    )
                    finished = True
                    break

                messages.append({"role": "tool", "tool_call_id": call.id, "content": f"unknown tool {name}"})
            if finished:
                break

        submitted = submitted or {}
        votes = [
            derive_vote(
                display,
                submitted.get(display["question_id"]),
                crops,
                self.annotator_id,
                self.model,
                batch["batch_id"],
            )
            for display in batch["questions"]
        ]
        report = {
            "schema_version": "stage3_blind_batch_report_v1",
            "protocol_version": PROTOCOL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "batch_id": batch["batch_id"],
            "sample_id": batch["sample_id"],
            "organ": batch["organ"],
            "image_path": str(image_path),
            "annotator_id": self.annotator_id,
            "model": self.model,
            "n_questions": len(batch["questions"]),
            "n_answers": len(submitted),
            "n_inspections": n_inspections,
            "resumed_from_report": str(resume_report_path) if resume_report_path else None,
            "n_recovered_crops": sum("recovered_from" in crop for crop in crops),
            "n_recovered_observations": sum(
                "recovered_from" in crop and crop.get("observation") is not None
                for crop in crops
            ),
            "elapsed_s": round(time.time() - started, 1),
            "votes": votes,
            "crops": crops,
            "trace": trace,
        }
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "report.json").write_text(
            json.dumps({key: value for key, value in report.items() if key != "trace"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_root / "trajectory.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report


def load_done(path: Path, annotator_id: str) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row.get("candidate_id"))
        for row in read_jsonl(path)
        if row.get("schema_version") == VOTE_SCHEMA and row.get("annotator_id") == annotator_id
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--host", default="http://localhost:8009")
    parser.add_argument("--hosts", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-file", type=Path, help="file with the API key on its first line")
    parser.add_argument("--api-config-file", type=Path)
    parser.add_argument("--question-language", choices=["original", "en", "de"], default="en")
    parser.add_argument("--option-language", choices=["original", "en", "de"], default="en")
    parser.add_argument("--shuffle-seed", default="independent-v1")
    parser.add_argument("--max-questions-per-batch", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="limit batches after sharding")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--max-submit-attempts",
        type=int,
        default=3,
        help="submit-only attempts after the inspection phase",
    )
    parser.add_argument("--max-active-crops", type=int, default=6)
    parser.add_argument("--max-parallel-tools", type=int, default=3)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "optional completion-token limit; GPT-5 uses the service default when "
            "omitted, while local OpenAI-compatible models retain the 4096 default"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help="GPT-5 reasoning effort; omit to use the model default",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="optional thinking-token limit; omit to use the service default",
    )
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument(
        "--resume-from-out-dir",
        type=Path,
        help="reuse crop observations from matching batch reports in an earlier output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not ANNOTATOR_RE.fullmatch(args.annotator_id):
        raise SystemExit("--annotator-id may contain only letters, numbers, dot, underscore, and hyphen")
    if args.max_questions_per_batch < 1:
        raise SystemExit("--max-questions-per-batch must be >= 1")
    if args.max_submit_attempts < 1:
        raise SystemExit("--max-submit-attempts must be >= 1")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("require --shard-count >= 1 and 0 <= --shard-index < --shard-count")

    tasks = list(read_jsonl(args.tasks))
    invalid = [task for task in tasks if task.get("schema_version") != BLIND_TASK_SCHEMA]
    if invalid:
        raise SystemExit(f"{len(invalid)} rows are not {BLIND_TASK_SCHEMA}")
    if len({task.get("candidate_id") for task in tasks}) != len(tasks):
        raise SystemExit("duplicate candidate_id in blind tasks")
    all_batches = build_batches(
        tasks,
        args.annotator_id,
        args.shuffle_seed,
        args.question_language,
        args.option_language,
        args.max_questions_per_batch,
    )
    batches = [
        batch for index, batch in enumerate(all_batches)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit:
        batches = batches[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    votes_path = args.out_dir / "votes.jsonl"
    invalid_path = args.out_dir / "invalid_votes.jsonl"
    if votes_path.exists() and not args.skip_done and not args.dry_run:
        raise SystemExit(f"output already exists: {votes_path}; use --skip-done or a new --out-dir")
    if args.skip_done:
        done = load_done(votes_path, args.annotator_id)
        pending_batches: list[dict[str, Any]] = []
        for batch in batches:
            questions = [
                question for question in batch["questions"]
                if question["candidate_id"] not in done
            ]
            if not questions:
                continue
            pending_batches.append(
                {
                    **batch,
                    "batch_id": make_batch_id(batch["sample_id"], batch["organ"], questions),
                    "questions": questions,
                }
            )
        batches = pending_batches

    n_candidates = sum(len(batch["questions"]) for batch in batches)
    print(
        f"[info] annotator={args.annotator_id} candidates={n_candidates} "
        f"batches={len(batches)} shard={args.shard_index}/{args.shard_count}"
    )
    if args.dry_run:
        return

    file_key = ""
    file_host = ""
    if args.api_key_file and args.api_config_file:
        raise SystemExit("use only one of --api-key-file and --api-config-file")
    if args.api_config_file:
        file_key, file_host = read_api_config(args.api_config_file)
    elif args.api_key_file:
        key_lines = args.api_key_file.read_text(encoding="utf-8").splitlines()
        file_key = key_lines[0].strip() if key_lines else ""
        if not file_key:
            raise SystemExit(f"empty API key file: {args.api_key_file}")
    host_values = tuple(value.strip() for value in args.hosts.split(",") if value.strip())
    if not host_values:
        host = file_host or args.host
        if host.rstrip("/").endswith("/v1"):
            host = host.rstrip("/")[:-3]
        host_values = (host,)
    cfg = AgentConfig(hosts=host_values, host=host_values[0], model=args.model)
    cfg.api_key = args.api_key or file_key or cfg.api_key
    cfg.out_dir = str(args.out_dir)
    cfg.max_steps = args.max_steps
    cfg.max_submit_attempts = args.max_submit_attempts
    cfg.max_active_crops = args.max_active_crops
    cfg.max_parallel_tools = args.max_parallel_tools
    cfg.max_tokens = args.max_tokens if args.max_tokens is not None else 4096
    cfg.temperature = args.temperature
    cfg.save_crops = args.save_crops

    if args.enable_thinking and args.model.casefold().startswith("gpt-5"):
        raise SystemExit(
            "--enable-thinking/--thinking-budget are vLLM extensions; "
            "use --reasoning-effort with GPT-5 models"
        )

    manifest = {
        "schema_version": "stage3_annotator_run_manifest_v1",
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotator_id": args.annotator_id,
        "requested_model": args.model,
        "hosts": list(host_values),
        "tasks_file": str(args.tasks),
        "tasks_sha256": file_sha256(args.tasks),
        "image_root": str(args.image_root),
        "question_language": args.question_language,
        "option_language": args.option_language,
        "shuffle_seed": args.shuffle_seed,
        "max_questions_per_batch": args.max_questions_per_batch,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "max_steps": args.max_steps,
        "max_submit_attempts": args.max_submit_attempts,
        "max_active_crops": args.max_active_crops,
        "max_parallel_tools": args.max_parallel_tools,
        "completion_token_limit": args.max_tokens,
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
        "thinking_budget": args.thinking_budget if args.enable_thinking else None,
        "reasoning_effort": args.reasoning_effort,
        "resume_from_out_dir": str(args.resume_from_out_dir) if args.resume_from_out_dir else None,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    runner = BlindVisualRunner(
        cfg,
        args.annotator_id,
        args.image_root,
        args.enable_thinking,
        args.thinking_budget,
        args.reasoning_effort,
        args.max_tokens,
        args.resume_from_out_dir,
    )
    manifest["resolved_model"] = runner.model
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lock = threading.Lock()
    n_votes = n_invalid = 0

    def work(batch: dict[str, Any]) -> dict[str, Any]:
        try:
            return runner.run_batch(batch)
        except Exception as exc:
            return {
                "schema_version": "stage3_blind_batch_report_v1",
                "batch_id": batch["batch_id"],
                "sample_id": batch["sample_id"],
                "error": f"{type(exc).__name__}: {exc}",
                "votes": [
                    derive_vote(
                        question,
                        None,
                        [],
                        args.annotator_id,
                        runner.model,
                        batch["batch_id"],
                    )
                    for question in batch["questions"]
                ],
            }

    def handle(report: dict[str, Any]) -> None:
        nonlocal n_votes, n_invalid
        for vote in report.get("votes") or []:
            append_jsonl(votes_path, vote, lock)
            n_votes += 1
            if not vote.get("vote_valid"):
                append_jsonl(invalid_path, vote, lock)
                n_invalid += 1

    if args.concurrency > 1:
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for report in pbar(executor.map(work, batches), total=len(batches), desc="blind visual batches"):
                handle(report)
    else:
        for batch in pbar(batches, desc="blind visual batches"):
            handle(work(batch))
    print(f"[done] votes={n_votes} invalid={n_invalid} -> {args.out_dir}")


if __name__ == "__main__":
    main()
