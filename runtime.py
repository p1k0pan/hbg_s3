"""Self-contained runtime helpers for the blind Stage 3 annotator."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from PIL import Image, ImageEnhance


PROTOCOL_VERSION = "stage3_multi_v1"
BLIND_TASK_SCHEMA = "stage3_blind_task_v1"
VOTE_SCHEMA = "stage3_visual_vote_v1"
PROMPT_VERSION = "stage3_blind_visual_v1"
INTERNAL_PREFIX = "_hbg_visual_preannotation_"
Image.MAX_IMAGE_PIXELS = None


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield row


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(payload: Any, length: int = 16) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:length]


def stable_id(prefix: str, payload: Any, length: int = 16) -> str:
    return f"{prefix}_{stable_hash(payload, length)}"


def stable_shuffle_indices(n: int, seed_payload: Any) -> list[int]:
    order = list(range(n))
    random.Random(int(stable_hash(seed_payload, 16), 16)).shuffle(order)
    return order


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def read_api_config(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or not lines[0].strip() or not lines[1].strip():
        raise ValueError(f"{path} must contain API key on line 1 and base URL on line 2")
    return lines[0].strip(), lines[1].strip()


def _to_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        return _to_float(value[0], default)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group())
    return default


@dataclass
class AgentConfig:
    hosts: tuple[str, ...]
    host: str
    model: str
    api_key: str = "EMPTY"
    out_dir: str = "stage3_runs"
    max_steps: int = 10
    max_active_crops: int = 6
    max_parallel_tools: int = 3
    max_tokens: int = 4096
    temperature: float = 0.0
    request_timeout: int = 600
    overview_max_side: int = 1536
    crop_max_side: int = 1024
    jpeg_quality: int = 90
    save_crops: bool = False


@dataclass
class _Backend:
    url: str
    client: OpenAI
    model: str
    inflight: int = 0
    errors: int = 0
    cooldown_until: float = 0.0


class ClientPool:
    def __init__(
        self,
        hosts: list[str],
        api_key: str,
        model: str,
        max_retries: int = 1,
        require_tools: bool = True,
    ):
        self._backends: list[_Backend] = []
        self._lock = threading.Lock()
        keys = [value.strip() for value in api_key.split(",")] if "," in api_key else []
        models = [value.strip() for value in model.split(",")] if "," in model else []
        for index, host in enumerate(hosts):
            base_url = host.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            backend_key = keys[index] if index < len(keys) else ("EMPTY" if keys else api_key)
            requested_model = models[index] if index < len(models) else ("auto" if models else model)
            client = OpenAI(base_url=base_url, api_key=backend_key or "EMPTY", max_retries=max_retries)
            if requested_model in {"", "auto"}:
                served = client.models.list().data
                if not served:
                    continue
                requested_model = served[0].id
            if require_tools and not self._probe_tools(client, requested_model):
                continue
            self._backends.append(_Backend(host, client, requested_model))
            print(f"[pool] + {host} (model={requested_model})")
        if not self._backends:
            raise RuntimeError("no usable tool-capable OpenAI-compatible backend")

    @staticmethod
    def _probe_tools(client: OpenAI, model: str) -> bool:
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "no-op",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }],
                tool_choice="auto",
                max_tokens=1,
                timeout=30,
            )
            return True
        except Exception as exc:
            text = str(exc).lower()
            return not ("tool" in text and "choice" in text)

    @property
    def model(self) -> str:
        return self._backends[0].model

    def chat(self, **kwargs):
        now = time.time()
        with self._lock:
            available = [backend for backend in self._backends if backend.cooldown_until <= now]
            backend = min(available or self._backends, key=lambda item: (item.inflight, item.errors))
            backend.inflight += 1
        try:
            kwargs.setdefault("model", backend.model)
            return backend.client.chat.completions.create(**kwargs)
        except Exception:
            backend.errors += 1
            backend.cooldown_until = time.time() + min(30, 2 ** min(backend.errors, 4))
            raise
        finally:
            with self._lock:
                backend.inflight = max(0, backend.inflight - 1)


@dataclass
class LoadedImage:
    img: Image.Image
    width: int
    height: int
    path: str


def load_image(path: str) -> LoadedImage:
    image = Image.open(path).convert("RGB")
    return LoadedImage(image, image.width, image.height, path)


def _downscale(image: Image.Image, max_side: int) -> Image.Image:
    long_side = max(image.width, image.height)
    if long_side <= max_side:
        return image
    scale = max_side / long_side
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)


def make_overview(loaded: LoadedImage, max_side: int = 1536) -> Image.Image:
    return _downscale(loaded.img, max_side)


def to_data_url(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def crop_region(loaded: LoadedImage, x0: float, y0: float, x1: float, y1: float, **kwargs) -> Image.Image:
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if x1 - x0 < 0.02:
        center = (x0 + x1) / 2
        x0, x1 = max(0, center - 0.01), min(1, center + 0.01)
    if y1 - y0 < 0.02:
        center = (y0 + y1) / 2
        y0, y1 = max(0, center - 0.01), min(1, center + 0.01)
    crop = loaded.img.crop((
        int(x0 * loaded.width),
        int(y0 * loaded.height),
        max(1, int(x1 * loaded.width)),
        max(1, int(y1 * loaded.height)),
    ))
    rotate = int(kwargs.get("rotate_deg", 0))
    if rotate % 360:
        crop = crop.rotate(-rotate, expand=True)
    brightness = float(kwargs.get("brightness", 1.0))
    contrast = float(kwargs.get("contrast", 1.0))
    sharpen = float(kwargs.get("sharpen", 1.0))
    if brightness != 1.0:
        crop = ImageEnhance.Brightness(crop).enhance(brightness)
    if contrast != 1.0:
        crop = ImageEnhance.Contrast(crop).enhance(contrast)
    if sharpen != 1.0:
        crop = ImageEnhance.Sharpness(crop).enhance(sharpen)
    return _downscale(crop, int(kwargs.get("max_side", 1024)))


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_COORD_KEYS = ("x0", "y0", "x1", "y1")
_COORD_LABEL_RE = re.compile(r"(?<!\w)(?:x0|y0|x1|y1)(?!\w)", re.I)


def _numbers(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numbers(item)]
    if isinstance(value, str):
        cleaned = _COORD_LABEL_RE.sub(" ", value)
        return [float(match.group()) for match in _NUMBER_RE.finditer(cleaned)]
    return []


def _parse_bbox(args: dict[str, Any]) -> tuple[float, float, float, float, str | None]:
    packed_key = next((key for key in ("region", "bbox") if args.get(key) is not None), None)
    note = None
    if packed_key:
        values = _numbers(args[packed_key])
        if len(values) != 4:
            raise ValueError(f"{packed_key} must contain exactly four coordinates")
        note = f"parsed packed {packed_key}"
    else:
        fields = {key: _numbers(args.get(key)) for key in _COORD_KEYS}
        if all(len(fields[key]) == 1 for key in _COORD_KEYS):
            values = [fields[key][0] for key in _COORD_KEYS]
        else:
            packed = [(key, value) for key, value in fields.items() if len(value) == 4]
            if len(packed) != 1:
                raise ValueError("provide numeric x0,y0,x1,y1")
            key, values = packed[0]
            note = f"repaired packed coordinates from {key}"
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError("crop coordinates must be finite and within [0,1]")
    return values[0], values[1], values[2], values[3], note


def execute_inspect_region(loaded: LoadedImage, args: dict[str, Any], cfg: AgentConfig) -> tuple[str, str]:
    x0, y0, x1, y1, note = _parse_bbox(args)
    crop = crop_region(
        loaded,
        x0,
        y0,
        x1,
        y1,
        brightness=_to_float(args.get("brightness"), 1.0),
        contrast=_to_float(args.get("contrast"), 1.0),
        sharpen=_to_float(args.get("sharpen"), 1.0),
        rotate_deg=int(_to_float(args.get("rotate_deg"), 0.0)),
        max_side=cfg.crop_max_side,
    )
    summary = (
        f"Cropped region [{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}] "
        f"(rendered {crop.width}x{crop.height})"
        f"{f' ({note})' if note else ''}. The crop image follows."
    )
    return summary, to_data_url(crop, cfg.jpeg_quality)


def tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "inspect_region",
                "description": (
                    "Crop a normalized region from the ORIGINAL high-resolution herbarium scan. "
                    "Coordinates are fractions: (0,0)=top-left, (1,1)=bottom-right."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x0": {"type": "number"}, "y0": {"type": "number"},
                        "x1": {"type": "number"}, "y1": {"type": "number"},
                        "reason": {"type": "string"},
                        "brightness": {"type": "number", "default": 1.0},
                        "contrast": {"type": "number", "default": 1.0},
                        "sharpen": {"type": "number", "default": 1.0},
                        "rotate_deg": {"type": "integer", "default": 0},
                    },
                    "required": ["x0", "y0", "x1", "y1", "reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_observation",
                "description": "Record visible findings for one returned crop.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "crop_id": {"type": "string"},
                        "utility": {"type": "string", "enum": ["useful", "partial", "not_useful"]},
                        "organ": {"type": "string"},
                        "findings": {"type": "array", "items": {"type": "string"}},
                        "uncertainty": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["crop_id", "utility", "organ", "findings", "uncertainty"],
                },
            },
        },
    ]


def api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in message.items() if not key.startswith(INTERNAL_PREFIX)} for message in messages]


def find_crop(crops: list[dict[str, Any]], crop_id: str) -> dict[str, Any] | None:
    return next((crop for crop in crops if crop.get("crop_id") == crop_id), None)


def pending_crop_ids(crops: list[dict[str, Any]]) -> list[str]:
    return [crop["crop_id"] for crop in crops if not crop.get("observation")]


def record_observation(crops: list[dict[str, Any]], args: dict[str, Any]) -> dict[str, Any]:
    crop = find_crop(crops, str(args.get("crop_id") or ""))
    if not crop:
        return {"status": "rejected", "message": "unknown crop_id"}
    findings = args.get("findings") or []
    if isinstance(findings, str):
        findings = [findings]
    utility = str(args.get("utility") or "partial")
    if utility not in {"useful", "partial", "not_useful"}:
        utility = "partial"
    crop["observation"] = {
        "utility": utility,
        "organ": str(args.get("organ") or "unknown"),
        "findings": [str(value) for value in findings],
        "uncertainty": str(args.get("uncertainty") or ""),
        "confidence": max(0.0, min(1.0, _to_float(args.get("confidence"), 0.5))),
    }
    return {"status": "accepted", "message": f"observation recorded for {crop['crop_id']}"}


def refresh_context(
    messages: list[dict[str, Any]],
    crops: list[dict[str, Any]],
    active_crop_ids: list[str],
    max_active_crops: int,
) -> None:
    messages[:] = [message for message in messages if message.get(f"{INTERNAL_PREFIX}kind") != "memory"]
    while len(active_crop_ids) > max_active_crops:
        crop = find_crop(crops, active_crop_ids[0])
        if not crop or not crop.get("observation"):
            break
        active_crop_ids.pop(0)
    active = set(active_crop_ids)
    messages[:] = [
        message for message in messages
        if not (
            message.get(f"{INTERNAL_PREFIX}kind") == "crop_image"
            and message.get(f"{INTERNAL_PREFIX}crop_id") not in active
        )
    ]
    if crops:
        lines = ["Compact crop observation memory:"]
        for crop in crops:
            observation = crop.get("observation")
            if observation:
                lines.append(
                    f"- {crop['crop_id']} organ={observation['organ']} utility={observation['utility']}: "
                    f"{'; '.join(observation['findings']) or 'no findings'}; "
                    f"uncertainty={observation['uncertainty'] or 'none'}"
                )
        lines.append(f"Pending observations: {', '.join(pending_crop_ids(crops)) or 'none'}")
        messages.append({
            "role": "user",
            "content": "\n".join(lines),
            f"{INTERNAL_PREFIX}kind": "memory",
        })


def save_data_url(data_url: str, path: Path) -> None:
    path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))


def _normalized_bbox(args: dict[str, Any]) -> list[float]:
    try:
        x0, y0, x1, y1, _ = _parse_bbox(args)
    except ValueError:
        return [0.0, 0.0, 1.0, 1.0]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def evidence_regions(crops: list[dict[str, Any]], crop_ids: list[str]) -> list[dict[str, Any]]:
    regions = []
    for crop_id in crop_ids:
        crop = find_crop(crops, crop_id)
        if crop:
            regions.append({
                "crop_id": crop_id,
                "bbox": _normalized_bbox(crop.get("args") or {}),
                "reason": crop.get("reason", ""),
                "observation": crop.get("observation"),
            })
    return regions

