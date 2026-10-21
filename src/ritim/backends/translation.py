"""Translation backends: Claude via the Anthropic API (structured JSON output), or a mock."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import Config, require_env
from ..glossary import Glossary
from ..timing import text_length

log = logging.getLogger(__name__)

STYLE = {
    "en": "International English, clear and direct; address learners as 'you'.",
    "de": "Hochdeutsch, Sie-Form, sachlicher Schulungston, Fachbegriffe wie in der VDA-Literatur.",
    "fr": "Français standard, vouvoiement, ton de formation professionnelle.",
    "es": "Español neutro (válido para España y Latinoamérica), trato de usted.",
    "ja": "です・ます調の研修講師の話し言葉。カタカナ語は品質管理分野の慣用に従う。",
    "zh": "简体中文，培训讲师口语风格，术语遵循汽车行业质量管理惯用译法。",
}

LINES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}


@dataclass
class CourseContext:
    title: str
    description: str = ""


@dataclass
class LineIn:
    id: str
    text: str
    max_chars: int
    current: str = ""  # previous attempt when asking for a shorter/corrected version
    problem: str = ""


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


class Translator(Protocol):
    name: str
    usage: Usage

    def translate(self, lang: str, lines: list[LineIn], context: list[tuple[str, str]],
                  course: CourseContext) -> dict[str, str]: ...

    def revise(self, lang: str, lines: list[LineIn], course: CourseContext) -> dict[str, str]: ...


def length_rule(lang: str) -> str:
    if lang in ("ja", "zh"):
        return "max_chars counts every non-space character."
    return "max_chars counts every character including spaces."


class AnthropicTranslator:
    name = "anthropic"

    def __init__(self, cfg: Config, glossary: Glossary, client: Any | None = None) -> None:
        self.cfg = cfg
        self.glossary = glossary
        self.usage = Usage()
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"),
                                         max_retries=cfg.translate.http_retries)
        self.client = client

    # -- prompt --------------------------------------------------------------
    def _system(self, lang: str, course: CourseContext) -> list[dict[str, Any]]:
        lang_name = self.cfg.language(lang).name
        instructions = f"""You translate spoken Turkish lines from Ritim Online quality-management training videos \
(automotive and aerospace quality: VDA, IATF 16949, ISO, CQI) into {lang_name} ({lang}).
The trainer, Hakan Borazan, speaks to learners. Your lines become both subtitles and the dubbed voice in his \
cloned voice, so they must sound natural when spoken aloud and fit the time the original took.

Rules:
- Translate every input line and return each id exactly once, in order, with no notes or comments.
- Keep each line within its max_chars. {length_rule(lang)} Condense by dropping Turkish fillers \
("yani", "işte", "şimdi", "evet", repeated words) and redundancy; never drop technical content, \
numbers, requirement clause references or negations.
- Lines are consecutive fragments of continuous speech; a line may end mid-sentence. Translate the fragment \
so that consecutive lines read naturally together; do not move content between lines.
- Silently correct obvious speech-recognition errors when the context makes the intended word clear.
- Keep the speaker's first-person teaching voice. Style: {STYLE.get(lang, "neutral, professional")}

{self.glossary.prompt_block(lang)}"""
        blocks: list[dict[str, Any]] = [{"type": "text", "text": instructions}]
        course_text = f"Course: {course.title}"
        if course.description:
            course_text += f"\nAbout the course: {course.description}"
        blocks.append({"type": "text", "text": course_text, "cache_control": {"type": "ephemeral"}})
        return blocks

    def _call(self, lang: str, course: CourseContext, user: dict[str, Any]) -> dict[str, str]:
        t = self.cfg.translate
        kwargs: dict[str, Any] = {
            "model": t.model,
            "max_tokens": t.max_tokens,
            "system": self._system(lang, course),
            "messages": [{"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            "output_config": {"effort": t.effort, "format": {"type": "json_schema", "schema": LINES_SCHEMA}},
        }
        if t.fallbacks == "default":
            response = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            response = self.client.messages.create(**kwargs)
        self._account(response)
        if response.stop_reason == "refusal":
            raise RuntimeError(f"translation refused: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("translation hit max_tokens; lower chapters.max_s or raise translate.max_tokens")
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        return {str(x["id"]): str(x["text"]).strip() for x in data.get("lines", [])}

    def _account(self, response: Any) -> None:
        u = response.usage
        self.usage.requests += 1
        self.usage.input_tokens += int(getattr(u, "input_tokens", 0) or 0)
        self.usage.output_tokens += int(getattr(u, "output_tokens", 0) or 0)
        self.usage.cache_read_input_tokens += int(getattr(u, "cache_read_input_tokens", 0) or 0)
        self.usage.cache_creation_input_tokens += int(getattr(u, "cache_creation_input_tokens", 0) or 0)

    # -- API -----------------------------------------------------------------
    def translate(self, lang: str, lines: list[LineIn], context: list[tuple[str, str]],
                  course: CourseContext) -> dict[str, str]:
        user = {
            "task": "translate",
            "previous_lines_for_context": [{"source": s, "translation": d} for s, d in context],
            "lines": [{"id": x.id, "text": x.text, "max_chars": x.max_chars} for x in lines],
        }
        return self._call(lang, course, user)

    def revise(self, lang: str, lines: list[LineIn], course: CourseContext) -> dict[str, str]:
        user = {
            "task": "revise: rewrite each translation so it fixes the stated problem and fits max_chars, "
                    "keeping the meaning of the Turkish source",
            "lines": [{"id": x.id, "source": x.text, "translation": x.current, "problem": x.problem,
                       "max_chars": x.max_chars} for x in lines],
        }
        return self._call(lang, course, user)


@dataclass
class MockTranslator:
    """Deterministic stand-in: '<lang>: <terms> <source>' cut to max_chars; glossary terms come first."""

    glossary: Glossary
    name: str = "mock"
    usage: Usage = field(default_factory=Usage)
    calls: list[str] = field(default_factory=list)

    def _render(self, lang: str, line: LineIn) -> str:
        terms = self.glossary.protected_in(line.text)
        terms += [t.targets[lang] for t in self.glossary.terms_in(line.text) if lang in t.targets]
        body = f"{lang}: {' '.join(terms)} {line.text}".strip()
        out = ""
        for word in body.split():
            cand = f"{out} {word}".strip()
            if text_length(cand, lang) > line.max_chars:
                break
            out = cand
        return out or body[: line.max_chars]

    def translate(self, lang: str, lines: list[LineIn], context: list[tuple[str, str]],
                  course: CourseContext) -> dict[str, str]:
        self.calls.append("translate")
        self.usage.requests += 1
        return {x.id: self._render(lang, x) for x in lines}

    def revise(self, lang: str, lines: list[LineIn], course: CourseContext) -> dict[str, str]:
        self.calls.append("revise")
        self.usage.requests += 1
        return {x.id: self._render(lang, x) for x in lines}


def make_translator(cfg: Config, glossary: Glossary) -> Translator:
    if cfg.translate.backend == "anthropic":
        return AnthropicTranslator(cfg, glossary)
    return MockTranslator(glossary)
