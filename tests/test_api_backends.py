"""Anthropic, ElevenLabs TTS and Scribe backends against mocked APIs (no network)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from ritim.backends.asr import AsrItem, ScribeAsr
from ritim.backends.translation import AnthropicTranslator, CourseContext, LineIn, MockTranslator
from ritim.backends.tts import ElevenLabsTTS, TTSItem
from ritim.config import Config
from ritim.glossary import Glossary
from ritim.media import audio_duration
from ritim.stages.text import translate_units

from .conftest import make_tone


class FakeMessages:
    def __init__(self, replies: list[dict[str, Any] | str], stop_reason: str = "end_turn") -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.stop_reason = stop_reason

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(
            stop_reason=self.stop_reason, stop_details=None,
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=80,
                                  cache_creation_input_tokens=0),
        )


class FakeAnthropic:
    def __init__(self, replies: list[dict[str, Any] | str], stop_reason: str = "end_turn") -> None:
        self.messages = FakeMessages(replies, stop_reason)
        self.beta = SimpleNamespace(messages=self.messages)


def test_anthropic_request_shape(cfg: Config) -> None:
    g = Glossary.load(cfg.glossary_path())
    client = FakeAnthropic([{"lines": [{"id": "u0001", "text": "We start the 8D report."}]}])
    tr = AnthropicTranslator(cfg, g, client=client)
    out = tr.translate("en", [LineIn("u0001", "8D raporuna başlıyoruz.", 40)], [("önce", "first")],
                       CourseContext("VDA 8D Problem Çözme Teknikleri"))
    assert out == {"u0001": "We start the 8D report."}
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["format"]["schema"]["required"] == ["lines"]
    assert "thinking" not in call and "temperature" not in call
    system = call["system"]
    assert "Never translate" in system[0]["text"] and "IATF 16949" in system[0]["text"]
    assert system[-1]["cache_control"] == {"type": "ephemeral"}
    assert "VDA 8D" in system[-1]["text"]
    user = json.loads(call["messages"][0]["content"])
    assert user["lines"][0] == {"id": "u0001", "text": "8D raporuna başlıyoruz.", "max_chars": 40}
    assert user["previous_lines_for_context"] == [{"source": "önce", "translation": "first"}]
    assert tr.usage.requests == 1 and tr.usage.cache_read_input_tokens == 80


def test_anthropic_refusal_raises(cfg: Config) -> None:
    g = Glossary.load(cfg.glossary_path())
    tr = AnthropicTranslator(cfg, g, client=FakeAnthropic([{"lines": []}], stop_reason="refusal"))
    with pytest.raises(RuntimeError, match="refused"):
        tr.translate("en", [LineIn("u1", "x", 10)], [], CourseContext("c"))


def test_anthropic_without_fallbacks_uses_plain_endpoint(cfg: Config) -> None:
    cfg.translate.fallbacks = "none"
    client = FakeAnthropic([{"lines": []}])
    AnthropicTranslator(cfg, Glossary.load(cfg.glossary_path()), client=client).translate(
        "de", [LineIn("u1", "x", 10)], [], CourseContext("c"))
    assert "betas" not in client.messages.calls[0]


class FlakyTranslator(MockTranslator):
    """First pass: drops one id, overshoots one budget, loses a protected term."""

    def translate(self, lang: str, lines: list[LineIn], context: list[tuple[str, str]],
                  course: CourseContext) -> dict[str, str]:
        if not self.calls:
            self.calls.append("translate")
            return {"u1": "x" * 80, "u2": "the eight-D report"}
        return super().translate(lang, lines, context, course)


def test_validation_rounds_fix_missing_long_and_lost_terms(cfg: Config) -> None:
    g = Glossary.load(cfg.glossary_path())
    tr = FlakyTranslator(g)
    units = [{"id": f"u{i}", "source": s, "max_chars": 40} for i, s in
             enumerate(["Bu bir deneme cümlesi.", "8D raporu.", "FMEA tablosu."], 1)]
    translate_units(tr, g, "en", units, [], CourseContext("c"), rounds=2)
    assert tr.calls[:3] == ["translate", "translate", "revise"]
    assert all(u["issues"] == [] for u in units), units
    assert "8D" in units[1]["text"] and "FMEA" in units[2]["text"]
    assert len(units[0]["text"]) <= 40


def _mp3_bytes(tmp_path: Path) -> bytes:
    wav = make_tone(tmp_path / "t.wav", 1.0)
    mp3 = tmp_path / "t.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "libmp3lame", str(mp3)],
                   check=True)
    return mp3.read_bytes()


def test_elevenlabs_tts_request_and_retry(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k-test")
    monkeypatch.setattr("ritim.backends.tts.time.sleep", lambda s: None)
    audio = _mp3_bytes(tmp_path)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(429, json={"detail": "busy"})
        return httpx.Response(200, content=audio, headers={"content-type": "audio/mpeg"})

    client = httpx.Client(base_url="https://api.elevenlabs.io", transport=httpx.MockTransport(handler))
    tts = ElevenLabsTTS(cfg, client=client, voice_id="voice123")
    item = TTSItem("L01-C001:u0001", "We start the 8D report.", "en", tmp_path / "out" / "u0001.wav",
                   prev_text="Hello.", next_text="")
    tts.synthesize([item], tmp_path / "job.json")
    assert item.out.exists() and 0.9 < audio_duration(item.out) < 1.2
    req = seen[-1]
    assert req.url.path == "/v1/text-to-speech/voice123"
    assert req.url.params["output_format"] == "mp3_44100_192"
    assert req.headers["xi-api-key"] == "k-test"
    body = json.loads(req.content)
    assert body["model_id"] == "eleven_multilingual_v2" and body["previous_text"] == "Hello."
    assert body["next_text"] is None and "language_code" not in body
    assert len(seen) == 2 and tts.usage["characters"] == len(item.text)
    tts.synthesize([item], tmp_path / "job.json")  # existing output is not re-billed
    assert len(seen) == 2


def test_scribe_backend_parses_words(cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k-test")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "Merhaba 8D.", "words": [
            {"text": "Merhaba", "start": 0.1, "end": 0.5, "type": "word", "logprob": 0},
            {"text": " ", "start": 0.5, "end": 0.6, "type": "spacing", "logprob": 0},
            {"text": "8D.", "start": 0.6, "end": 0.9, "type": "word", "logprob": 0},
            {"text": "(laughter)", "start": 1.0, "end": 1.2, "type": "audio_event", "logprob": 0}]})

    client = httpx.Client(base_url="https://api.elevenlabs.io", transport=httpx.MockTransport(handler))
    asr = ScribeAsr(cfg, ["8D", "FMEA"], client=client)
    wav = make_tone(tmp_path / "a.wav", 1.5)
    words = asr.transcribe([AsrItem("c1", wav, tmp_path / "c1.raw.json")], tmp_path / "job.json")["c1"]
    assert [w["word"] for w in words] == ["Merhaba", "8D."]
    assert captured["path"] == "/v1/speech-to-text"
    body = captured["body"]
    assert b'name="model_id"' in body and b"scribe_v2" in body and b'name="keyterms"' in body
    assert (tmp_path / "c1.raw.json").exists()
