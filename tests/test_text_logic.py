"""Glossary, sentence units, timing budgets and subtitle formatting."""

from __future__ import annotations

import itertools

from ritim.config import load_config
from ritim.glossary import Glossary
from ritim.segmentation import Word, build_units, fill_missing_times
from ritim.subtitles import make_cues, parse_vtt, to_srt, to_vtt, wrap
from ritim.timing import assign_slots, char_budget, decide_fit, shorter_budget, text_length


def glossary() -> Glossary:
    return Glossary.load(load_config().glossary_path())


def test_protected_terms_survive_turkish_suffixes_and_cjk_neighbours() -> None:
    g = glossary()
    found = g.protected_in("VDA 6.3'e göre FMEA'da 8D raporu, IATF 16949 ve Ritim Yönetim.")
    assert found == ["Ritim Yönetim", "IATF 16949", "VDA 6.3", "FMEA", "8D"]
    assert "VDA" not in found  # contained in the longer "VDA 6.3" match
    assert "Ritim" not in found
    assert g.protected_in("8Dレポートを作成します") == ["8D"]
    assert g.protected_in("PFMEA") == ["PFMEA"]  # not also FMEA
    assert g.protected_in("SPCs and MSAx") == []  # no partial-word hits


def test_glossary_check_flags_lost_terms_and_wrong_terminology() -> None:
    g = glossary()
    issues = g.check("İç denetçi eğitimi FMEA ile başlar.", "Die interne Prüfer-Schulung beginnt mit F-MEA.", "de")
    assert "missing protected term 'FMEA'" in issues
    assert any("internal_auditor" in i for i in issues)
    assert g.check("İç denetçi eğitimi FMEA ile başlar.",
                   "Die Schulung für interner Auditor beginnt mit FMEA.", "de") == []
    assert "VDA 6.3" in g.prompt_block("en") and "-> Schulung" in g.prompt_block("de")


def test_units_split_at_sentence_end_pause_and_length() -> None:
    words = [Word(word=w, start=i * 0.5, end=i * 0.5 + 0.4) for i, w in enumerate(
        ["Merhaba.", "Bugün", "8D", "konuşacağız", "ve", "sonra"])]
    words.append(Word(word="FMEA.", start=5.0, end=5.4))  # 2.1 s pause before
    units = build_units(words, max_unit_s=12, pause_s=0.8)
    assert [u["text"] for u in units] == ["Merhaba.", "Bugün 8D konuşacağız ve sonra", "FMEA."]
    assert units[1]["start"] == 0.5 and units[1]["end"] == 2.9
    long = [Word(word=f"w{i}" + ("," if i == 20 else ""), start=i * 0.5, end=i * 0.5 + 0.45) for i in range(40)]
    parts = build_units(long, max_unit_s=8, pause_s=0.8)
    assert all(u["end"] - u["start"] <= 8 for u in parts) and len(parts) >= 3
    assert [u["id"] for u in parts][:2] == ["u0001", "u0002"]


def test_fill_missing_times_interpolates_unaligned_tokens() -> None:
    raw = [{"word": "madde", "start": 1.0, "end": 1.4}, {"word": "8.5.1"}, {"word": "der", "start": 2.0, "end": 2.2}]
    words = fill_missing_times(raw, 1.0, 2.2)
    assert words[1]["start"] >= 1.4 and words[1]["end"] <= 2.0


def test_slots_budgets_and_fit() -> None:
    units = [{"start": 0.0, "end": 2.0}, {"start": 2.5, "end": 4.0}, {"start": 8.0, "end": 9.0}]
    assign_slots(units, chapter_duration=9.5, max_extend_s=1.2, min_gap_s=0.06)
    assert units[0]["slot_end"] == 2.44  # stops before the next line
    assert units[1]["slot_end"] == 5.2  # may run 1.2 s into the pause
    assert units[2]["slot_end"] == 9.5  # chapter end
    assert char_budget(units[1], cps=15, factor=0.95) == int(15 * 2.7 * 0.95)
    assert decide_fit(2.0, 2.5, 1.12).action == "ok"
    assert decide_fit(2.6, 2.5, 1.12).action == "tempo"
    assert decide_fit(3.5, 2.5, 1.12).action == "shorten"
    assert shorter_budget("x" * 60, "en", duration=4.0, slot=2.5, max_tempo=1.12) < 60
    assert text_length("品質 管理", "ja") == 4 and text_length(" ab c ", "en") == 4


def test_wrap_and_cues() -> None:
    assert all(len(line) <= 42 for line in wrap("word " * 30, 42, "en"))
    assert all(len(line) <= 18 for line in wrap("这是一个很长的中文句子，用来测试字幕换行功能是否正常。" * 2, 18, "zh"))
    units = [{"start": 0.0, "end": 3.0, "text": "A short line."},
             {"start": 2.9, "end": 9.0, "text": " ".join(["longer"] * 30)}]
    cues = make_cues(units, "en", 42)
    assert len(cues) >= 3  # the long unit needs more than two lines -> split
    for a, b in itertools.pairwise(cues):
        assert a.end <= b.start
    assert all(len(c.text.split("\n")) <= 2 for c in cues)
    vtt = to_vtt(cues)
    assert vtt.startswith("WEBVTT") and "00:00:00.000 --> " in vtt
    assert "00:00:00,000 --> " in to_srt(cues)
    back = parse_vtt(vtt)
    assert [round(c.start, 3) for c in back] == [round(c.start, 3) for c in cues]
