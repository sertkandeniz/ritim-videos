"""Never-translate list and terminology table (glossary/never_translate.txt, glossary/terms.yaml)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Latin letters/digits only: CJK characters next to a term ("8Dの") must not block a match.
_EDGE = r"A-Za-z0-9"


def tr_lower(text: str) -> str:
    """Turkish-aware lower-casing (İ -> i, I -> ı)."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def term_pattern(term: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(part) for part in term.split())
    return re.compile(rf"(?<![{_EDGE}]){body}(?![{_EDGE}])")


@dataclass
class Term:
    id: str
    status: str
    tr: list[str]
    targets: dict[str, str]


@dataclass
class Glossary:
    never_translate: list[str]
    terms: list[Term]
    _patterns: dict[str, re.Pattern[str]] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, directory: Path) -> Glossary:
        never: list[str] = []
        nt = directory / "never_translate.txt"
        if nt.exists():
            for line in nt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    never.append(line)
        terms: list[Term] = []
        tf = directory / "terms.yaml"
        if tf.exists():
            data = yaml.safe_load(tf.read_text(encoding="utf-8")) or {}
            for raw in data.get("terms", []):
                targets = {k: str(v) for k, v in raw.items() if k not in ("id", "status", "tr")}
                terms.append(Term(id=raw["id"], status=raw.get("status", "review"),
                                  tr=list(raw.get("tr", [])), targets=targets))
        # longest first so "VDA 6.3" wins over "VDA"
        never.sort(key=len, reverse=True)
        return cls(never_translate=never, terms=terms)

    def _pattern(self, term: str) -> re.Pattern[str]:
        if term not in self._patterns:
            self._patterns[term] = term_pattern(term)
        return self._patterns[term]

    def protected_in(self, text: str) -> list[str]:
        """Never-translate terms in `text`; a term inside a longer matched term is not repeated."""
        found: list[str] = []
        taken: list[tuple[int, int]] = []
        for term in self.never_translate:
            for m in self._pattern(term).finditer(text):
                span = m.span()
                if any(s <= span[0] and span[1] <= e for s, e in taken):
                    continue
                taken.append(span)
                if term not in found:
                    found.append(term)
        return found

    def terms_in(self, source_text: str) -> list[Term]:
        low = tr_lower(source_text)
        return [t for t in self.terms if any(tr_lower(stem) in low for stem in t.tr)]

    def prompt_block(self, lang: str) -> str:
        lines = ["Never translate, transliterate or re-space these terms; copy them exactly:"]
        lines.append(", ".join(sorted(self.never_translate, key=str.lower)))
        lines.append("")
        lines.append(f"Mandatory terminology (Turkish -> {lang}):")
        for t in self.terms:
            if lang in t.targets:
                lines.append(f"- {' / '.join(t.tr)} -> {t.targets[lang]}")
        return "\n".join(lines)

    def check(self, source: str, target: str, lang: str) -> list[str]:
        """Problems with one translated line: lost protected terms, missing approved terminology."""
        issues = [f"missing protected term {term!r}" for term in self.protected_in(source)
                  if not self._pattern(term).search(target)]
        low_target = target.lower()
        for t in self.terms_in(source):
            want = t.targets.get(lang)
            if want and t.status == "approved" and want.lower() not in low_target:
                issues.append(f"term {t.id!r} should be rendered as {want!r}")
        return issues
