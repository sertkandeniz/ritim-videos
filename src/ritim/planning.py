"""Pure planning logic: where to cut chapters and which stretches show a large face."""

from __future__ import annotations

from typing import Any, TypedDict


class FaceSegment(TypedDict):
    start: float
    end: float
    box: list[float]  # union face box, normalised x1, y1, x2, y2


def large_face_segments(samples: list[dict[str, Any]], min_height: float, min_len_s: float,
                        period_s: float, max_hole: int = 1) -> list[FaceSegment]:
    """Merge consecutive samples whose largest face is tall enough into segments.

    `samples` are {"t", "h", "box"} with `h` the face box height as a fraction of the frame
    (0 when there is no face). Up to `max_hole` missed samples inside a run are tolerated.
    """
    segs: list[FaceSegment] = []
    run: list[dict[str, Any]] = []
    holes = 0

    def close() -> None:
        if not run:
            return
        start, end = run[0]["t"], run[-1]["t"] + period_s
        if end - start >= min_len_s:
            boxes = [s["box"] for s in run]
            segs.append(FaceSegment(
                start=round(start, 3), end=round(end, 3),
                box=[min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes)],
            ))

    for s in samples:
        if s.get("h", 0.0) >= min_height and s.get("box"):
            run.append(s)
            holes = 0
        elif run and holes < max_hole:
            holes += 1
        else:
            close()
            run, holes = [], 0
    close()
    return segs


def _inside(t: float, segments: list[FaceSegment]) -> bool:
    return any(s["start"] < t < s["end"] for s in segments)


def plan_chapters(duration: float, silences: list[tuple[float, float]], scenes: list[float],
                  min_s: float, target_s: float, max_s: float,
                  avoid: list[FaceSegment] | None = None) -> list[tuple[float, float]]:
    """Cut a lesson into chapters of min_s..max_s seconds, preferring long pauses at scene changes.

    Cuts land in the middle of a silence (so no word is split) and never inside a large-face
    segment (so lip sync never straddles two chapters). Falls back to a hard cut at max_s.
    """
    avoid = avoid or []
    if duration <= max_s:
        return [(0.0, round(duration, 3))]
    candidates: list[tuple[float, float]] = []  # (time, base score)
    for a, b in silences:
        t = (a + b) / 2
        score = min(b - a, 3.0)
        if any(abs(sc - t) <= max(1.0, (b - a) / 2 + 0.5) for sc in scenes):
            score += 2.0
        candidates.append((t, score))
    for sc in scenes:  # a scene change alone is a weaker cut point
        candidates.append((sc, 0.3))

    cuts: list[float] = []
    pos = 0.0
    while duration - pos > max_s:
        lo, hi = pos + min_s, pos + max_s
        best_t, best = None, float("-inf")
        for t, base in candidates:
            if lo <= t <= hi and not _inside(t, avoid):
                score = base - 1.5 * abs(t - (pos + target_s)) / target_s
                if score > best:
                    best_t, best = t, score
        cut = best_t if best_t is not None else hi
        cuts.append(round(cut, 3))
        pos = cut
    # A short tail is merged into the last chapter when that stays within 1.25 x max.
    if cuts and duration - cuts[-1] < min_s:
        prev = cuts[-2] if len(cuts) > 1 else 0.0
        if duration - prev <= max_s * 1.25:
            cuts.pop()
    bounds = [0.0, *cuts, round(duration, 3)]
    return list(zip(bounds[:-1], bounds[1:], strict=True))
