# metacog/gauges.py — Jauges temps réel
from __future__ import annotations

from dataclasses import dataclass

from db.metacog import CRITERIA

SESSION_INHERITANCE_FACTOR = 0.8
PROFILE_SESSION_WEIGHT = 0.1


@dataclass
class GaugeState:
    name: str
    value: float

    def update(
        self,
        signal: float = 0.0,
        verdict: str | None = None,
        response_time_ms: int | None = None,
        consecutive_incorrect: int = 0,
    ) -> float:
        if self.name == "attention":
            self.value = _update_attention(
                self.value,
                signal,
                verdict,
                response_time_ms,
                consecutive_incorrect,
            )
        elif self.name == "meta_cognition":
            self.value = _clamp(self.value)
        else:
            delta = max(-2.0, min(2.0, float(signal))) * 8.0
            if verdict == "correct":
                delta += 1.5
            elif verdict == "partial":
                delta += 0.3
            elif verdict == "incorrect":
                delta -= 1.5
            self.value = _clamp(self.value + delta)
        return self.value

    def apply_delta(self, delta: float) -> float:
        self.value = _clamp(self.value + float(delta))
        return self.value


def make_gauges(profile: dict | None = None) -> dict[str, GaugeState]:
    values = initialize_session_gauges(profile or {})
    return {
        criterion: GaugeState(criterion, values[criterion])
        for criterion in CRITERIA
    }


def initialize_session_gauges(profile_gauges: dict | None) -> dict[str, float]:
    profile_gauges = profile_gauges or {}
    values: dict[str, float] = {}
    for criterion in CRITERIA:
        values[criterion] = _clamp(float(profile_gauges.get(criterion, 50.0)) * SESSION_INHERITANCE_FACTOR)
    return values


def update_profile_gauges_from_session(
    profile_gauges: dict | None,
    session_gauges: dict | None,
    session_weight: float = PROFILE_SESSION_WEIGHT,
) -> dict[str, float]:
    profile_gauges = profile_gauges or {}
    session_gauges = session_gauges or {}
    weight = max(0.0, min(1.0, float(session_weight)))
    profile_weight = 1.0 - weight
    updates: dict[str, float] = {}
    for criterion in CRITERIA:
        current = _clamp(float(profile_gauges.get(criterion, 50.0)))
        session_value = _clamp(float(session_gauges.get(criterion, current)))
        updates[criterion] = _clamp(current * profile_weight + session_value * weight)
    return updates


def update_gauges_from_evaluation(
    gauges: dict[str, GaugeState],
    evaluation: dict,
    response_time_ms: int | None = None,
    consecutive_incorrect: int = 0,
) -> dict[str, float]:
    signals = evaluation.get("metacog_signals") or {}
    verdict = evaluation.get("verdict")
    values = {}
    for criterion, gauge in gauges.items():
        if criterion == "meta_cognition":
            values[criterion] = gauge.value
            continue
        values[criterion] = gauge.update(
            signal=_effective_signal(criterion, signals, evaluation),
            verdict=verdict,
            response_time_ms=response_time_ms,
            consecutive_incorrect=consecutive_incorrect,
        )
    return values


def snapshot(gauges: dict[str, GaugeState]) -> dict[str, float]:
    return {name: gauge.value for name, gauge in gauges.items()}


def clamp_gauge(value: float) -> float:
    return _clamp(value)


def _effective_signal(criterion: str, signals: dict, evaluation: dict) -> float:
    try:
        signal = float(signals.get(criterion, 0.0))
    except (TypeError, ValueError):
        signal = 0.0

    if criterion == "curiosity":
        curiosity_signals = evaluation.get("curiosity_signals") or {}
        if isinstance(curiosity_signals, dict) and any(bool(value) for value in curiosity_signals.values()):
            signal += 0.6
    elif criterion == "creativity":
        creativity_signals = evaluation.get("creativity_signals") or {}
        if isinstance(creativity_signals, dict):
            positives = sum(
                1
                for key in ("goes_beyond_prompt", "makes_connections", "uses_analogy", "personal_reformulation", "original_hypothesis")
                if creativity_signals.get(key)
            )
            try:
                depth = float(creativity_signals.get("depth_of_reflection", 0.0))
            except (TypeError, ValueError):
                depth = 0.0
            if positives:
                signal += min(0.7, positives * 0.18)
            if depth >= 0.65:
                signal += 0.25
            elif depth <= 0.2:
                signal -= 0.15

    return max(-2.0, min(2.0, signal))


def _update_attention(
    value: float,
    signal: float,
    verdict: str | None,
    response_time_ms: int | None,
    consecutive_incorrect: int,
) -> float:
    delta = max(-2.0, min(2.0, float(signal))) * 5.0
    if response_time_ms is not None and response_time_ms > 12000:
        delta -= min(12.0, (response_time_ms - 12000) / 2000.0)
    if verdict == "correct":
        delta += 1.0
    elif verdict == "partial":
        delta -= 1.0
    elif verdict == "incorrect":
        delta -= 3.0
    delta -= max(0, consecutive_incorrect - 1) * 2.0
    return _clamp(value + delta)


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))
