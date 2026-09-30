from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Callable, Literal, Mapping
import warnings

import numpy as np
import optuna
from scipy.stats import qmc


SearchScale = Literal["linear", "log"]


@dataclass(frozen=True)
class SearchParameter:
    name: str
    low: float
    high: float
    scale: SearchScale = "linear"

    def value_from_unit(self, value: float) -> float:
        position = float(np.clip(value, 0.0, 1.0))
        if self.scale == "log":
            return float(math.exp(math.log(self.low) + position * math.log(self.high / self.low)))
        return float(self.low + position * (self.high - self.low))


@dataclass(frozen=True)
class OptimizationEvaluation:
    score: float
    components: Mapping[str, float] = field(default_factory=dict)
    constraint_violations: tuple[float, ...] = ()

    @property
    def feasible(self) -> bool:
        return bool(
            np.isfinite(float(self.score))
            and all(np.isfinite(value) and value <= 0.0 for value in self.constraint_violations)
        )


@dataclass(frozen=True)
class SobolOptunaSettings:
    sobol_trials: int = 32
    optuna_trials: int = 64
    seed: int = 0
    timeout_seconds: float | None = 2.0
    top_k: int = 3


@dataclass(frozen=True)
class OptimizationTrialResult:
    params: dict[str, float]
    evaluation: OptimizationEvaluation
    source: str


@dataclass(frozen=True)
class SobolOptunaResult:
    best: OptimizationTrialResult
    top_trials: tuple[OptimizationTrialResult, ...]
    completed_trials: int
    feasible_trials: int
    sobol_trials: int
    optuna_trials: int
    seed: int
    elapsed_seconds: float


def _validated_sobol_count(value: int) -> int:
    count = int(value)
    if count < 1 or count & (count - 1):
        raise ValueError("Sobol trial count must be a positive power of two")
    return count


def optimize_sobol_optuna(
    parameters: tuple[SearchParameter, ...],
    objective: Callable[[dict[str, float]], OptimizationEvaluation],
    *,
    settings: SobolOptunaSettings = SobolOptunaSettings(),
    initial_params: tuple[Mapping[str, float], ...] = (),
) -> SobolOptunaResult:
    """Run deterministic Sobol initialization followed by multivariate TPE."""

    started_at = perf_counter()
    if not parameters:
        raise ValueError("Sobol + Optuna optimization requires at least one parameter")
    names = [item.name for item in parameters]
    if len(set(names)) != len(names):
        raise ValueError("Search parameter names must be unique")
    for item in parameters:
        if (
            not np.isfinite(item.low)
            or not np.isfinite(item.high)
            or item.high < item.low
            or item.scale not in {"linear", "log"}
            or (item.scale == "log" and item.low <= 0.0)
        ):
            raise ValueError(f"invalid search bounds for {item.name}")

    sobol_count = _validated_sobol_count(settings.sobol_trials)
    try:
        engine = qmc.Sobol(
            d=len(parameters),
            scramble=True,
            rng=np.random.default_rng(int(settings.seed)),
        )
    except TypeError:  # SciPy < 1.15 uses ``seed`` instead of ``rng``.
        engine = qmc.Sobol(
            d=len(parameters),
            scramble=True,
            seed=int(settings.seed),
        )
    points = engine.random_base2(int(math.log2(sobol_count)))
    queued: list[dict[str, float]] = []
    initial_count = len(initial_params)
    for supplied in initial_params:
        values = {name: float(supplied[name]) for name in names}
        for parameter in parameters:
            if not parameter.low <= values[parameter.name] <= parameter.high:
                raise ValueError(f"initial value for {parameter.name} is outside search bounds")
        queued.append(values)
    for point in points:
        queued.append(
            {
                parameter.name: parameter.value_from_unit(float(point[index]))
                for index, parameter in enumerate(parameters)
            }
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=int(settings.seed),
            multivariate=True,
            group=False,
            n_startup_trials=len(queued),
        )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    for values in queued:
        study.enqueue_trial(values)

    evaluated: list[OptimizationTrialResult] = []

    def optuna_objective(trial: optuna.Trial) -> float:
        values: dict[str, float] = {}
        for parameter in parameters:
            values[parameter.name] = trial.suggest_float(
                parameter.name,
                parameter.low,
                parameter.high,
                log=parameter.scale == "log",
            )
        evaluation = objective(values)
        if trial.number < initial_count:
            source = "initial"
        elif trial.number < len(queued):
            source = "sobol"
        else:
            source = "optuna"
        evaluated.append(OptimizationTrialResult(dict(values), evaluation, source))
        trial.set_user_attr("components", dict(evaluation.components))
        trial.set_user_attr("constraint_violations", list(evaluation.constraint_violations))
        trial.set_user_attr("feasible", evaluation.feasible)
        if evaluation.feasible:
            return float(evaluation.score)
        violation = sum(max(float(value), 0.0) for value in evaluation.constraint_violations)
        return 1_000_000.0 + violation if np.isfinite(violation) else 10_000_000.0

    optimize_kwargs = {
        "n_jobs": 1,
        "show_progress_bar": False,
        "catch": (FloatingPointError, ValueError),
    }
    # Complete the balanced 2**m Sobol design before applying the adaptive
    # timeout to TPE. Otherwise slow objective calls could truncate the space-
    # filling initialization and make runs less comparable.
    study.optimize(optuna_objective, n_trials=len(queued), **optimize_kwargs)
    requested_optuna_trials = max(int(settings.optuna_trials), 0)
    if requested_optuna_trials:
        study.optimize(
            optuna_objective,
            n_trials=requested_optuna_trials,
            timeout=(
                None
                if settings.timeout_seconds is None
                else max(float(settings.timeout_seconds), 0.01)
            ),
            **optimize_kwargs,
        )
    feasible = sorted(
        (item for item in evaluated if item.evaluation.feasible),
        key=lambda item: item.evaluation.score,
    )
    if not feasible:
        raise ValueError("Sobol + Optuna search produced no feasible trial")
    top_count = max(1, int(settings.top_k))
    optuna_count = sum(item.source == "optuna" for item in evaluated)
    sobol_evaluated = sum(item.source == "sobol" for item in evaluated)
    return SobolOptunaResult(
        best=feasible[0],
        top_trials=tuple(feasible[:top_count]),
        completed_trials=len(evaluated),
        feasible_trials=len(feasible),
        sobol_trials=sobol_evaluated,
        optuna_trials=optuna_count,
        seed=int(settings.seed),
        elapsed_seconds=perf_counter() - started_at,
    )
