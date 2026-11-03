"""LSTM / GRU forecasters for the AGGREGATE CA_1/FOODS_3 daily sales series (Phase 3, notebook 07).

What this module is
-------------------
A small recurrent network (one LSTM or GRU layer) that reads the last `input_window` days and
emits all `horizon` (28) future days in ONE forward pass -- a *direct* multi-horizon forecaster.
Nothing is fed back into the network, so no forecast ever becomes an input (no recursion) and
no actual future sales value can enter a prediction.

Two information sets (the "variants", pre-declared in config.yaml, section `neural`):

  calendar     the past sales window plus calendar facts that are PUBLISHED IN ADVANCE and
               therefore known for the forecast days too: day of week, SNAP (CA), the M5
               event types and the Christmas closure flag. Known future calendar columns for
               the 28 forecast days go straight into the output layer.
  univariate   the past sales window only -- the same information set as the Phase 2
               classical models (Seasonal Naive, ARIMA, SARIMA, Holt-Winters).

Leakage rules enforced here (each one has a unit test in tests/models/test_lstm_gru.py)
-------------------------------------------------------------------------------------
* Training samples are built only from the history `fit` receives. A sample at origin position
  t uses days t-W+1..t as input and t+1..t+H as target, and t+H never exceeds the last day of
  the history.
* Early stopping uses an INNER chronological validation tail: the last `validation_days`
  days of the training history. Training targets stop before that tail starts; validation
  targets lie entirely inside it. The tail is never used to update the weights.
* The standardising scaler (mean, std) is fit on the inner-TRAIN part only (everything before
  the tail) -- not on the tail, and never on the forecast window.
* Calendar covariates are 0/1 facts, never sales. A calendar that does not cover a requested
  date raises; a missing date is never filled with zeros.
* Forecasts are clipped at zero (units sold cannot be negative). This is the documented
  non-negativity method; the number of values it changed is reported in `fit_diagnostics()`.

PyTorch is imported lazily. Importing this module never needs torch; using a forecaster
without it raises an ImportError that names the exact install command. No other method is
ever substituted and labelled LSTM or GRU.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.models.xgboost_model import EVENT_FLAG_COLUMNS
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_CELLS: Tuple[str, ...] = ("lstm", "gru")
SUPPORTED_FEATURES: Tuple[str, ...] = ("calendar", "univariate")
SUPPORTED_LOSSES: Tuple[str, ...] = ("huber", "mse")
DAY_OF_WEEK_NAMES: Tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
#: One-hot day of week (Monday first) followed by the six published-in-advance event flags.
CALENDAR_FEATURE_NAMES: Tuple[str, ...] = tuple(f"dow_{d}" for d in DAY_OF_WEEK_NAMES) + tuple(EVENT_FLAG_COLUMNS)
N_CALENDAR_FEATURES = len(CALENDAR_FEATURE_NAMES)

#: A fold whose inner-train region yields fewer samples than this is refused, not fitted.
MIN_TRAIN_WINDOWS = 10

TORCH_INSTALL_COMMAND = "python -m pip install torch"
PHASE3_REQUIREMENTS_COMMAND = "python -m pip install -r requirements-phase3.txt"


# =============================================================================================
# Configuration
# =============================================================================================

def _config_error(key: str, message: str) -> ValueError:
    return ValueError(f"Invalid configuration: neural.{key} {message}")


def _positive_int(section: Mapping[str, Any], key: str, minimum: int = 1) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise _config_error(key, f"must be an integer >= {minimum} (got {value!r}).")
    return int(value)


def _positive_float(section: Mapping[str, Any], key: str, allow_zero: bool = False) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
        raise _config_error(key, f"must be a finite number (got {value!r}).")
    value = float(value)
    if value < 0 or (value == 0 and not allow_zero):
        raise _config_error(key, f"must be {'>= 0' if allow_zero else '> 0'} (got {value!r}).")
    return value


@dataclass(frozen=True)
class NeuralConfig:
    """Validated training settings (config.yaml, section `neural`, plus the backtest horizon)."""

    input_window: int = 56
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 0.002
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    validation_days: int = 56
    loss: str = "huber"
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    torch_num_threads: int = 1
    horizon: int = 28

    def __post_init__(self) -> None:
        for key, minimum in (("input_window", 2), ("hidden_size", 1), ("num_layers", 1), ("batch_size", 1),
                             ("max_epochs", 1), ("patience", 1), ("torch_num_threads", 1), ("horizon", 1)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise _config_error(key, f"must be an integer >= {minimum} (got {value!r}).")
        if not (isinstance(self.dropout, (int, float)) and 0.0 <= self.dropout < 1.0):
            raise _config_error("dropout", f"must be in [0, 1) (got {self.dropout!r}).")
        for key in ("learning_rate", "huber_delta", "grad_clip"):
            value = getattr(self, key)
            if not (isinstance(value, (int, float)) and np.isfinite(value) and value > 0):
                raise _config_error(key, f"must be a positive number (got {value!r}).")
        if not (isinstance(self.weight_decay, (int, float)) and np.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise _config_error("weight_decay", f"must be >= 0 (got {self.weight_decay!r}).")
        if self.loss not in SUPPORTED_LOSSES:
            raise _config_error("loss", f"must be one of {list(SUPPORTED_LOSSES)} (got {self.loss!r}).")
        if self.validation_days < self.horizon or not isinstance(self.validation_days, (int, np.integer)):
            raise _config_error(
                "validation_days",
                f"must be an integer >= the horizon ({self.horizon}) so the validation tail holds at least one "
                f"full forecast window (got {self.validation_days!r}).",
            )
        seeds = tuple(self.seeds)
        if len(seeds) == 0 or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) for s in seeds):
            raise _config_error("seeds", f"must be a non-empty list of integers (got {self.seeds!r}).")
        if len(set(seeds)) != len(seeds):
            raise _config_error("seeds", f"must not contain duplicates (got {list(seeds)!r}).")
        object.__setattr__(self, "seeds", tuple(int(s) for s in seeds))

    @property
    def required_history(self) -> int:
        """Shortest training history that leaves MIN_TRAIN_WINDOWS inner-train samples."""
        return self.input_window + self.horizon + self.validation_days + MIN_TRAIN_WINDOWS - 1

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None, horizon: Optional[int] = None) -> "NeuralConfig":
        """Read and validate `config["neural"]`; the horizon defaults to `config["backtest"]["horizon"]`."""
        config = load_config() if config is None else config
        if "neural" not in config or not isinstance(config["neural"], Mapping):
            raise ValueError("Invalid configuration: config.yaml has no 'neural' section.")
        section = config["neural"]
        if horizon is None:
            backtest = config.get("backtest") or {}
            if "horizon" not in backtest:
                raise ValueError("Invalid configuration: backtest.horizon is missing.")
            horizon = backtest["horizon"]
        return cls(
            input_window=_positive_int(section, "input_window", 2),
            hidden_size=_positive_int(section, "hidden_size"),
            num_layers=_positive_int(section, "num_layers"),
            dropout=_positive_float(section, "dropout", allow_zero=True),
            learning_rate=_positive_float(section, "learning_rate"),
            weight_decay=_positive_float(section, "weight_decay", allow_zero=True),
            batch_size=_positive_int(section, "batch_size"),
            max_epochs=_positive_int(section, "max_epochs"),
            patience=_positive_int(section, "patience"),
            validation_days=_positive_int(section, "validation_days"),
            loss=str(section.get("loss", "")),
            huber_delta=_positive_float(section, "huber_delta"),
            grad_clip=_positive_float(section, "grad_clip"),
            seeds=tuple(section.get("seeds") or ()),
            torch_num_threads=_positive_int(section, "torch_num_threads"),
            horizon=horizon,
        )


@dataclass(frozen=True)
class NeuralVariant:
    """One pre-declared experiment cell: a recurrent cell type and an information set."""

    cell: str
    features: str

    @property
    def name(self) -> str:
        return variant_name(self.cell, self.features)


def variant_name(cell: str, features: str) -> str:
    """The stable model name used in every saved table, e.g. "LSTM(calendar)"."""
    _check_cell_and_features(cell, features)
    return f"{cell.upper()}({features})"


def _check_cell_and_features(cell: str, features: str) -> None:
    if cell not in SUPPORTED_CELLS:
        raise ValueError(f"cell must be one of {list(SUPPORTED_CELLS)} (got {cell!r}).")
    if features not in SUPPORTED_FEATURES:
        raise ValueError(f"features must be one of {list(SUPPORTED_FEATURES)} (got {features!r}).")


def variants_from_config(config: Optional[Mapping[str, Any]] = None) -> List[NeuralVariant]:
    """The experiment cells declared in `config["neural"]["variants"]` (validated, no duplicates)."""
    config = load_config() if config is None else config
    section = config.get("neural") if isinstance(config, Mapping) else None
    if not isinstance(section, Mapping):
        raise ValueError("Invalid configuration: config.yaml has no 'neural' section.")
    declared = section.get("variants")
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)) or len(declared) == 0:
        raise _config_error("variants", f"must be a non-empty list of {{cell, features}} entries (got {declared!r}).")
    variants: List[NeuralVariant] = []
    for entry in declared:
        if not isinstance(entry, Mapping) or set(entry) != {"cell", "features"}:
            raise _config_error("variants", f"entries must have exactly the keys 'cell' and 'features' (got {entry!r}).")
        try:
            _check_cell_and_features(str(entry["cell"]), str(entry["features"]))
        except ValueError as exc:
            raise _config_error("variants", f"has an invalid entry: {exc}") from exc
        variants.append(NeuralVariant(str(entry["cell"]), str(entry["features"])))
    if len({v.name for v in variants}) != len(variants):
        raise _config_error("variants", "must not list the same variant twice.")
    primary = section.get("primary_features")
    if primary not in SUPPORTED_FEATURES:
        raise _config_error("primary_features", f"must be one of {list(SUPPORTED_FEATURES)} (got {primary!r}).")
    return variants


# =============================================================================================
# Pure helpers (numpy / pandas only -- importable and testable without torch)
# =============================================================================================

def check_history(history: pd.Series) -> np.ndarray:
    """Validate a training history and return its values as a float array.

    It must be a date-indexed, ascending, gap-free DAILY series of finite, non-negative sales.
    """
    if not isinstance(history, pd.Series):
        raise TypeError("history must be a pandas Series of daily sales indexed by date.")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("history must be indexed by date (a pd.DatetimeIndex).")
    if len(history) == 0:
        raise ValueError("history is empty.")
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("history must be sorted ascending by date with no duplicated dates.")
    expected = pd.date_range(history.index[0], periods=len(history), freq="D")
    if not history.index.equals(expected):
        raise ValueError("history must be a continuous daily series (no missing calendar days).")
    values = history.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("history contains missing or non-finite values.")
    if np.any(values < 0):
        raise ValueError("history contains negative sales; unit sales cannot be negative.")
    return values


def calendar_feature_matrix(index: pd.DatetimeIndex, event_calendar: pd.DataFrame) -> np.ndarray:
    """Calendar covariates for `index`: one-hot day of week (Mon..Sun) + the six event flags.

    Shape (len(index), 13), float32. Only calendar FACTS are used (they are published before
    the day happens), never sales. If the event calendar does not cover every requested date,
    this raises -- a missing date is never filled with zeros.
    """
    index = pd.DatetimeIndex(index)
    if not isinstance(event_calendar, pd.DataFrame) or not isinstance(event_calendar.index, pd.DatetimeIndex):
        raise ValueError("event_calendar must be a DataFrame indexed by date (see load_event_calendar).")
    absent = [c for c in EVENT_FLAG_COLUMNS if c not in event_calendar.columns]
    if absent:
        raise ValueError(f"event_calendar is missing column(s): {absent}.")
    uncovered = index.difference(event_calendar.index)
    if len(uncovered):
        raise ValueError(
            f"The event calendar does not cover {len(uncovered)} requested date(s) "
            f"(first: {uncovered[0].date()}). Missing dates are never filled with zeros."
        )
    flags = event_calendar.loc[index, list(EVENT_FLAG_COLUMNS)].to_numpy(dtype=float)
    if not np.all(np.isfinite(flags)) or not np.all((flags == 0) | (flags == 1)):
        raise ValueError("event_calendar flags must be 0/1 with no missing values.")
    day_of_week = np.eye(7, dtype=np.float32)[np.asarray(index.dayofweek, dtype=int)]
    return np.concatenate([day_of_week, flags.astype(np.float32)], axis=1)


@dataclass(frozen=True)
class Scaler:
    """Standardiser (x - mean) / scale. Fit it on training data ONLY."""

    mean: float
    scale: float

    @classmethod
    def fit(cls, values: Sequence[float]) -> "Scaler":
        values = np.asarray(values, dtype=float)
        if values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError("A scaler needs at least two finite training values.")
        std = float(values.std(ddof=0))
        # A constant training series has no spread to divide by: use 1.0 so it maps to zeros.
        return cls(mean=float(values.mean()), scale=std if std > 1e-8 else 1.0)

    def transform(self, values: Sequence[float]) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse(self, values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


def split_origins(n_obs: int, input_window: int, horizon: int, validation_days: int) -> Tuple[np.ndarray, np.ndarray]:
    """Origin positions of the training and the inner-validation samples of a history of `n_obs` days.

    An origin t is the position of the LAST observed day of a sample's input window; its targets are
    positions t+1 .. t+horizon. With V = validation_days, the inner-validation tail is positions
    n_obs-V .. n_obs-1:

      training samples    t from input_window-1 to n_obs-V-1-horizon   (targets end BEFORE the tail)
      validation samples  t from n_obs-V-1       to n_obs-1-horizon    (targets entirely INSIDE the tail)

    so no training target ever lies in the tail, and the V-horizon+1 validation samples only
    check generalisation. Raises if fewer than MIN_TRAIN_WINDOWS training samples would remain.
    """
    if validation_days < horizon:
        raise ValueError(f"validation_days ({validation_days}) must be at least the horizon ({horizon}).")
    first = input_window - 1
    train = np.arange(first, n_obs - validation_days - horizon)
    validation = np.arange(n_obs - validation_days - 1, n_obs - horizon)
    if len(train) < MIN_TRAIN_WINDOWS:
        needed = input_window + horizon + validation_days + MIN_TRAIN_WINDOWS - 1
        raise ValueError(
            f"A history of {n_obs} days is too short: input window {input_window} + horizon {horizon} + validation "
            f"tail {validation_days} needs at least {needed} days to leave {MIN_TRAIN_WINDOWS} training samples "
            f"(only {max(len(train), 0)} would remain)."
        )
    return train, validation


def window_indices(origins: Sequence[int], input_window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Row positions used by each sample: (past (n, input_window), future (n, horizon))."""
    origins = np.asarray(origins, dtype=int)
    past = origins[:, None] + np.arange(-input_window + 1, 1)[None, :]
    future = origins[:, None] + np.arange(1, horizon + 1)[None, :]
    return past, future


def build_windows(
    values: np.ndarray,
    covariates: Optional[np.ndarray],
    origins: Sequence[int],
    input_window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut supervised samples out of a (scaled) history.

    values      (T,) scaled sales of the history
    covariates  (T, k) calendar features aligned with `values`, or None for a univariate model
    origins     positions t (last day of each input window); see `split_origins`
    returns     past (n, input_window, 1 + k), future (n, horizon, k), target (n, horizon); float32.

    A sample at t reads positions t-input_window+1 .. t for its input and t+1 .. t+horizon for its
    target; both ranges must lie inside the history, otherwise this raises.
    """
    values = np.asarray(values, dtype=np.float32)
    origins = np.asarray(origins, dtype=int)
    n_obs = len(values)
    if covariates is not None and len(covariates) != n_obs:
        raise ValueError(f"covariates has {len(covariates)} rows for {n_obs} observations.")
    if origins.size and (origins.min() < input_window - 1 or origins.max() > n_obs - 1 - horizon):
        raise ValueError(
            f"origin positions must lie in [{input_window - 1}, {n_obs - 1 - horizon}] so that both the input window "
            "and the target window fall inside the history."
        )
    idx_past, idx_future = window_indices(origins, input_window, horizon)
    past = values[idx_past][..., None]
    target = values[idx_future]
    if covariates is None:
        return past, np.zeros((len(origins), horizon, 0), dtype=np.float32), target
    covariates = np.asarray(covariates, dtype=np.float32)
    return np.concatenate([past, covariates[idx_past]], axis=2), covariates[idx_future], target


def make_forecast_input(
    values: np.ndarray,
    past_covariates: Optional[np.ndarray],
    future_covariates: Optional[np.ndarray],
    input_window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """The single input that produces the forecast: the LAST `input_window` observed days.

    `future_covariates` holds the published-in-advance calendar features of the `horizon` days to be
    forecast (shape (horizon, k)); for a univariate model both covariate arguments are None.
    Returns past (1, input_window, 1 + k) and future (1, horizon, k).
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values) < input_window:
        raise ValueError(f"need at least {input_window} observed days to forecast (got {len(values)}).")
    past = values[-input_window:][None, :, None]
    if past_covariates is None or future_covariates is None:
        if past_covariates is not None or future_covariates is not None:
            raise ValueError("pass both past and future covariates, or neither.")
        return past, np.zeros((1, horizon, 0), dtype=np.float32)
    past_covariates = np.asarray(past_covariates, dtype=np.float32)
    future_covariates = np.asarray(future_covariates, dtype=np.float32)
    if future_covariates.shape[0] != horizon:
        raise ValueError(f"future covariates must cover {horizon} days (got {future_covariates.shape[0]}).")
    return np.concatenate([past, past_covariates[-input_window:][None]], axis=2), future_covariates[None]


# =============================================================================================
# PyTorch plumbing (lazy)
# =============================================================================================

def _require_torch():
    """Import and return `torch`, or raise an ImportError that says exactly how to install it."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "LSTM/GRU forecasting needs PyTorch, which is not installed in this Python environment. "
            f"Install the CPU build with:  {TORCH_INSTALL_COMMAND}  "
            f"(or everything Phase 3 needs with:  {PHASE3_REQUIREMENTS_COMMAND}). "
            "No other method is substituted for the neural network."
        ) from exc
    except OSError as exc:  # e.g. a missing Visual C++ runtime DLL on Windows
        raise ImportError(
            f"PyTorch is installed but could not be loaded ({exc}). On Windows this is usually a missing "
            "'Microsoft Visual C++ Redistributable (x64)': install it, restart the terminal, and try again; "
            f"or reinstall PyTorch with:  {TORCH_INSTALL_COMMAND}. No other method is substituted for the neural network."
        ) from exc
    return torch


@contextmanager
def _torch_runtime(torch, n_threads: int):
    """Fixed thread count + deterministic kernels while training; restores the caller's settings."""
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(int(n_threads))
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with torch.random.fork_rng(devices=[]):  # the global RNG state is restored afterwards
            yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.set_num_threads(previous_threads)


def _build_network(torch, cell: str, n_past: int, n_future: int, cfg: NeuralConfig):
    """One recurrent layer + dropout on its last hidden state + a linear head emitting `horizon` values.

    The head also receives the flattened published-in-advance calendar features of the forecast
    days (n_future per day), which is what lets the `calendar` variants know about e.g. Christmas.
    """
    nn = torch.nn

    class _SequenceNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = recurrent(input_size=n_past, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, batch_first=True)
            self.dropout = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(cfg.hidden_size + cfg.horizon * n_future, cfg.horizon)

        def forward(self, past, future):
            output, _ = self.rnn(past)
            state = self.dropout(output[:, -1, :])
            if n_future > 0:
                state = torch.cat([state, future.reshape(future.shape[0], -1)], dim=1)
            return self.head(state)

    return _SequenceNet()


def _train_member(torch, cfg: NeuralConfig, cell: str, seed: int, train, validation, n_past: int, n_future: int):
    """Train ONE ensemble member; return (network with best-epoch weights, history rows, summary)."""
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    model = _build_network(torch, cell, n_past, n_future, cfg)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.HuberLoss(delta=cfg.huber_delta) if cfg.loss == "huber" else torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(int(seed))

    train_past, train_future, train_target = (torch.from_numpy(a) for a in train)
    val_past, val_future, val_target = (torch.from_numpy(a) for a in validation)
    n_train = train_past.shape[0]

    rows: List[Dict[str, Any]] = []
    best_val, best_epoch, best_state, stale = float("inf"), 0, None, 0
    stopped_early = False
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator)   # batch order only; every sample is a training-period window
        running = 0.0
        for start in range(0, n_train, cfg.batch_size):
            batch = order[start:start + cfg.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_past[batch], train_future[batch]), train_target[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()
            running += float(loss.item()) * len(batch)
        train_loss = running / n_train
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_past, val_future), val_target).item())
        if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
            raise FloatingPointError(
                f"training diverged for seed {seed} at epoch {epoch} (train loss {train_loss}, validation loss {val_loss})."
            )
        rows.append({"seed": int(seed), "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                stopped_early = True
                break
    model.load_state_dict(best_state)
    model.eval()
    for row in rows:
        row["is_best_epoch"] = row["epoch"] == best_epoch
    summary = {
        "seed": int(seed),
        "epochs_run": len(rows),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "final_train_loss": float(rows[-1]["train_loss"]),
        "stopped_early": bool(stopped_early),
        "hit_max_epochs": bool(len(rows) == cfg.max_epochs and not stopped_early),
        "fit_seconds": float(time.perf_counter() - started),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return model, rows, summary


# =============================================================================================
# The forecaster
# =============================================================================================

class NeuralSequenceForecaster(BaseForecaster):
    """Direct multi-horizon LSTM / GRU forecaster with a seed ensemble (see the module docstring).

    Parameters
    ----------
    cell       "lstm" or "gru"
    features   "calendar" (past sales + published-in-advance calendar facts) or "univariate"
    config     validated NeuralConfig (use `from_config` to read config.yaml)
    calendar   date-indexed 0/1 event-flag table (`load_event_calendar`); required for "calendar"
               and must cover the training dates AND the forecast dates. Ignored for "univariate".

    `fit(history)` trains one network per seed; `predict(h)` returns the mean of the per-seed
    forecasts (each already clipped at zero) for the h days after the history, h <= horizon.
    """

    def __init__(self, cell: str, features: str, config: NeuralConfig, calendar: Optional[pd.DataFrame] = None) -> None:
        _check_cell_and_features(cell, features)
        if not isinstance(config, NeuralConfig):
            raise TypeError("config must be a NeuralConfig (use NeuralSequenceForecaster.from_config).")
        if features == "calendar" and calendar is None:
            raise ValueError(
                "The 'calendar' variants need the event calendar (pass calendar=load_event_calendar(...)); "
                "use features='univariate' for a model that sees past sales only."
            )
        self.cell = cell
        self.features = features
        self.config = config
        self.calendar = calendar if features == "calendar" else None
        self.name = variant_name(cell, features)
        self._reset()

    @classmethod
    def from_config(
        cls,
        cell: str,
        features: str,
        config: Optional[Mapping[str, Any]] = None,
        calendar: Optional[pd.DataFrame] = None,
    ) -> "NeuralSequenceForecaster":
        """Build a forecaster from config.yaml (`neural` section + backtest horizon)."""
        return cls(cell, features, NeuralConfig.from_config(config), calendar=calendar)

    # ------------------------------------------------------------------ state
    def _reset(self) -> None:
        self._seed_forecasts: Optional[np.ndarray] = None       # (n_seeds, horizon), clipped, original units
        self._history_rows: List[Dict[str, Any]] = []
        self._diagnostics: Dict[str, Any] = {}
        self._history_end: Optional[pd.Timestamp] = None

    @property
    def is_fitted(self) -> bool:
        return self._seed_forecasts is not None

    @property
    def min_history(self) -> int:
        """Shortest history this configuration can be fitted on."""
        return self.config.required_history

    def _require_fitted(self) -> np.ndarray:
        if self._seed_forecasts is None:
            raise RuntimeError(f"{self.name} has not been fitted: call fit(history) first.")
        return self._seed_forecasts

    # ------------------------------------------------------------------ fit
    def fit(self, history: pd.Series) -> "NeuralSequenceForecaster":
        """Train one network per seed on `history` (the data available at the forecast origin)."""
        torch = _require_torch()
        cfg = self.config
        values = check_history(history)
        n_obs = len(values)
        if n_obs < cfg.required_history:
            raise ValueError(
                f"{self.name}: a history of {n_obs} days is too short; this configuration needs at least "
                f"{cfg.required_history} (input window {cfg.input_window} + horizon {cfg.horizon} + validation tail "
                f"{cfg.validation_days} + {MIN_TRAIN_WINDOWS - 1})."
            )
        self._reset()
        started = time.perf_counter()
        history_end = history.index[-1]
        future_index = pd.date_range(history_end + pd.Timedelta(days=1), periods=cfg.horizon, freq="D")

        # Covariates: published-in-advance calendar facts for the training days AND the forecast days.
        if self.features == "calendar":
            past_cov_all = calendar_feature_matrix(history.index, self.calendar)
            future_cov = calendar_feature_matrix(future_index, self.calendar)
        else:
            past_cov_all, future_cov = None, None

        # Chronological inner split; the scaler sees the inner-TRAIN part only.
        train_origins, validation_origins = split_origins(n_obs, cfg.input_window, cfg.horizon, cfg.validation_days)
        inner_train_end = n_obs - cfg.validation_days                  # first position of the validation tail
        scaler = Scaler.fit(values[:inner_train_end])
        scaled = scaler.transform(values)
        train = build_windows(scaled, past_cov_all, train_origins, cfg.input_window, cfg.horizon)
        validation = build_windows(scaled, past_cov_all, validation_origins, cfg.input_window, cfg.horizon)
        past_in, future_in = make_forecast_input(scaled, past_cov_all, future_cov, cfg.input_window, cfg.horizon)
        n_past, n_future = train[0].shape[2], train[1].shape[2]

        member_forecasts, summaries, history_rows = [], [], []
        n_clipped = 0
        with _torch_runtime(torch, cfg.torch_num_threads):
            for seed in cfg.seeds:
                model, rows, summary = _train_member(torch, cfg, self.cell, seed, train, validation, n_past, n_future)
                with torch.no_grad():
                    raw = scaler.inverse(model(torch.from_numpy(past_in), torch.from_numpy(future_in)).numpy()[0])
                n_clipped += int(np.sum(raw < 0))
                member_forecasts.append(np.clip(raw, 0.0, None))
                summaries.append(summary)
                history_rows.extend(rows)

        self._seed_forecasts = np.vstack(member_forecasts)
        self._history_rows = history_rows
        self._history_end = history_end
        self._diagnostics = {
            "model": self.name,
            "cell": self.cell,
            "features": self.features,
            "n_obs": int(n_obs),
            "history_end": str(history_end.date()),
            "input_window": cfg.input_window,
            "horizon": cfg.horizon,
            "validation_days": cfg.validation_days,
            "n_train_windows": int(len(train_origins)),
            "n_validation_windows": int(len(validation_origins)),
            "n_past_features": int(n_past),
            "n_future_features": int(n_future),
            "scaler_mean": scaler.mean,
            "scaler_scale": scaler.scale,
            "scaler_fit_end_date": str(history.index[inner_train_end - 1].date()),
            "seeds": list(cfg.seeds),
            "epochs_run": [s["epochs_run"] for s in summaries],
            "best_epoch": [s["best_epoch"] for s in summaries],
            "best_val_loss": [s["best_val_loss"] for s in summaries],
            "final_train_loss": [s["final_train_loss"] for s in summaries],
            "stopped_early": [s["stopped_early"] for s in summaries],
            "hit_max_epochs": [s["hit_max_epochs"] for s in summaries],
            "member_fit_seconds": [s["fit_seconds"] for s in summaries],
            "n_parameters": summaries[0]["n_parameters"],
            "loss": cfg.loss,
            "n_negative_values_clipped": int(n_clipped),
            "fit_seconds": float(time.perf_counter() - started),
            "torch_version": str(torch.__version__),
        }
        logger.info(
            "[%s] fitted on %d days ending %s: %d seeds, epochs run %s, %.1fs",
            self.name, n_obs, history_end.date(), len(cfg.seeds), self._diagnostics["epochs_run"], self._diagnostics["fit_seconds"],
        )
        return self

    # ------------------------------------------------------------------ predict
    def _check_horizon(self, horizon: int) -> int:
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError(f"horizon must be a positive integer (got {horizon!r}).")
        if horizon > self.config.horizon:
            raise ValueError(
                f"{self.name} is a direct {self.config.horizon}-step model: it can forecast at most "
                f"{self.config.horizon} days ahead (asked for {horizon})."
            )
        return int(horizon)

    def predict(self, horizon: int) -> np.ndarray:
        """Mean over the seeds of the (zero-clipped) forecasts for the `horizon` days after the history."""
        forecasts = self._require_fitted()
        return forecasts[:, : self._check_horizon(horizon)].mean(axis=0)

    def seed_predictions(self, horizon: Optional[int] = None) -> np.ndarray:
        """Per-seed forecasts, shape (n_seeds, horizon): the ensemble members behind `predict`."""
        forecasts = self._require_fitted()
        return forecasts[:, : self._check_horizon(self.config.horizon if horizon is None else horizon)].copy()

    def forecast_index(self, horizon: Optional[int] = None) -> pd.DatetimeIndex:
        """The calendar days that `predict(horizon)` refers to."""
        self._require_fitted()
        steps = self._check_horizon(self.config.horizon if horizon is None else horizon)
        return pd.date_range(self._history_end + pd.Timedelta(days=1), periods=steps, freq="D")

    # ------------------------------------------------------------------ reporting
    def training_history(self) -> pd.DataFrame:
        """Per-epoch losses of every seed: seed, epoch, train_loss, val_loss, is_best_epoch.

        Losses are in standardised units. `train_loss` is measured with dropout active (slightly
        pessimistic); `val_loss` is measured on the inner validation tail with dropout off.
        """
        self._require_fitted()
        return pd.DataFrame(self._history_rows, columns=["seed", "epoch", "train_loss", "val_loss", "is_best_epoch"])

    def fit_diagnostics(self) -> dict:
        """Facts about the most recent fit (JSON-friendly): windows, scaler, epochs, losses, seconds."""
        return dict(self._diagnostics)


# ------------------------------------------------------------------------------------------------ resumable runs
#: Bump when the checkpoint file layout changes; older checkpoints are then ignored (and retrained).
CHECKPOINT_VERSION = 1


def checkpoint_fingerprint(config: Mapping[str, Any], variant_name: str, fold: int, train_values: Sequence[float]) -> str:
    """Hash that decides whether a saved fold may be reused: settings, variant, fold, training data and library versions.

    Any change to the neural settings, the horizon, the training values of the fold, or the numpy / torch versions gives a
    different fingerprint, so a stale checkpoint is never restored into a run that would have produced something else.
    """
    import hashlib

    from src.utils.provenance import canonical_json

    try:
        import torch

        torch_version = str(torch.__version__)
    except ImportError:  # pragma: no cover - the notebook requires torch anyway
        torch_version = "none"
    values = np.ascontiguousarray(np.asarray(train_values, dtype="<f8"))
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "neural": {k: v for k, v in dict(config["neural"]).items() if k != "variants"},
        "horizon": int(config["backtest"]["horizon"]),
        "variant": str(variant_name),
        "fold": int(fold),
        "n_train": int(values.size),
        "train_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "numpy": np.__version__,
        "torch": torch_version,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _checkpoint_path(directory, variant_name: str, fold: int):
    from pathlib import Path
    import re

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant_name).strip("_")
    return Path(directory) / f"{safe}_fold{int(fold)}.pkl"


def save_checkpoint(directory, variant_name: str, fold: int, fingerprint: str, result: Any, model: Any, seconds: float):
    """Write one finished (variant, fold) so an interrupted run can resume. The write is atomic (temp file, then rename)."""
    import pickle
    from pathlib import Path

    path = _checkpoint_path(directory, variant_name, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"version": CHECKPOINT_VERSION, "fingerprint": fingerprint, "variant": variant_name, "fold": int(fold),
              "result": result, "model": model, "seconds": float(seconds)}
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return path


def load_checkpoint(directory, variant_name: str, fold: int, fingerprint: str) -> Optional[Dict[str, Any]]:
    """The saved record of (variant, fold) if it exists AND its fingerprint matches; otherwise None (retrain).

    Checkpoints are pickles written by `save_checkpoint` into the project's own git-ignored artifacts folder. Only restore
    checkpoints you created yourself: unpickling a file from an untrusted source can run arbitrary code.
    """
    import pickle

    path = _checkpoint_path(directory, variant_name, fold)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            record = pickle.load(handle)
    except Exception:  # truncated, corrupt or from an incompatible version: retrain instead of failing
        return None
    if not isinstance(record, dict) or record.get("version") != CHECKPOINT_VERSION or record.get("fingerprint") != fingerprint:
        return None
    return record
