"""Residual diagnostics and classical-model validation (Phase 2 Step 6).

Two distinct kinds of "error" exist in this project and this module is
built specifically not to conflate them:

- **Training (in-sample) residuals**: a fitted model's one-step-ahead
  prediction error on the *same* data it was fit on -- i.e. a backtest
  fold's training window. This is what "residual diagnostics" means
  below. These come exclusively from each forecaster's own
  `fit_diagnostics()["resid"]` / `["resid_diagnostic"]`
  (`src/models/arima.py`, `sarima.py`, `holt_winters.py`), which is
  computed once, inside `fit()`, directly from the fitted result
  object's own `.resid` attribute -- nothing in this module recomputes
  a residual from scratch or from a forecast error.
- **Out-of-sample forecast error**: actual-minus-predicted on a fold's
  28-day held-out *test* window. That is `src.evaluation.metrics` /
  `src.evaluation.backtesting`'s job (Phase 2 Steps 2 and 5) and is
  already computed. This module never reads a fold's `actuals` /
  `predictions` to build something it calls a "residual" -- doing so
  would silently relabel out-of-sample forecast error as in-sample fit,
  which Step 6 explicitly prohibits.

Every function here operates on development-fold data only (the same
five chronological folds Steps 2-5 already use). Nothing in this module
accepts or reads the final holdout period.

Not every model in this project overrides `fit_diagnostics()`
(`SeasonalNaive` returns the `BaseForecaster` default, `{}`) -- folds
with no `resid` key are skipped, not treated as an error, since
residual diagnostics are inherently specific to models that produce a
fitted in-sample residual. A fold that failed outright
(`BacktestFoldResult.success is False`) is also skipped: there is no
fitted model, and therefore no residual, to diagnose. A fold whose
model fit but did **not converge** (e.g. SARIMA fold 3 -- see
`notebooks/04_classical_models.ipynb`, section 11.3) is *not* skipped:
its residuals are still computed and reported, with
`diagnostics["converged"] is False` carried through so every table and
plot built from it is labeled, and every written interpretation of it
stays cautious -- never silently dropped, never presented as if nothing
were wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.evaluation.backtesting import BacktestFoldResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Weekly seasonality for this project's daily retail data -- the same
#: period used throughout (SeasonalNaive, SARIMA, Holt-Winters).
SEASONAL_PERIOD = 7

#: Supplementary Ljung-Box lags always reported alongside the primary
#: lag: one week, three weeks, and this project's own 28-day forecast
#: horizon (four weeks) -- weekly multiples, so a reader can see whether
#: any remaining autocorrelation specifically lines up with the weekly
#: cycle a SARIMA/Holt-Winters model is supposed to have absorbed.
SUPPLEMENTARY_LJUNG_BOX_LAGS: Tuple[int, ...] = (7, 21, 28)

#: Default significance level for both the Ljung-Box verdict and the
#: notable-lag reading of the residual ACF, matching this project's use
#: of alpha=0.05 everywhere else (ADF test, ACF/PACF confidence bands).
DEFAULT_ALPHA = 0.05

#: Reused everywhere a residual-diagnostic result is reported, so the
#: "one diagnostic alone doesn't prove adequacy" caveat Step 6 requires
#: is stated identically (not reworded, diluted, or dropped) wherever
#: this module's output reaches a table, plot caption, or printed
#: summary.
MODEL_ADEQUACY_CAVEAT = (
    "No single diagnostic here -- a residual plot, an ACF, a histogram, "
    "or one Ljung-Box lag -- proves a model is adequate on its own. A "
    "model can pass every in-sample residual check and still forecast "
    "poorly out-of-sample (and vice versa); this project's primary "
    "evidence for model quality remains the out-of-sample backtest "
    "MAE/RMSE from Phase 2 Steps 2 and 5, not the diagnostics below."
)


def select_ljung_box_lag(n_obs: int, seasonal_period: int = SEASONAL_PERIOD) -> int:
    """Primary Ljung-Box lag via Hyndman & Athanasopoulos' rule of thumb
    for seasonal data (*Forecasting: Principles and Practice*, 3rd ed.,
    s9.5 "Residual diagnostics" -- already cited in this project's
    `reports/phase1/references.md`, entry 10, as the general walk-
    forward/backtesting methodology reference): `min(2 * m, T / 5)`,
    where `m` is the seasonal period and `T` the number of residuals
    available. The cap at `T / 5` keeps the test meaningful on the
    smallest development fold, where `2 * 7 = 14` could otherwise be a
    large fraction of the sample.
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs}.")
    candidate = min(2 * seasonal_period, n_obs // 5)
    return max(candidate, 1)


@dataclass
class LjungBoxResult:
    """Ljung-Box test of residual autocorrelation at one or more lags.

    **Null hypothesis (H0)**: the residuals are independently
    distributed (no autocorrelation) up to the tested lag -- i.e. the
    fitted model has already captured all the *linear* autocorrelation
    structure in the series through that lag.

    **Interpretation**: a p-value below `alpha` rejects H0 -- evidence
    that autocorrelation remains in the residuals, i.e. the model has
    *not* fully captured the series' structure at that lag. A p-value at
    or above `alpha` fails to reject H0 -- consistent with (but not
    proof of) residual white noise at that lag. This test, on its own,
    never proves a model is adequate; see `MODEL_ADEQUACY_CAVEAT`.

    `model_df` (passed straight through to
    `statsmodels.stats.diagnostic.acorr_ljungbox`) is the number of
    fitted parameters to subtract from the test's chi-square degrees of
    freedom at each lag (df = lag - model_df). At a lag at or below
    `model_df`, the correction leaves no degrees of freedom to test
    with, and `lb_pvalue` comes back `NaN` -- not computable, not "fails
    to reject" -- `reject_h0` is `None` for those rows rather than a
    misleading `False`. This genuinely happens in this project: every
    model here has `model_df` (8-11 parameters) larger than the 7-day
    weekly supplementary lag, so the lag-7 row is always `NaN` by
    construction, not a bug -- `summary()` says so explicitly.
    """

    table: pd.DataFrame  # index=lag (int) -- columns: lb_stat, lb_pvalue, reject_h0
    model_df: int
    n_obs: int
    alpha: float = DEFAULT_ALPHA

    def untestable_lags(self) -> List[int]:
        """Lags where `lag <= model_df`, so no valid p-value exists."""
        return [lag for lag in self.table.index if lag <= self.model_df]

    def summary(self, label: str = "residuals") -> str:
        lines = [
            f"Ljung-Box test on {label} (H0: independently distributed / "
            f"no autocorrelation up to lag L; n_obs={self.n_obs}, "
            f"model_df={self.model_df}):"
        ]
        for lag, row in self.table.iterrows():
            if lag <= self.model_df:
                lines.append(
                    f"  lag {lag:>3}: lb_stat={row['lb_stat']:.3f}  "
                    f"p-value not computable (lag <= model_df={self.model_df}, "
                    "0 degrees of freedom after the parameter correction)"
                )
                continue
            verdict = (
                f"p < {self.alpha} -> reject H0 -> autocorrelation remains"
                if row["lb_pvalue"] < self.alpha
                else f"p >= {self.alpha} -> fail to reject H0 -> no evidence of remaining autocorrelation at this lag"
            )
            lines.append(
                f"  lag {lag:>3}: lb_stat={row['lb_stat']:8.3f}  "
                f"p-value={row['lb_pvalue']:.4f}  {verdict}"
            )
        return "\n".join(lines)


def ljung_box_test(
    resid: np.ndarray,
    lags: Sequence[int],
    model_df: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> LjungBoxResult:
    """Run the Ljung-Box test on `resid` at every lag in `lags` in one call.

    `resid` should already be the series the diagnostic is meant to run
    on (e.g. burn-in-trimmed training residuals,
    `fit_diagnostics()["resid_diagnostic"]`) -- this function has no
    opinion on which residual series it's given, same division of
    responsibility as `src.analysis.diagnostics.adf_test`.
    """
    resid = np.asarray(resid, dtype=float)
    resid = resid[~np.isnan(resid)]
    lags = sorted(set(int(l) for l in lags))
    if resid.size == 0:
        raise ValueError("resid is empty after dropping NaNs -- nothing to test.")
    if max(lags) >= resid.size:
        raise ValueError(
            f"Largest lag {max(lags)} must be smaller than the number of "
            f"residuals ({resid.size})."
        )

    raw = acorr_ljungbox(resid, lags=lags, model_df=model_df, return_df=True)
    raw = raw.copy()
    raw["reject_h0"] = np.where(
        raw.index <= model_df, None, raw["lb_pvalue"] < alpha
    )
    return LjungBoxResult(table=raw, model_df=model_df, n_obs=int(resid.size), alpha=alpha)


@dataclass
class FoldResidualDiagnostics:
    """Everything Step 6 asks be reported for one (model, fold)'s
    training residuals, bundled together because every table/plot in
    section 11 of the notebook reads them together.

    `dates`/`resid` are the *full* training-residual series (length
    `n_obs_fit`); `dates_diagnostic`/`resid_diagnostic` have the first
    `loglikelihood_burn` observations dropped (0 for Holt-Winters -- see
    `src/models/holt_winters.py`) and are what `ljung_box`, the residual
    ACF, and the histogram are actually computed on. Both are kept so
    the trim is visible, not silent.
    """

    model_name: str
    fold: int
    origin_date: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    n_obs_fit: int
    n_params: int
    nobs: int
    df_resid: int
    loglikelihood_burn: int
    converged: Optional[bool]
    dates: pd.DatetimeIndex
    resid: np.ndarray
    dates_diagnostic: pd.DatetimeIndex
    resid_diagnostic: np.ndarray
    ljung_box: LjungBoxResult
    notes: List[str] = field(default_factory=list)


def _reconstruct_training_dates(origin_date: pd.Timestamp, n_obs_fit: int) -> pd.DatetimeIndex:
    """The exact daily date index of a fold's training window.

    `BacktestFoldResult` records `origin_date` (the training window's
    *last* date -- `train.index[-1]`, set in
    `src.evaluation.backtesting.run_backtest`) and each forecaster's
    `fit_diagnostics()` records `n_obs_fit` (`len(history)`). Since the
    development series is daily and contiguous (`dev_series_d =
    dev_series.asfreq("D")`, Phase 2 Step 5), those two numbers alone
    are enough to reconstruct the training window's date index exactly,
    without threading the original series itself through this module.
    """
    return pd.date_range(end=origin_date, periods=n_obs_fit, freq="D")


def compute_fold_residual_diagnostics(
    fold_result: BacktestFoldResult,
    primary_lag: Optional[int] = None,
    supplementary_lags: Sequence[int] = SUPPLEMENTARY_LJUNG_BOX_LAGS,
    seasonal_period: int = SEASONAL_PERIOD,
    alpha: float = DEFAULT_ALPHA,
) -> Optional[FoldResidualDiagnostics]:
    """Build one fold's `FoldResidualDiagnostics`, or `None` if this fold
    has nothing to diagnose (failed fit, or a model with no residual
    diagnostics -- e.g. SeasonalNaive).

    `primary_lag` defaults to `select_ljung_box_lag()` evaluated on the
    actual post-burn-in diagnostic sample size, so the lag is justified
    by -- and always valid for -- the data it's tested against.
    """
    diag = fold_result.diagnostics or {}
    if not fold_result.success:
        logger.info(
            "[%s] fold %d: skipped (fold failed, no fitted model to diagnose)",
            fold_result.model_name,
            fold_result.fold,
        )
        return None
    if "resid_diagnostic" not in diag or diag["resid_diagnostic"] is None:
        logger.info(
            "[%s] fold %d: skipped (no residual diagnostics available from this model)",
            fold_result.model_name,
            fold_result.fold,
        )
        return None

    resid_full = np.asarray(diag["resid"], dtype=float)
    resid_diag = np.asarray(diag["resid_diagnostic"], dtype=float)
    burn = int(diag.get("loglikelihood_burn", 0) or 0)
    n_obs_fit = int(diag["n_obs_fit"])
    n_params = int(diag["n_params"])
    nobs = int(diag.get("nobs", n_obs_fit))
    df_resid = diag.get("df_resid")
    df_resid = int(df_resid) if df_resid is not None else int(len(resid_diag) - n_params)
    converged = diag.get("converged")

    dates_full = _reconstruct_training_dates(fold_result.origin_date, n_obs_fit)
    dates_diag = dates_full[burn:]

    lag = primary_lag if primary_lag is not None else select_ljung_box_lag(
        len(resid_diag), seasonal_period=seasonal_period
    )
    all_lags = sorted(set([lag, *supplementary_lags]))
    # A supplementary lag can legitimately exceed what this fold's
    # (post-burn-in) sample supports -- drop it rather than crash the
    # whole fold's diagnostics over one oversized lag, and say so.
    usable_lags = [l for l in all_lags if l < len(resid_diag)]
    dropped_lags = sorted(set(all_lags) - set(usable_lags))

    ljung_box = ljung_box_test(resid_diag, usable_lags, model_df=n_params, alpha=alpha)

    notes: List[str] = []
    if converged is False:
        notes.append(
            "Fit did NOT converge (optimizer's own converged flag is False). "
            "Residuals and all diagnostics below are reported as computed, "
            "but should be interpreted cautiously -- see the dedicated "
            "convergence investigation (notebook section 11.3)."
        )
    if burn:
        notes.append(
            f"First {burn} training residual(s) excluded from this fold's "
            "diagnostic series (loglikelihood_burn, diffuse Kalman "
            "initialization artifact -- see src/models/arima.py)."
        )
    if dropped_lags:
        notes.append(
            f"Lag(s) {dropped_lags} skipped for this fold: not smaller than "
            f"the {len(resid_diag)}-observation diagnostic sample."
        )

    return FoldResidualDiagnostics(
        model_name=fold_result.model_name,
        fold=fold_result.fold,
        origin_date=fold_result.origin_date,
        forecast_start=fold_result.forecast_start,
        forecast_end=fold_result.forecast_end,
        n_obs_fit=n_obs_fit,
        n_params=n_params,
        nobs=nobs,
        df_resid=df_resid,
        loglikelihood_burn=burn,
        converged=converged,
        dates=dates_full,
        resid=resid_full,
        dates_diagnostic=dates_diag,
        resid_diagnostic=resid_diag,
        ljung_box=ljung_box,
        notes=notes,
    )


def build_fold_diagnostics(
    fold_results: Sequence[BacktestFoldResult],
    primary_lag: Optional[int] = None,
    supplementary_lags: Sequence[int] = SUPPLEMENTARY_LJUNG_BOX_LAGS,
    seasonal_period: int = SEASONAL_PERIOD,
    alpha: float = DEFAULT_ALPHA,
) -> List[FoldResidualDiagnostics]:
    """`compute_fold_residual_diagnostics` over every fold in
    `fold_results`, skipping (not raising on) folds with nothing to
    diagnose. Intended to be called directly on this project's existing
    Step 5 backtest result lists (`arima_results`, `sarima_results`,
    `hw_add_results`, `hw_none_results`) -- every one of them is already
    development-fold-only, so there is nothing for this function to
    filter by date; it trusts its caller never passes it a holdout-
    period result, the same contract `run_backtest()` itself relies on.
    """
    out = []
    for fr in fold_results:
        d = compute_fold_residual_diagnostics(
            fr,
            primary_lag=primary_lag,
            supplementary_lags=supplementary_lags,
            seasonal_period=seasonal_period,
            alpha=alpha,
        )
        if d is not None:
            out.append(d)
    return out


def residual_diagnostics_table(diagnostics: Sequence[FoldResidualDiagnostics]) -> pd.DataFrame:
    """Long-format table, one row per (model, fold, lag) -- the exportable
    record behind every number in section 11's Ljung-Box discussion.
    """
    rows = []
    for d in diagnostics:
        for lag, row in d.ljung_box.table.iterrows():
            rows.append(
                {
                    "model": d.model_name,
                    "fold": d.fold,
                    "forecast_start": d.forecast_start.date(),
                    "forecast_end": d.forecast_end.date(),
                    "converged": d.converged,
                    "n_obs_fit": d.n_obs_fit,
                    "n_params": d.n_params,
                    "df_resid": d.df_resid,
                    "loglikelihood_burn": d.loglikelihood_burn,
                    "lag": lag,
                    "lb_stat": row["lb_stat"],
                    "lb_pvalue": row["lb_pvalue"],
                    "reject_h0": row["reject_h0"],
                }
            )
    return pd.DataFrame(rows)


def fold_summary_table(diagnostics: Sequence[FoldResidualDiagnostics]) -> pd.DataFrame:
    """One row per (model, fold): the sample-size/DOF bookkeeping Step 6
    explicitly asks to have reported, independent of any particular lag.
    """
    rows = []
    for d in diagnostics:
        rows.append(
            {
                "model": d.model_name,
                "fold": d.fold,
                "forecast_start": d.forecast_start.date(),
                "forecast_end": d.forecast_end.date(),
                "converged": d.converged,
                "n_obs_fit": d.n_obs_fit,
                "loglikelihood_burn": d.loglikelihood_burn,
                "n_obs_diagnostic": len(d.resid_diagnostic),
                "n_params": d.n_params,
                "df_resid": d.df_resid,
                "notes": "; ".join(d.notes) if d.notes else "",
            }
        )
    return pd.DataFrame(rows)


def plot_residual_diagnostics(
    diag: FoldResidualDiagnostics,
    acf_nlags: int = 28,
    alpha: float = DEFAULT_ALPHA,
):
    """Three-panel residual diagnostic figure for one (model, fold):
    residuals over time, residual ACF (reusing
    `src.analysis.diagnostics.compute_acf`, the same ACF machinery Step
    4 already uses), and a residual histogram. Computed on
    `diag.resid_diagnostic` (the burn-in-trimmed series), never
    `diag.resid`, so the figure matches what `diag.ljung_box` was
    actually tested on.
    """
    import matplotlib.pyplot as plt

    from src.analysis.diagnostics import compute_acf

    resid = diag.resid_diagnostic
    nlags = min(acf_nlags, len(resid) - 1)
    acf_values, acf_confint = compute_acf(pd.Series(resid), nlags=nlags, alpha=alpha)

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_time = fig.add_subplot(gs[0, :])
    ax_acf = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    # Panel 1: residuals over time.
    ax_time.plot(diag.dates_diagnostic, resid, color="#4C4C4C", linewidth=1)
    ax_time.axhline(0, color="black", linewidth=0.8, linestyle=":")
    status = "" if diag.converged is not False else "  [fit did NOT converge -- interpret cautiously]"
    ax_time.set_title(
        f"{diag.model_name} -- fold {diag.fold} training residuals "
        f"({diag.dates_diagnostic[0].date()} to {diag.dates_diagnostic[-1].date()}){status}",
        color="#B00000" if diag.converged is False else "black",
    )
    ax_time.set_xlabel("Date")
    ax_time.set_ylabel("Residual (units)")
    ax_time.tick_params(axis="x", rotation=30)

    # Panel 2: residual ACF.
    lags = np.arange(nlags + 1)
    highlight = {diag.ljung_box.table.index.min(), *SUPPLEMENTARY_LJUNG_BOX_LAGS} & set(lags)
    colors = ["#D55E00" if lag in highlight else "#4C72B0" for lag in lags]
    ax_acf.vlines(lags, 0, acf_values, color=colors, linewidth=1.3)
    ax_acf.scatter(lags, acf_values, color=colors, s=10, zorder=3)
    ax_acf.axhline(0, color="black", linewidth=0.8)
    band_lower = acf_confint[1:, 0] - acf_values[1:]
    band_upper = acf_confint[1:, 1] - acf_values[1:]
    ax_acf.fill_between(lags[1:], band_lower, band_upper, color="#4C72B0", alpha=0.15)
    ax_acf.set_title("Residual ACF")
    ax_acf.set_xlabel("Lag (days)")
    ax_acf.set_ylabel("ACF")

    # Panel 3: residual distribution.
    ax_hist.hist(resid, bins=30, color="#8C8C8C", edgecolor="white")
    ax_hist.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax_hist.axvline(float(np.mean(resid)), color="#D55E00", linewidth=1.2, label=f"mean={np.mean(resid):.1f}")
    ax_hist.set_title("Residual distribution")
    ax_hist.set_xlabel("Residual (units)")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_residual_diagnostics_grid(
    diagnostics: Sequence[FoldResidualDiagnostics],
    acf_nlags: int = 28,
    alpha: float = DEFAULT_ALPHA,
):
    """One compact row per fold (residuals-over-time / ACF / histogram)
    for every fold of a single model, side by side.

    This is the "for each successful fit: plot residuals over time; plot
    the residual ACF; plot a residual distribution" requirement applied
    to every fold of one model at once, without a full-page
    `plot_residual_diagnostics()` figure per fold -- `diagnostics` is
    typically all 5 development folds of one model
    (`build_fold_diagnostics(arima_results)`, say). A fold whose fit did
    not converge is still plotted (never dropped), labeled in its row
    title and drawn in red rather than silently blended in with the
    rest.
    """
    import matplotlib.pyplot as plt

    from src.analysis.diagnostics import compute_acf

    if not diagnostics:
        raise ValueError("diagnostics is empty -- nothing to plot.")

    n = len(diagnostics)
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.6 * n), squeeze=False)

    for i, d in enumerate(diagnostics):
        resid = d.resid_diagnostic
        nlags = min(acf_nlags, len(resid) - 1)
        acf_values, acf_confint = compute_acf(pd.Series(resid), nlags=nlags, alpha=alpha)
        lags = np.arange(nlags + 1)
        flag = " [NOT converged]" if d.converged is False else ""
        line_color = "#B00000" if d.converged is False else "#4C4C4C"

        ax_t, ax_a, ax_h = axes[i]
        ax_t.plot(d.dates_diagnostic, resid, color=line_color, linewidth=0.8)
        ax_t.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax_t.set_title(f"fold {d.fold}{flag} -- residuals", fontsize=9)
        ax_t.tick_params(labelsize=7, axis="x", rotation=30)

        ax_a.vlines(lags, 0, acf_values, color="#4C72B0", linewidth=1.0)
        ax_a.axhline(0, color="black", linewidth=0.6)
        band_lower = acf_confint[1:, 0] - acf_values[1:]
        band_upper = acf_confint[1:, 1] - acf_values[1:]
        ax_a.fill_between(lags[1:], band_lower, band_upper, color="#4C72B0", alpha=0.15)
        ax_a.set_title(f"fold {d.fold} -- ACF", fontsize=9)
        ax_a.tick_params(labelsize=7)

        ax_h.hist(resid, bins=20, color="#8C8C8C", edgecolor="white")
        ax_h.axvline(0, color="black", linewidth=0.6, linestyle=":")
        ax_h.set_title(f"fold {d.fold} -- distribution", fontsize=9)
        ax_h.tick_params(labelsize=7)

    model_name = diagnostics[0].model_name
    fig.suptitle(f"{model_name} -- training residual diagnostics, all folds", y=1.004)
    fig.tight_layout()
    return fig
