"""Stationarity and Box-Jenkins order-identification diagnostics for Phase 2.

adf_test() wraps statsmodels' Augmented Dickey-Fuller test behind this
project's own small, explicit result type (ADFResult) rather than
statsmodels' raw positional tuple, so callers don't need to remember
tuple order.

compute_acf_pacf() / run_auto_arima_cross_check() (Step 4) are *order
identification* diagnostics: they read the correlation structure of the
(differenced) raw series to suggest candidate ARIMA/SARIMA orders. They
are deliberately candidate-generating only -- nothing in this module
picks a final (p, d, q) for you, because the project's own instructions
are explicit that final order selection is validated later via model
fitting, information criteria, and backtesting. *Residual* diagnostics
(ACF/PACF of a fitted model's residuals, Ljung-Box, etc.) are a separate,
later concern, once a Phase 2 model that actually produces residuals
exists -- this module reuses the same ACF/PACF machinery for that when
it arrives, rather than duplicating it.

This module has no opinion on *which* period of a series gets tested --
that's the caller's responsibility (e.g. notebooks/04_classical_models.ipynb
passes only the development-period series, via src.data.splitter's
already-tested final-holdout split, never the holdout itself).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, pacf

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Below this many non-null observations, adfuller's own lag-selection
#: machinery can fail with an opaque error rather than a clear one -- this
#: is a usability guard, not a statistical minimum-sample-size claim.
MIN_OBS_FOR_ADF = 20

#: Default max lag for ACF/PACF on daily retail data -- Phase 2 Step 4
#: asks for "at least through 60 lags", enough to see several weekly
#: cycles (60 / 7 ~= 8.6 weeks) without the plot becoming unreadable.
DEFAULT_ACF_PACF_NLAGS = 60

#: Daily retail data: 7 = weekly seasonality (the project's own baseline,
#: Seasonal Naive, already assumes this same period).
SEASONAL_PERIOD = 7

#: Lags Phase 2 Step 4 explicitly asks to pay attention to: every lag
#: from 1-7, then the weekly multiples out to 56 (8 weeks).
LAGS_OF_INTEREST: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 56)

#: statsmodels' newer, numerically-stable PACF estimator (bounded in
#: [-1, 1]; the older default "yw"/"ywadjusted" can produce out-of-range
#: values and is slated for replacement) -- used as this module's default
#: rather than leaving callers to discover the warning themselves.
DEFAULT_PACF_METHOD = "ywm"


@dataclass
class ADFResult:
    """Augmented Dickey-Fuller test result, plus this project's own interpretation of it.

    H0 (null hypothesis): the series has a unit root (is non-stationary).
    A p-value below `alpha` rejects H0 -- evidence of stationarity. A
    p-value at or above `alpha` fails to reject H0 -- insufficient
    evidence of stationarity. Either way this is a statistical test, not
    proof: rejecting H0 doesn't prove stationarity absolutely, and
    failing to reject it doesn't prove non-stationarity either.
    """

    statistic: float
    p_value: float
    n_lags: int
    n_obs: int
    critical_values: Dict[str, float]
    alpha: float = 0.05

    @property
    def is_stationary(self) -> bool:
        """True if p_value < alpha (reject H0 -- evidence of stationarity)."""
        return self.p_value < self.alpha

    def summary(self, label: str = "series") -> str:
        crit = ", ".join(f"{k}: {v:.4f}" for k, v in self.critical_values.items())
        verdict = (
            f"p-value < {self.alpha} -> reject H0 -> evidence of stationarity"
            if self.is_stationary
            else f"p-value >= {self.alpha} -> fail to reject H0 -> insufficient evidence of stationarity"
        )
        return (
            f"ADF test on {label}:\n"
            f"  statistic: {self.statistic:.4f}\n"
            f"  p-value:   {self.p_value:.6f}\n"
            f"  n_lags:    {self.n_lags}\n"
            f"  n_obs:     {self.n_obs}\n"
            f"  critical values: {crit}\n"
            f"  {verdict}"
        )


def adf_test(series: pd.Series, alpha: float = 0.05) -> ADFResult:
    """Run the Augmented Dickey-Fuller test on `series` and wrap the result.

    Lag order is selected automatically (AIC), matching statsmodels'
    default and avoiding a hand-picked lag count this project has no
    principled value for yet. Raises ValueError if `series` has fewer
    than MIN_OBS_FOR_ADF non-null observations.
    """
    clean = pd.Series(series).dropna()
    if len(clean) < MIN_OBS_FOR_ADF:
        raise ValueError(
            f"Need at least {MIN_OBS_FOR_ADF} observations to run the ADF test, "
            f"got {len(clean)}."
        )

    # result_object=False pins today's documented tuple-return shape
    # explicitly (a future statsmodels release changes the default) so
    # this unpacking stays correct regardless of which version runs it.
    statistic, p_value, n_lags, n_obs, critical_values, _ = adfuller(
        clean.to_numpy(), autolag="AIC", result_object=False
    )

    result = ADFResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_lags=int(n_lags),
        n_obs=int(n_obs),
        critical_values={k: float(v) for k, v in critical_values.items()},
        alpha=alpha,
    )
    logger.info("\n%s", result.summary())
    return result


def first_difference(series: pd.Series) -> pd.Series:
    """First-differenced series: series.diff().dropna().

    Never applied more than once automatically -- a second differencing
    pass, if ever needed, is a separate, deliberate decision made after
    reviewing this diagnostic, not something this function chains into.
    """
    return series.diff().dropna()


def plot_stationarity_diagnostic(
    original: pd.Series,
    differenced: "pd.Series | None" = None,
    original_label: str = "Original series",
    differenced_label: str = "First-differenced series",
):
    """One figure: the original series, and -- only if provided -- the
    first-differenced series stacked below it. `differenced` is omitted
    entirely (one panel only) when differencing wasn't needed.
    """
    import matplotlib.pyplot as plt

    n_panels = 2 if differenced is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.8 * n_panels))
    axes = [axes] if n_panels == 1 else list(axes)

    axes[0].plot(original.index, original.to_numpy(), color="black", linewidth=1)
    axes[0].set_title(original_label)
    axes[0].set_ylabel("Sales (units)")

    if differenced is not None:
        axes[1].plot(differenced.index, differenced.to_numpy(), color="#8C8C8C", linewidth=1)
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle=":")
        axes[1].set_title(differenced_label)
        axes[1].set_ylabel("Δ Sales (units)")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    return fig


# =============================================================================
# Phase 2 Step 4 -- ACF / PACF order-identification diagnostics
# =============================================================================


def _validate_series_for_correlogram(series: pd.Series, nlags: int) -> pd.Series:
    """Shared input guard for compute_acf/compute_pacf.

    Mirrors MIN_OBS_FOR_ADF's role for adf_test(): a clear, early
    ValueError instead of letting statsmodels fail deep inside its own
    lag machinery with a less legible message.
    """
    clean = pd.Series(series).dropna()
    if nlags < 1:
        raise ValueError(f"nlags must be at least 1, got {nlags}.")
    if len(clean) <= nlags:
        raise ValueError(
            f"Need more observations than nlags to compute a correlogram: "
            f"got {len(clean)} observations for nlags={nlags}."
        )
    return clean


@dataclass
class ACFPACFResult:
    """ACF and PACF of one series out to `nlags`, bundled together because
    Step 4's diagnostic reading always uses them side by side.

    `lags` runs 0..nlags inclusive, matching `acf_values`/`pacf_values`
    (both length nlags + 1; index 0 is the trivial lag-0 value of 1.0).
    `acf_confint`/`pacf_confint` (shape (nlags + 1, 2)) are statsmodels'
    own per-lag confidence intervals for the *estimate* at that lag --
    `notable_acf_lags`/`notable_pacf_lags` turn each row into the
    standard "does this bar poke outside the significance band" verdict
    (recentre the interval at zero and check whether it still excludes
    zero, equivalent to statsmodels' own plot_acf/plot_pacf shading)
    rather than leaving every caller to rederive that test by hand.

    This is a candidate-generating diagnostic only: nothing on this class
    picks a final (p, q) for you.
    """

    lags: np.ndarray
    acf_values: np.ndarray
    acf_confint: np.ndarray
    pacf_values: np.ndarray
    pacf_confint: np.ndarray
    nlags: int
    alpha: float = 0.05

    def _check_lags(self, lags_of_interest: Sequence[int]) -> None:
        too_far = [lag for lag in lags_of_interest if lag < 1 or lag > self.nlags]
        if too_far:
            raise ValueError(
                f"Lag(s) {too_far} fall outside the computed range [1, {self.nlags}]. "
                "Recompute compute_acf_pacf() with a larger nlags to inspect them."
            )

    @staticmethod
    def _is_notable(confint_row: np.ndarray) -> bool:
        lower, upper = confint_row
        return not (lower <= 0.0 <= upper)

    def notable_acf_lags(self, lags_of_interest: Sequence[int] = LAGS_OF_INTEREST) -> Dict[int, bool]:
        """{lag: True if the ACF bar at that lag sits outside the band}."""
        self._check_lags(lags_of_interest)
        return {lag: self._is_notable(self.acf_confint[lag]) for lag in lags_of_interest}

    def notable_pacf_lags(self, lags_of_interest: Sequence[int] = LAGS_OF_INTEREST) -> Dict[int, bool]:
        """{lag: True if the PACF bar at that lag sits outside the band}."""
        self._check_lags(lags_of_interest)
        return {lag: self._is_notable(self.pacf_confint[lag]) for lag in lags_of_interest}

    def notable_lags_table(self, lags_of_interest: Sequence[int] = LAGS_OF_INTEREST) -> pd.DataFrame:
        """One row per lag in `lags_of_interest` with the real, computed
        ACF/PACF values and their notability verdicts -- the evidence
        this project's written interpretation is read off of, rather
        than a hard-coded conclusion.
        """
        self._check_lags(lags_of_interest)
        rows = [
            {
                "lag": lag,
                "acf": float(self.acf_values[lag]),
                "acf_notable": self._is_notable(self.acf_confint[lag]),
                "pacf": float(self.pacf_values[lag]),
                "pacf_notable": self._is_notable(self.pacf_confint[lag]),
            }
            for lag in lags_of_interest
        ]
        return pd.DataFrame(rows)


def compute_acf(
    series: pd.Series, nlags: int = DEFAULT_ACF_PACF_NLAGS, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ACF of `series` out to `nlags`, with its significance band.

    Returns (acf_values, confint), both length nlags + 1 (lag 0 first).
    `result_object=False` pins today's documented tuple-return shape
    explicitly, the same guard already used for adfuller() in adf_test()
    -- a future statsmodels release changes the default return type.
    """
    clean = _validate_series_for_correlogram(series, nlags)
    acf_values, confint = acf(
        clean.to_numpy(), nlags=nlags, alpha=alpha, fft=True, result_object=False
    )
    return acf_values, confint


def compute_pacf(
    series: pd.Series,
    nlags: int = DEFAULT_ACF_PACF_NLAGS,
    alpha: float = 0.05,
    method: str = DEFAULT_PACF_METHOD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample PACF of `series` out to `nlags`, with its significance band.

    Same shape/convention as compute_acf(). `method="ywm"` (this module's
    default -- see DEFAULT_PACF_METHOD) is statsmodels' numerically
    stable Yule-Walker estimator, bounded in [-1, 1].
    """
    clean = _validate_series_for_correlogram(series, nlags)
    pacf_values, confint = pacf(
        clean.to_numpy(), nlags=nlags, alpha=alpha, method=method, result_object=False
    )
    return pacf_values, confint


def compute_acf_pacf(
    series: pd.Series,
    nlags: int = DEFAULT_ACF_PACF_NLAGS,
    alpha: float = 0.05,
    pacf_method: str = DEFAULT_PACF_METHOD,
) -> ACFPACFResult:
    """Compute ACF and PACF together and bundle them into one ACFPACFResult.

    Thin convenience wrapper over compute_acf() + compute_pacf() -- no new
    statistics, just the pairing Step 4's diagnostic reading needs.
    """
    acf_values, acf_confint = compute_acf(series, nlags=nlags, alpha=alpha)
    pacf_values, pacf_confint = compute_pacf(series, nlags=nlags, alpha=alpha, method=pacf_method)
    return ACFPACFResult(
        lags=np.arange(nlags + 1),
        acf_values=acf_values,
        acf_confint=acf_confint,
        pacf_values=pacf_values,
        pacf_confint=pacf_confint,
        nlags=nlags,
        alpha=alpha,
    )


def _plot_correlogram_panel(ax, lags, values, confint, highlight_lags, ylabel) -> None:
    """Shared bar+band drawing for one ACF or PACF panel.

    The shaded band is each lag's confidence interval re-centred at zero
    (confint[k] - values[k]), the same construction
    statsmodels.graphics.tsaplots.plot_acf/plot_pacf use for their own
    shading -- a bar poking outside this band is the standard "notable
    spike" visual cue, not a claim this module makes on its own.
    """
    highlight_set = set(highlight_lags)
    colors = ["#D55E00" if lag in highlight_set else "#4C72B0" for lag in lags]

    ax.vlines(lags, 0, values, color=colors, linewidth=1.3)
    ax.scatter(lags, values, color=colors, s=12, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)

    # Band is degenerate at lag 0 (acf/pacf == 1.0 exactly) -- skip it so
    # the shading reflects the lags that are actually informative.
    band_lags = lags[1:]
    band_lower = confint[1:, 0] - values[1:]
    band_upper = confint[1:, 1] - values[1:]
    ax.fill_between(
        band_lags, band_lower, band_upper, color="#4C72B0", alpha=0.15,
        label="95% significance band",
    )
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=-0.5)


def plot_acf_pacf(
    series: pd.Series,
    nlags: int = DEFAULT_ACF_PACF_NLAGS,
    highlight_lags: Sequence[int] = LAGS_OF_INTEREST,
    alpha: float = 0.05,
    pacf_method: str = DEFAULT_PACF_METHOD,
    series_label: str = "First-differenced development series",
):
    """Clean two-panel ACF + PACF figure (Phase 2 Step 4 deliverable).

    Bars at `highlight_lags` (default: LAGS_OF_INTEREST -- lags 1-7, then
    the weekly multiples through 56) are drawn in a distinct color so the
    lags this project's instructions single out are easy to find by eye;
    every other bar is neutral blue. Returns (fig, ACFPACFResult) so the
    caller gets the plotted figure and the exact numbers it was drawn
    from in one call.
    """
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    result = compute_acf_pacf(series, nlags=nlags, alpha=alpha, pacf_method=pacf_method)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    _plot_correlogram_panel(axes[0], result.lags, result.acf_values, result.acf_confint, highlight_lags, "ACF")
    _plot_correlogram_panel(axes[1], result.lags, result.pacf_values, result.pacf_confint, highlight_lags, "PACF")

    axes[0].set_title(f"Autocorrelation (ACF) -- {series_label}")
    axes[1].set_title(f"Partial autocorrelation (PACF) -- {series_label}")
    axes[-1].set_xlabel("Lag (days)")

    legend_handles = [
        mlines.Line2D([0], [0], color="#D55E00", lw=2, label="Lags of interest (1-7, weekly multiples to 56)"),
        mlines.Line2D([0], [0], color="#4C72B0", lw=2, label="Other lags"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, result


def plot_seasonal_lag7_diagnostic(
    series: pd.Series,
    nlags: int = DEFAULT_ACF_PACF_NLAGS,
    alpha: float = 0.05,
    seasonal_period: int = SEASONAL_PERIOD,
    series_label: str = "First-differenced development series",
):
    """Seasonal (lag-7) diagnostic figure: a scatter of the series against
    its own value `seasonal_period` steps earlier (a visual check for the
    weekly echo differencing can leave behind), plus a bar chart of ACF
    restricted to the multiples of `seasonal_period` out to `nlags`, each
    bar colored by whether it clears its own significance band.
    """
    import matplotlib.pyplot as plt

    clean = _validate_series_for_correlogram(series, nlags)
    acf_values, confint = compute_acf(clean, nlags=nlags, alpha=alpha)

    lagged = clean.shift(seasonal_period)
    paired = pd.concat([clean, lagged], axis=1, keys=["t", f"t_minus_{seasonal_period}"]).dropna()

    seasonal_lags = list(range(seasonal_period, nlags + 1, seasonal_period))
    seasonal_acf = [float(acf_values[lag]) for lag in seasonal_lags]
    margins = [float(confint[lag, 1] - acf_values[lag]) for lag in seasonal_lags]
    bar_colors = ["#D55E00" if abs(v) > m else "#4C72B0" for v, m in zip(seasonal_acf, margins)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].scatter(paired[f"t_minus_{seasonal_period}"], paired["t"], s=10, alpha=0.5, color="#4C72B0")
    axes[0].axhline(0, color="black", linewidth=0.6, linestyle=":")
    axes[0].axvline(0, color="black", linewidth=0.6, linestyle=":")
    axes[0].set_xlabel(f"value at t-{seasonal_period}")
    axes[0].set_ylabel("value at t")
    axes[0].set_title(f"Lag-{seasonal_period} scatter -- {series_label}")

    x_pos = range(len(seasonal_lags))
    axes[1].bar(x_pos, seasonal_acf, color=bar_colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    for x, m in zip(x_pos, margins):
        axes[1].plot([x - 0.4, x + 0.4], [m, m], color="gray", linewidth=0.8, linestyle="--")
        axes[1].plot([x - 0.4, x + 0.4], [-m, -m], color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(list(x_pos))
    axes[1].set_xticklabels([str(lag) for lag in seasonal_lags])
    axes[1].set_xlabel(f"Seasonal lag (multiples of {seasonal_period})")
    axes[1].set_ylabel("ACF")
    axes[1].set_title("ACF at weekly seasonal lags (orange = outside band)")

    fig.tight_layout()
    return fig


# =============================================================================
# Phase 2 Step 4 -- auto_arima cross-check
# =============================================================================


@dataclass
class AutoARIMAResult:
    """pmdarima.auto_arima's selected order, recorded as plain data.

    This is an automated-search *candidate*, meant to be compared against
    the manual ACF/PACF reading (compute_acf_pacf) -- not accepted on its
    own. `n_obs` records exactly how many observations the search saw,
    so a caller/test can confirm it matches the development-only series
    length and never the final holdout.
    """

    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    aic: float
    n_obs: int
    model_summary: str = field(repr=False, default="")

    def summary(self) -> str:
        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        return (
            f"auto_arima candidate (n_obs={self.n_obs}):\n"
            f"  ARIMA{self.order}\n"
            f"  SARIMA{self.order}({P},{D},{Q})[{m}]\n"
            f"  AIC: {self.aic:.4f}"
        )


def run_auto_arima_cross_check(
    series: pd.Series,
    d: int = 1,
    seasonal_m: int = SEASONAL_PERIOD,
    max_p: int = 7,
    max_q: int = 7,
    max_P: int = 2,
    max_Q: int = 2,
    seasonal: bool = True,
    **auto_arima_kwargs,
) -> AutoARIMAResult:
    """Run pmdarima.auto_arima on `series` as an efficient order-search
    cross-check against the manual ACF/PACF reading -- explicitly not a
    replacement for it (see this module's docstring and Step 4's own
    instructions: compare the two, don't just accept this one).

    `d` is fixed at the value the ADF test already established (Step 3:
    d=1 for this project's aggregate series) rather than re-searched,
    since that question is already answered. `seasonal_m` fixes the
    seasonal period (7 = weekly for daily retail data); D, P, and Q are
    still searched by auto_arima itself within `max_P`/`max_Q`.

    As with adf_test()/first_difference(), this function has no opinion
    on which period of the series it's given -- pass only the
    development-period series, never one that includes the final holdout.
    """
    import pmdarima as pm

    clean = pd.Series(series).dropna()
    model = pm.auto_arima(
        clean.to_numpy(),
        d=d,
        start_p=0,
        max_p=max_p,
        start_q=0,
        max_q=max_q,
        seasonal=seasonal,
        m=seasonal_m if seasonal else 1,
        start_P=0,
        max_P=max_P,
        start_Q=0,
        max_Q=max_Q,
        D=None,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
        **auto_arima_kwargs,
    )

    order = tuple(int(v) for v in model.order)
    seasonal_order = tuple(int(v) for v in model.seasonal_order) if seasonal else (0, 0, 0, 0)
    result = AutoARIMAResult(
        order=order,
        seasonal_order=seasonal_order,
        aic=float(model.aic()),
        n_obs=int(len(clean)),
        model_summary=str(model.summary()),
    )
    logger.info("\n%s", result.summary())
    return result
