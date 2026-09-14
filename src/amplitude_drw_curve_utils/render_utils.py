# src/utils/render_utils.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Dict, Callable

# --- use existing utilities; do NOT reimplement ---
from .utils import periodogram, red_noise_psd


# ======================= small math helpers =======================

def _to_numpy(signals) -> np.ndarray:
    """
    Normalize input to a plain numpy float array.

    Accepts numpy arrays, lists, or torch tensors (including tensors still
    on GPU or still requiring grad) without the caller having to remember
    to call .detach().cpu().numpy() themselves. This matters here because
    model_wrapper.sample(...) typically returns a torch.Tensor, and passing
    that straight into np.fft.rfft can silently misbehave rather than
    raising a clear error.
    """
    if hasattr(signals, "detach"):
        signals = signals.detach()
    if hasattr(signals, "cpu"):
        signals = signals.cpu()
    if hasattr(signals, "numpy"):
        signals = signals.numpy()
    return np.asarray(signals, dtype=float)


def _uniform_unitvar_bounds() -> float:
    """Uniform with unit variance has support [-a, a] where a = sqrt(3)."""
    return np.sqrt(3.0)

def _uniform_unitvar_ppf(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    a = _uniform_unitvar_bounds()
    return (2.0 * p - 1.0) * a

def _uniform_unitvar_pdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    a = _uniform_unitvar_bounds()
    return np.where((z >= -a) & (z <= a), 1.0 / (2.0 * a), 0.0)


# --- standardized Rayleigh reference (the true-DRW amplitude shape) ---
# If R ~ Rayleigh(sigma), Z = (R - sigma*sqrt(pi/2)) / (sigma*sqrt(2-pi/2))
# has a fixed distribution independent of sigma. u(z) recovers the
# corresponding unit-Rayleigh (sigma=1) value.
_RAYLEIGH_MEAN_COEF = np.sqrt(np.pi / 2.0)
_RAYLEIGH_STD_COEF = np.sqrt(2.0 - np.pi / 2.0)

def _standardized_rayleigh_pdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    u = z * _RAYLEIGH_STD_COEF + _RAYLEIGH_MEAN_COEF
    return np.where(u >= 0, u * np.exp(-0.5 * u * u) * _RAYLEIGH_STD_COEF, 0.0)

def _standardized_rayleigh_cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    u = z * _RAYLEIGH_STD_COEF + _RAYLEIGH_MEAN_COEF
    return np.where(u >= 0, 1.0 - np.exp(-0.5 * u * u), 0.0)

def _standardized_rayleigh_ppf(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    u = np.sqrt(-2.0 * np.log(1.0 - p))
    return (u - _RAYLEIGH_MEAN_COEF) / _RAYLEIGH_STD_COEF


def _ks_statistic(sample: np.ndarray, cdf) -> float:
    """
    One-sample Kolmogorov–Smirnov statistic:
      D_n = sup_x |F_n(x) - F(x)|
    'cdf' must be a callable returning model CDF values at x.
    """
    x = np.sort(np.asarray(sample))
    n = x.size
    if n == 0:
        return 0.0
    ecdf = (np.arange(1, n + 1)) / n
    F = np.asarray(cdf(x))
    d_plus = np.max(ecdf - F)
    d_minus = np.max(F - (np.arange(0, n) / n))
    return float(max(d_plus, d_minus))

def _std_per_frequency(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns to mean 0 / std 1."""
    values = np.asarray(values, dtype=float)
    mu = values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, ddof=0, keepdims=True) + 1e-12
    z = (values - mu) / sd
    return z, mu.squeeze(0), sd.squeeze(0)


# ======================= core extraction =======================

def extract_amplitudes(
    signals: np.ndarray,
    freq_indices: Optional[Sequence[int]] = None,
    freq_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rFFT amplitudes for each signal and select a subset of frequencies.

    Args:
        signals: array-like [N, T] of real time-series (numpy array, list,
                 or torch tensor -- normalized to numpy internally).
        freq_indices: explicit rFFT bin indices to keep; if None, use freq_range.
        freq_range: (fmin, fmax) in normalized freq units (cycles/sample).
                    If None, use all positive freqs (f > 0).

    Returns:
        amps:   [N, K_sel] amplitudes |X_k| across signals
        f_sel:  [K_sel] selected frequency values
        idx_sel:[K_sel] selected rFFT indices
    """
    signals = _to_numpy(signals)
    assert signals.ndim == 2, f"signals must be [N, T], got shape {signals.shape}"
    N, T = signals.shape
    f = np.fft.rfftfreq(T, d=1.0)

    if freq_indices is not None:
        idx_sel = np.array(freq_indices, dtype=int)
        idx_sel = idx_sel[(idx_sel >= 0) & (idx_sel < f.size)]
        if idx_sel.size == 0:
            raise ValueError("freq_indices contained no valid bins.")
    else:
        mask = f > 0
        if freq_range is not None:
            fmin, fmax = freq_range
            if fmin is not None:
                mask &= (f >= fmin)
            if fmax is not None:
                mask &= (f <= fmax)
        idx_sel = np.where(mask)[0]
        if idx_sel.size == 0:
            raise ValueError("No frequencies selected (check freq_range).")

    amps = []
    for i in range(N):
        X = np.fft.rfft(signals[i])
        amps.append(np.abs(X)[idx_sel])
    amps = np.vstack(amps)
    return amps, f[idx_sel], idx_sel

def pick_log_freq_indices(f: np.ndarray, n: int, fmin: float = 1e-3, fmax: float = 0.5) -> np.ndarray:
    """Pick ~log-spaced frequency indices in (fmin..fmax], avoiding DC."""
    mask = (f > 0) & (f >= fmin) & (f <= fmax)
    fpos = f[mask]
    if fpos.size == 0:
        raise ValueError("No positive frequencies in requested range.")
    targets = np.geomspace(fpos[0], fpos[-1], num=n)
    idx_all = np.arange(f.size)[mask]
    idx = np.searchsorted(fpos, targets)
    idx = np.clip(idx, 0, fpos.size - 1)
    return idx_all[np.unique(idx)]


# ======================= rendering diagnostics =======================

def render_amp_distribution_diagnostics(
    signals: np.ndarray,
    *,
    num_freqs: int = 4,
    freq_range: Tuple[float, float] = (1e-3, 0.4),
    bins: int = 50,
    title: Optional[str] = None,
    fontsize: float = 12,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Visual diagnostic for whether standardized Fourier-amplitude
    distributions per frequency look Uniform or Rayleigh (the true-DRW
    shape). Each histogram panel also reports the raw (pre-standardization)
    mean and variance of the amplitude at that frequency bin.

    `signals` may be a numpy array, list, or torch tensor (including a
    tensor still on GPU / requiring grad) -- it is normalized to a plain
    numpy array internally.

    For each selected frequency (K in total), produces 2 rows x K columns:
      Row A: Histogram (z-scores) + overlays of U[-sqrt3, sqrt3] and
             standardized Rayleigh + KS D stats + raw mean/variance box
      Row B: QQ plots vs Uniform and vs Rayleigh (unit-variance)

    Returns:
        fig, summary dict with averaged KS statistics:
          {"avg_KS_vs_Uniform": float, "avg_KS_vs_Rayleigh": float}
    """
    signals = _to_numpy(signals)
    assert signals.ndim == 2, f"signals must be [N, T], got shape {signals.shape}"
    N, T = signals.shape

    f_full = np.fft.rfftfreq(T, d=1.0)
    idx_sel = pick_log_freq_indices(f_full, num_freqs, fmin=freq_range[0], fmax=freq_range[1])
    amps, f_sel, _ = extract_amplitudes(signals, freq_indices=idx_sel)

    # Standardize per frequency (keep raw mean/std for annotation)
    z, mu, sd = _std_per_frequency(amps)
    K = f_sel.size

    fig, axes = plt.subplots(2, K, figsize=(4.2 * K, 6.0), sharey=False)
    if K == 1:
        axes = np.array(axes).reshape(2, 1)

    # Precompute references
    zu_line = np.linspace(-2.5, 2.5, 600)
    pdf_unif = _uniform_unitvar_pdf(zu_line)
    zr_line = np.linspace(-4, 4, 600)
    pdf_rayl = _standardized_rayleigh_pdf(zr_line)

    npoints = max(60, int(np.sqrt(N) * 10))
    pp = (np.arange(1, npoints + 1) - 0.5) / npoints
    q_unif = _uniform_unitvar_ppf(pp)
    q_rayl = _standardized_rayleigh_ppf(pp)

    # CDFs for KS (vectorized)
    def cdf_unif(x: np.ndarray) -> np.ndarray:
        a = _uniform_unitvar_bounds()
        return np.where(
            x <= -a, 0.0,
            np.where(x >= a, 1.0, (x + a) / (2.0 * a))
        )

    ks_vals_unif = []
    ks_vals_rayl = []

    for j, f_j in enumerate(f_sel):
        z_j = z[:, j]
        mu_j = float(mu[j])
        var_j = float(sd[j]) ** 2

        # ---- Row A: Histogram + overlays + KS + mean/var annotation ----
        axA = axes[0, j]
        axA.hist(z_j, bins=bins, density=True, alpha=0.35, color="tab:blue",
                  edgecolor="none", label="Data (std)", zorder=1)
        axA.plot(zu_line, pdf_unif, "k-.", lw=1.5, label="U[-\u221a3, \u221a3]", zorder=2)
        axA.plot(zr_line, pdf_rayl, "g:", lw=1.8, label="Rayleigh (DRW)", zorder=2)
        if j == 0:
            axA.set_ylabel("Density")
        axA.set_xlabel("Standardized amplitude z")
        axA.grid(True, ls=":")

        d_unif = _ks_statistic(z_j, cdf_unif)
        d_rayl = _ks_statistic(z_j, _standardized_rayleigh_cdf)
        ks_vals_unif.append(d_unif)
        ks_vals_rayl.append(d_rayl)
        axA.set_title(f"f = {f_j:.5f}\nKS D: U={d_unif:.3f}, R={d_rayl:.3f}", fontsize=10)

        # raw (pre-standardization) mean/variance box -- drawn last with a
        # high zorder so it always sits on top of the hist/pdf curves
        axA.text(
            0.97, 0.95,
            f"$\\mu = {mu_j:.3g}$\n$\\sigma^2 = {var_j:.3g}$",
            transform=axA.transAxes, ha="right", va="top",
            fontsize=max(fontsize * 0.9, 9),
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.6", alpha=0.9),
        )

        # ---- Row B: QQ vs Uniform & Rayleigh ----
        axB = axes[1, j]
        q_emp = np.quantile(z_j, pp)

        axB.plot(q_unif, q_emp, "o", ms=3.0, alpha=0.8, label="QQ vs Uniform", color="tab:orange")
        axB.plot(q_rayl, q_emp, "o", ms=3.0, alpha=0.8, label="QQ vs Rayleigh", color="tab:green")

        lo = min(q_emp.min(), q_unif.min(), q_rayl.min())
        hi = max(q_emp.max(), q_unif.max(), q_rayl.max())
        pad = 0.05 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        axB.plot([lo, hi], [lo, hi], "k:", lw=1.0)
        axB.set_xlim(lo, hi)
        axB.set_ylim(lo, hi)

        if j == 0:
            axB.set_ylabel("Empirical quantiles")
        axB.set_xlabel("Theoretical quantiles")
        axB.grid(True, ls=":")
        axB.legend(fontsize=8, loc="upper left")

    if title:
        fig.suptitle(title, y=0.98)

    fig.tight_layout()
    return fig, {
        "avg_KS_vs_Uniform": float(np.mean(ks_vals_unif)) if ks_vals_unif else np.nan,
        "avg_KS_vs_Rayleigh": float(np.mean(ks_vals_rayl)) if ks_vals_rayl else np.nan,
    }


# ======================= PSD comparison renderer =======================

def render_avg_psd_vs_target(
    signals: np.ndarray,
    *,
    target_psd_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    decay: float = 0.95,
    sigma: float = 0.08,
    tau: float | None = None,
    title: Optional[str] = None,
    loglog: bool = True,
    skip_lasts=3,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Compare the average periodogram (PSD) of a batch of signals against a target PSD.

    `signals` may be a numpy array, list, or torch tensor -- normalized to
    a plain numpy array internally.

    Uses existing utils:
      - periodogram(x) -> one-sided PSD from squared Fourier amplitudes
      - red_noise_psd(f, ...) using Kozlowski et al. (2010), equation 2

    Steps:
      1) Compute PSD for each row in `signals` via `periodogram`.
      2) Average the PSDs across curves (same freq grid).
      3) Build target PSD on that grid.
      4) Plot both curves and report a relative L2 error over f>0.

    Returns:
      fig, {"rel_L2": float}
    """
    signals = _to_numpy(signals)
    assert signals.ndim == 2, f"signals must be [N, T], got shape {signals.shape}"
    N, _ = signals.shape

    # Average PSD
    psds = []
    f_ref = None
    for i in range(N):
        f, P = periodogram(signals[i])
        if f_ref is None:
            f_ref = f
        psds.append(P)
    avg_psd = np.mean(np.vstack(psds), axis=0)

    # Target PSD
    if target_psd_fn is None:
        target_psd = red_noise_psd(f_ref, decay=decay, tau=tau, sigma_hat=sigma)
    else:
        target_psd = target_psd_fn(f_ref)

    # Relative L2 error over f>0
    mask = f_ref > 0
    num = np.linalg.norm(avg_psd[mask] - target_psd[mask])
    den = np.linalg.norm(target_psd[mask]) + 1e-12
    rel_l2 = float(num / den)

    # Plot
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    stop = max(2, f_ref.size - skip_lasts)
    plot_slice = slice(1, stop)
    if loglog:
        ax.loglog(f_ref[plot_slice], avg_psd[plot_slice], lw=2, label="Average PSD")
        ax.loglog(f_ref[plot_slice], target_psd[plot_slice], "--", lw=2, label="Target PSD")
    else:
        ax.plot(f_ref[plot_slice], avg_psd[plot_slice], lw=2, label="Average PSD")
        ax.plot(f_ref[plot_slice], target_psd[plot_slice], "--", lw=2, label="Target PSD")
    ax.set_xlabel("Normalized frequency [cycles/sample]")
    ax.set_ylabel("PSD = |FFT|^2 (one-sided normalized)")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    ttl = title if title else "Average PSD vs target"
    ax.set_title(f"{ttl}\nRel L2 (f>0) = {rel_l2:.3%}")
    fig.tight_layout()

    return fig, {"rel_L2": rel_l2}
