import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# ============================================================
#   Rayleigh-matching coefficients
#   (moments of Rayleigh(scale=1): E[R]=sqrt(pi/2), Var[R]=2-pi/2)
# ============================================================
_RAYLEIGH_MEAN_COEF = np.sqrt(np.pi / 2.0)          # ≈ 1.2533
_RAYLEIGH_STD_COEF = np.sqrt(2.0 - np.pi / 2.0)      # ≈ 0.6551

# For reference: relative std of a true-DRW amplitude at any single bin
RAYLEIGH_RELATIVE_STD = _RAYLEIGH_STD_COEF / _RAYLEIGH_MEAN_COEF  # ≈ 0.523


# ============================================================
#   Amplitude distribution samplers
#
#   Each takes `sigma`, the Rayleigh *scale* parameter at each frequency
#   bin -- sigma_k = sqrt(mu_k / 2), where mu_k = E[|X_k|^2] is the mean
#   power set by the target PSD. This is NOT a free "spread" knob: for a
#   true DRW (a stationary Gaussian process), Re(X_k) and Im(X_k) are
#   i.i.d. N(0, sigma_k^2), which forces |X_k| ~ Rayleigh(sigma_k).
#
#   uniform_distribution builds an *ablation* dataset: same mean and
#   variance as the true DRW's Rayleigh amplitude at every frequency, but
#   a different (non-negative) shape. This isolates shape deviations from
#   the true DRW while holding the target PSD fixed.
# ============================================================

def rayleigh_distribution(sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    True-DRW amplitude distribution: |X_k| ~ Rayleigh(scale=sigma).
    E[R] = sigma*sqrt(pi/2), Var[R] = sigma^2*(2 - pi/2). Non-negative by
    construction.
    """
    return rng.rayleigh(scale=sigma)


# Uniform bounds solved from: mean = (a+b)/2, var = (b-a)^2/12,
# matched to Rayleigh(sigma)'s mean & variance.
_UNIFORM_A_COEF = _RAYLEIGH_MEAN_COEF - np.sqrt(3.0) * _RAYLEIGH_STD_COEF  # ≈ 0.1184
_UNIFORM_B_COEF = _RAYLEIGH_MEAN_COEF + np.sqrt(3.0) * _RAYLEIGH_STD_COEF  # ≈ 2.3882


def uniform_distribution(sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Amplitude distribution matched to Rayleigh(sigma)'s mean & variance,
    Uniform-shaped. Bounds are fixed positive multiples of sigma
    (a ≈ 0.118*sigma, b ≈ 2.388*sigma), so this is non-negative by
    construction -- no clipping needed.
    """
    sigma = np.asarray(sigma, dtype=float)
    a = sigma * _UNIFORM_A_COEF
    b = sigma * _UNIFORM_B_COEF
    return rng.uniform(a, b)


# ============================================================
#   PSD model
# ============================================================
def _tau_from_decay(decay: float, dt: float) -> float:
    decay = float(decay)
    if not (0.0 < decay < 1.0):
        raise ValueError("decay must lie in (0, 1) when converting to tau.")
    return -float(dt) / np.log(decay)


def red_noise_psd(
    f: np.ndarray,
    decay: float | None = 0.95,
    sigma: float = 0.08,
    *,
    tau: float | None = None,
    sigma_hat: float | None = None,
    dt: float = 1.0,
) -> np.ndarray:
    """
    One-sided DRW PSD from Kozlowski et al. (2010), equation 2:

      P(f) = 2 * sigma_hat^2 * tau^2 / (1 + (2 * pi * tau * f)^2)

    where PSD is the squared Fourier amplitude with one-sided normalization.

    `tau` is the preferred interface. `decay` is kept as a legacy alias and is
    converted via decay = exp(-dt / tau).
    """
    f = np.asarray(f)
    sigma_hat = float(sigma if sigma_hat is None else sigma_hat)
    if tau is None:
        if decay is None:
            raise ValueError("Provide either tau or decay.")
        tau = _tau_from_decay(decay, dt)
    tau = float(tau)
    return (2.0 * sigma_hat**2 * tau**2) / (1.0 + (2.0 * np.pi * tau * f) ** 2)


# ============================================================
#   Fourier synthesis
# ============================================================
def generate_curve(
    n: int,
    *,
    decay: float = 0.95,
    tau: float | None = None,
    sigma_shape: float = 0.08,
    zero_dc: bool = True,
    rng: np.random.Generator | None = None,
    distribution: Callable[[np.ndarray, np.random.Generator], np.ndarray] = rayleigh_distribution,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Generate one light curve directly in the frequency domain.

    At each interior rFFT bin k, the target PSD S(f_k) (Kozlowski et al.
    2010, eq. 2) sets the mean power mu_k = E[|X_k|^2]. For a true DRW
    (a stationary Gaussian process), Re(X_k) and Im(X_k) are i.i.d.
    N(0, mu_k/2), so the amplitude |X_k| is Rayleigh-distributed with
    scale sigma_k = sqrt(mu_k / 2). Fixing tau, sigma_shape therefore fully
    determines the mean, variance, *and* shape of the amplitude
    distribution at every frequency -- it is not a free perturbation to
    tune.

    `distribution(sigma_k, rng)` draws |X_k| given sigma_k at each bin.
    The default, `rayleigh_distribution`, gives a true DRW realization.
    Swap in `uniform_distribution` to build an ablation dataset whose
    amplitude has the *same* mean and variance as the true DRW at every
    frequency, but a bounded, non-Rayleigh shape.

    Phase at each bin is drawn independently and uniformly in [0, 2*pi).
    DC and Nyquist bins are purely real (1 degree of freedom) and are
    drawn directly as N(0, mu_k), not passed through `distribution`.

    Args:
        n: number of time-domain samples.
        decay: legacy AR(1)-style damping coefficient; converted to tau
            via tau = -dt / log(decay). Ignored if `tau` is given.
        tau: DRW damping timescale (preferred interface).
        sigma_shape: DRW variability amplitude (sigma_hat in the PSD).
        zero_dc: if True, force the DC bin to exactly zero (typical for
            detrended light curves). If False, draw it as N(0, mu_0).
        rng: numpy random Generator; a fresh default_rng() is used if None.
        distribution: callable(sigma_k, rng) -> amplitude samples at each
            interior bin. Defaults to `rayleigh_distribution` (true DRW).
        dt: sample spacing.

    Returns:
        Real-valued time-domain curve of length n.
    """
    if rng is None:
        rng = np.random.default_rng()

    f = np.fft.rfftfreq(n, d=dt)
    S = red_noise_psd(f, decay=decay, tau=tau, sigma_hat=sigma_shape, dt=dt)

    # mean power mu_k = E[|X_k|^2] at each bin
    pow_target = (n / (2.0 * dt)) * S
    pow_target[0] = (n / dt) * S[0]
    has_nyquist = (n % 2 == 0) and (pow_target.size > 1)
    if has_nyquist:
        pow_target[-1] = (n / dt) * S[-1]

    interior = np.ones(f.size, dtype=bool)
    interior[0] = False
    if has_nyquist:
        interior[-1] = False

    X = np.zeros(f.size, dtype=complex)

    # interior bins: Rayleigh scale sigma_k = sqrt(mu_k / 2)
    sigma_k = np.sqrt(pow_target[interior] / 2.0)
    R = np.clip(distribution(sigma_k, rng), 0.0, None)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=sigma_k.shape)
    X[interior] = R * np.exp(1j * phi)

    # DC bin: real, N(0, mu_0)
    if zero_dc:
        X[0] = 0.0
    else:
        X[0] = rng.normal(0.0, np.sqrt(pow_target[0]))

    # Nyquist bin (n even): real, N(0, mu_{n/2})
    if has_nyquist:
        X[-1] = rng.normal(0.0, np.sqrt(pow_target[-1]))

    return np.fft.irfft(X, n=n)


# ============================================================
#   Periodogram
# ============================================================
def periodogram(x: np.ndarray, dt=1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-sided PSD from squared Fourier amplitudes:
      interior: S_hat = (2 dt / n) |X_k|^2
      DC/Nyquist: (dt / n) |X_k|^2
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    f = np.fft.rfftfreq(n, d=dt)
    z = x - np.mean(x)
    X = np.fft.rfft(z)
    P = (2.0 * dt / n) * (np.abs(X) ** 2)
    P[0] *= 0.5
    if n % 2 == 0:
        P[-1] *= 0.5
    return f, P


# ============================================================
#   Demo
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(1234)
    decay, sigma_shape = 0.995, 0.08

    for dist_name, dist_fn, color in [
        ("Rayleigh (true DRW)", rayleigh_distribution, "tab:green"),
        ("Uniform (ablation)", uniform_distribution, "tab:orange"),
    ]:
        n = 2048
        trials = 300
        psds = []
        for _ in range(trials):
            x = generate_curve(
                n,
                decay=decay,
                sigma_shape=sigma_shape,
                rng=rng,
                distribution=dist_fn,
            )
            f, P = periodogram(x)
            psds.append(P)
        psd_avg = np.mean(psds, axis=0)
        target = red_noise_psd(f, decay=decay, sigma=sigma_shape)
        plt.loglog(f[1:], psd_avg[1:], lw=2, color=color, label=dist_name)
        plt.loglog(f[1:], target[1:], "--", color=color, alpha=0.7)

    plt.xlabel("Normalized frequency [cycles/sample]")
    plt.ylabel("PSD = |FFT|^2 (one-sided normalized)")
    plt.title("Averaged PSD vs Target (same PSD, different amplitude shapes)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
#   QQ
# ============================================================
def qq_distance(
    arr_1: np.ndarray,
    arr_2: np.ndarray,
    *,
    n_quantiles: int = 200,
    metric: str = "L2"
) -> float:
    """
    Quantile–Quantile (QQ) distribution similarity metric.

    This function measures how *similar* two 1D empirical distributions are
    by comparing their *quantile functions* directly — i.e., how close the
    quantile–quantile (QQ) curve is to the identity line.

    -------------------
    Mathematical basis:
    -------------------

    Let X₁ and X₂ be two empirical random variables with sorted samples
    {x₁₍₁₎ ≤ ... ≤ x₁₍ₙ₎} and {x₂₍₁₎ ≤ ... ≤ x₂₍ₘ₎}.

    Define their *empirical quantile functions* Q₁(p) and Q₂(p), which
    return the value below which a fraction p of samples lie.

    We evaluate both quantile functions on a grid of probabilities
    p ∈ (0, 1), e.g., pᵢ = (i - 0.5)/n_quantiles for i = 1..n_quantiles.

    The QQ–difference curve is:
        Δ(p) = Q₁(p) - Q₂(p)

    Then the QQ–distance metric is:

      • For metric="L2":
            D = sqrt( ∫₀¹ [Δ(p)]² dp )
          (approximated numerically over the grid)

      • For metric="L1":
            D = ∫₀¹ |Δ(p)| dp

      • For metric="KS":
            D = maxₚ |Δ(p)|
          (analogous to Kolmogorov–Smirnov distance, but in value-space)

    Interpretation:
      - D = 0 means identical distributions.
      - Larger D means the distributions differ more strongly.
      - D has the same units as the data (e.g., amplitude units).

    Args:
        arr_1, arr_2: 1D arrays of samples (need not be same length).
        n_quantiles:  number of quantile points to sample between (0,1).
        metric:       one of {"L2", "L1", "KS"} for L2 norm, L1 norm, or max abs diff.

    Returns:
        A single scalar float — the QQ-based distance between the two distributions.

    Example:
        >>> x = np.random.normal(0, 1, 10000)
        >>> y = np.random.normal(0, 1.2, 10000)
        >>> qq_distance(x, y)
        0.25  # roughly proportional to the variance difference
    """
    x1 = np.asarray(arr_1).ravel()
    x2 = np.asarray(arr_2).ravel()
    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]
    if x1.size == 0 or x2.size == 0:
        return np.nan

    # define quantile grid
    p = (np.arange(1, n_quantiles + 1) - 0.5) / n_quantiles
    q1 = np.quantile(x1, p)
    q2 = np.quantile(x2, p)
    diff = q1 - q2

    metric = metric.upper()
    if metric == "L2":
        D = np.sqrt(np.mean(diff ** 2))
    elif metric == "L1":
        D = np.mean(np.abs(diff))
    elif metric == "KS":
        D = np.max(np.abs(diff))
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from 'L1', 'L2', 'KS'.")

    return float(D)
