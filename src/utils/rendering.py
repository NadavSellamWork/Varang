import torch
import matplotlib.pyplot as plt 

# rendering
# =========================
# Utils
# =========================
def _as_cpu_flat_float64(x):
    return torch.as_tensor(x, dtype=torch.float64).flatten().cpu()

def _as_cpu_flat_bool(x):
    return torch.as_tensor(x, dtype=torch.bool).flatten().cpu()

def _roc_from_pvalues_single(p_values, labels, num_thresholds: int = 400):
    p = _as_cpu_flat_float64(p_values)
    y = _as_cpu_flat_bool(labels)
    if p.numel() != y.numel():
        raise ValueError(f"size mismatch: p_values {tuple(p.shape)} vs labels {tuple(y.shape)}")

    scores = 1.0 - p
    thr = torch.linspace(scores.max(), scores.min(), steps=int(num_thresholds), dtype=torch.float64)

    pred_pos = scores[:, None] >= thr[None, :]
    y_pos = y[:, None]

    tp = (pred_pos & y_pos).sum(dim=0).to(torch.float64)
    fp = (pred_pos & (~y_pos)).sum(dim=0).to(torch.float64)
    fn = ((~pred_pos) & y_pos).sum(dim=0).to(torch.float64)
    tn = ((~pred_pos) & (~y_pos)).sum(dim=0).to(torch.float64)

    tpr = tp / (tp + fn).clamp_min(1.0)
    fpr = fp / (fp + tn).clamp_min(1.0)

    fpr = torch.cat([torch.tensor([0.0], dtype=torch.float64), fpr, torch.tensor([1.0], dtype=torch.float64)])
    tpr = torch.cat([torch.tensor([0.0], dtype=torch.float64), tpr, torch.tensor([1.0], dtype=torch.float64)])

    order = torch.argsort(fpr)
    fpr_s = fpr[order]
    tpr_s = tpr[order]
    auc = torch.trapz(tpr_s, fpr_s).item()
    return fpr_s, tpr_s, auc

# =========================
# Rendering: ROC (multi)
# =========================
def plot_rocs_from_items(
    items,
    num_thresholds: int = 400,
    ax=None,
    title="ROC curves",
    show_diagonal=True,
    annotate_endpoints=False,
):
    """
    items: list of dicts (or tuples) describing cases to compare.

    Supported item formats:
      1) {"p_values": ..., "labels": ..., "name": "Flow"}
      2) (p_values, labels)
      3) (name, p_values, labels)

    Returns: ax, list_of_aucs
    """
    parsed = []
    for it in items:
        if isinstance(it, dict):
            p = it["p_values"]
            y = it["labels"]
            name = it.get("name", "case")
        elif isinstance(it, (list, tuple)) and len(it) == 2:
            p, y = it
            name = "case"
        elif isinstance(it, (list, tuple)) and len(it) == 3:
            name, p, y = it
        else:
            raise ValueError("Each item must be dict with keys {p_values, labels} (optional name), or tuple (p, y) / (name, p, y).")
        parsed.append((name, p, y))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 6.0))

    # prettier axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    aucs = []
    for name, p, y in parsed:
        fpr, tpr, auc = _roc_from_pvalues_single(p, y, num_thresholds=num_thresholds)
        aucs.append(auc)
        ax.plot(fpr.numpy(), tpr.numpy(), linewidth=2.2, label=f"{name}  (AUC={auc:.4f})")

        if annotate_endpoints:
            ax.scatter([0.0, 1.0], [0.0, 1.0], s=20)

    if show_diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    return ax, aucs

# =========================
# Rendering: p-value histograms (multi-row, nice colors)
# =========================
def plot_pvalue_histograms_from_items(
    items,
    bins: int = 40,
    density: bool = True,
    title="p-value distributions",
    alpha=0.55,
    ylim_max: float = None,
    fontsize: float = 12,
):
    """
    Each item gets its own row:

        Row i, Col 0 = positives (blue)
        Row i, Col 1 = negatives (red)

    Colors match the attached screenshot style.
    
    Parameters:
        ylim_max: Optional maximum y-axis value. If provided, all histograms will have the same y-range [0, ylim_max].
        fontsize: Font size for labels, titles, and tick labels.
    """

    # -------------------------
    # Parse inputs
    # -------------------------
    parsed = []
    for it in items:
        if isinstance(it, dict):
            p = it["p_values"]
            y = it["labels"]
            name = it.get("name", "case")
        elif isinstance(it, (list, tuple)) and len(it) == 2:
            p, y = it
            name = "case"
        elif isinstance(it, (list, tuple)) and len(it) == 3:
            name, p, y = it
        else:
            raise ValueError(
                "Each item must be dict with keys {p_values, labels} (optional name), "
                "or tuple (p, y) / (name, p, y)."
            )
        parsed.append((name, p, y))

    n_items = len(parsed)

    # -------------------------
    # Colors (like screenshot)
    # -------------------------
    blue = "cornflowerblue"
    red  = "salmon"

    # -------------------------
    # Subplot grid: rows = items, cols = 2
    # -------------------------
    fig, axes = plt.subplots(
        n_items, 2,
        figsize=(12.4, 3.2 * n_items),
        sharex=True,
        sharey=True,
    )

    # If only 1 item, axes isn't 2D → fix shape
    if n_items == 1:
        axes = axes[None, :]

    # -------------------------
    # Plot each item on its own row
    # -------------------------
    for row_idx, (name, p, y) in enumerate(parsed):

        # Convert tensors safely
        p_t = _as_cpu_flat_float64(p)
        y_t = _as_cpu_flat_bool(y)

        p_pos = p_t[y_t].numpy()
        p_neg = p_t[~y_t].numpy()

        ax_pos = axes[row_idx, 0]
        ax_neg = axes[row_idx, 1]

        # Style both axes
        for ax in (ax_pos, ax_neg):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
            ax.set_axisbelow(True)
            ax.set_xlim(0, 1)
            if ylim_max is not None:
                ax.set_ylim(0, ylim_max)
            ax.tick_params(axis="both", labelsize=fontsize)

        # Positive histogram (blue)
        ax_pos.hist(
            p_pos,
            bins=bins,
            range=(0, 1),
            density=density,
            histtype="stepfilled",
            linewidth=1.4,
            alpha=alpha,
            color=blue,
            edgecolor="white",
        )

        # Negative histogram (red)
        ax_neg.hist(
            p_neg,
            bins=bins,
            range=(0, 1),
            density=density,
            histtype="stepfilled",
            linewidth=1.4,
            alpha=alpha,
            color=red,
            edgecolor="white",
        )

        # Titles per row
        ax_pos.set_title(f"{name} | Positive", fontsize=fontsize)
        ax_neg.set_title(f"{name} | Negative", fontsize=fontsize)

        # Y-label only on left
        ax_pos.set_ylabel("Density" if density else "Count", fontsize=fontsize)

    # X-label only on bottom row
    axes[-1, 0].set_xlabel("p-value", fontsize=fontsize)
    axes[-1, 1].set_xlabel("p-value", fontsize=fontsize)

    # Global title
    fig.suptitle(title, y=1.01, fontsize=fontsize)

    fig.tight_layout()
    return fig, axes