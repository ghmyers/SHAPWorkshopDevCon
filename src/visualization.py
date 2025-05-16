import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

def shap_beeswarm_kdepoints(
    shap_values,
    data,
    feature_names,
    expected_values=None,
    *,
    max_display=15,
    cmap=mpl.cm.get_cmap("coolwarm"),
    point_size=9,
    n_grid=200,
    max_half_height=0.3,
    figsize=(12, 6),
    ax=None,
    show=True,
    x_min=None,
    x_max=None,
    colorbar=True
):
    """Beeswarm where the point cloud itself traces the KDE/violin shape."""
    shap_values = np.asarray(shap_values)
    # x_min = np.min(np.min(shap_values, axis=0)) * 1.1
    # x_max = np.max(np.max(shap_values, axis=0)) * 1.1
    data        = np.asarray(data)
    if shap_values.shape != data.shape:
        raise ValueError("`shap_values` and `data` must have identical shape.")

    n_samples, n_features = shap_values.shape
    max_display = min(max_display, n_features)

    # ‑‑‑ 1. rank features
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    feat_order = np.argsort(mean_abs)[::-1][:max_display]

    # ‑‑‑ 2. colour normalisation
    vmin = np.nanpercentile(data[:, feat_order],  0)
    vmax = np.nanpercentile(data[:, feat_order], 100)
    norm = mpl.colors.Normalize(vmin, vmax)

    # ‑‑‑ 3. figure boiler‑plate
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # 4. loop bottom→top so most‑important ends up on top
    for row, feat_idx in enumerate(feat_order[::-1]):
        sv = shap_values[:, feat_idx]
        fv = data[:,  feat_idx]

        # 4a ‑ KDE on fixed grid
        kde = stats.gaussian_kde(sv, bw_method="scott")
        xs  = np.linspace(sv.min(), sv.max(), n_grid)
        dens = kde(xs)
        dens = dens / dens.max() * max_half_height

        # 4b ‑ assign each sample to nearest grid point
        bin_idx = np.searchsorted(xs, sv, side="left")
        bin_idx = np.clip(bin_idx, 1, n_grid-1) - 1

        y_off = np.zeros_like(sv)
        for b in np.unique(bin_idx):
            pts = np.where(bin_idx == b)[0]
            if not len(pts):
                continue
            h = dens[b]
            k = len(pts)
            offs = np.linspace(-h, h, k)
            ordering = np.argsort(np.abs(sv[pts]))[::-1]
            y_off[pts[ordering]] = offs

        # 4c ‑‑‑ Plot the points
        ax.scatter(
            sv,
            row + y_off,
            s=point_size,
            c=cmap(norm(fv)),
            alpha=0.8,
            linewidth=0,
            rasterized=True,
        )

    # 5. axes cosmetics
    ax.axvline(0, color="grey", lw=0.8)
    if expected_values is not None:
        base = np.median(expected_values) if np.ndim(expected_values) else expected_values
        ax.axvline(base, color="grey", lw=0.8, ls="--")

    yticks = np.arange(max_display)
    ax.set_yticks(yticks)
    ax.set_yticklabels([feature_names[i] for i in feat_order[::-1]])
    ax.set_ylim(-1, max_display)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("SHAP value")
    ax.grid(axis="x", ls=":", lw=0.4)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    if colorbar:
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.outline.set_visible(False)

        cbar.set_label("Feature value", rotation=270, labelpad=15)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)


    fig.tight_layout()
    if show:
        plt.show()
    
    return fig, ax

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap

def shap_bar_plot(
    shap_exp,
    *,
    max_display=20,
    color="#fc0454",   # SHAP’s default diverging palette
    axis_color="#333333",
    text_color="#333333",
    bar_thickness=0.6,
    show=True,
    figsize=(6, 4),
    ax=None,
    x_max=None
):
    """
    Bar plot of global feature importance (mean|SHAP|) **without** the
    automatic "Sum of n other features" bar that SHAP’s built‑in bar plot
    adds when `max_display < n_features`.

    Parameters
    ----------
    shap_exp : shap.Explanation or (values, feature_names) tuple
        A 2‑D SHAP Explanation (or matrix) for a single‑output model.
    max_display : int, default 20
        Show at most this many most‑important features.  No aggregate bar is
        added even when fewer than n_features are displayed.
    color : matplotlib colormap or callable, default SHAP red‑blue
        Colour map used for the bars; called with [0‑1] importance scaling.
    axis_color, text_color : str
        Colours for axes / tick labels / bar annotations.
    bar_thickness : float, default 0.6
        Height of each horizontal bar (in axes coordinates).
    value_format : str
        Format‐string for the numeric annotation at bar ends.
    show : bool, default True
        If True, calls `plt.show()` at the end.
    figsize : tuple, default (6, 4)
        Figure size if `ax` is not provided.
    ax : matplotlib.axes.Axes, optional
        If given, draw into this existing axes; otherwise a new figure/
        axes is created.

    Returns
    -------
    fig, ax : the Matplotlib Figure and Axes objects.
    """
    # ------------- 0. Unpack / validate inputs ---------------------------
    if isinstance(shap_exp, shap._explanation.Explanation):
        shap_values = shap_exp.values
        feature_names = shap_exp.feature_names
    else:
        shap_values, feature_names = shap_exp  # assume iterable/tuple

    shap_values = np.asarray(shap_values)
    if shap_values.ndim != 2:
        raise ValueError("`shap_exp` must be 2‑D (n_samples, n_features)")

    n_samples, n_features = shap_values.shape
    if len(feature_names) != n_features:
        raise ValueError("feature_names length mismatch")

    # ------------- 1. Global importance & ordering -----------------------
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_raw = np.mean(shap_values, axis=0)      # sign for colour map
    order = np.argsort(mean_abs)[::-1]

    # limit how many bars to draw
    k = min(max_display, n_features)
    order = order[:k]

    mean_abs = mean_abs[order]
    mean_raw = mean_raw[order]
    labels   = np.array(feature_names, dtype=str)[order]

    # ------------- 2. Make / reuse axes ----------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ------------- 3. Draw horizontal bars ------------------------------
    y_pos = np.arange(k)   # most important at top
    bars  = ax.barh(
        y_pos,
        mean_abs,
        height=bar_thickness,
        color=color,
        edgecolor="none",
    )

    # ------------- 4. Annotate bars with ±value -------------------------
    for bar, val in zip(bars, mean_abs):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            x + mean_abs.max() * 0.01,  # small offset
            y,
            round(x, 4),
            va="center",
            ha="left",
            fontsize=9,
            color=text_color,
        )

    # ------------- 5. Axes styling --------------------------------------
    if x_max is None:
        x_max = mean_abs.max() * 1.15
    else:
        x_max = x_max
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, color=text_color)
    ax.invert_yaxis()                   # top‑rank first
    ax.set_xlabel("mean(|SHAP value|)", color=axis_color)
    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)
    ax.grid(axis="x", ls=":", lw=0.4)
    ax.set_xlim(0, x_max)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

def plot_shap_bar_unscaled(shap_vals,
                           feature_names=None,
                           max_display=20,
                           title="Global feature importance (original target units)",
                           figsize=(6, 4),
                           ax=None,
                           abs_vals=True):
    """
    Draw a horizontal bar chart of mean SHAP magnitudes.

    Parameters
    ----------
    shap_vals : shap.Explanation | np.ndarray
        SHAP values after you have *already* rescaled them into the
        original target units ( multiply by σ_y, etc. ).
        Shape should be (n_samples, n_features).

    feature_names : list[str] | None
        Names for the y-axis.  If None and `shap_vals` is an Explanation,
        the function will try `shap_vals.feature_names`.

    max_display : int
        Plot only the `max_display` most important features.

    title : str
        Figure title.

    figsize : tuple
        Matplotlib figure size in inches.

    ax : matplotlib.axes.Axes | None
        Supply an existing axes if you want to embed the plot; otherwise
        the function creates a new figure.

    abs_vals : bool
        If True (default) ranks features by `mean(abs(SHAP))`.  Set False
        if you prefer signed means to see net positive / negative effects.

    Returns
    -------
    matplotlib.axes.Axes
    """
    # ------------------------------------------------------------
    # 1.  Get a 2-D NumPy array from whatever the caller passed in
    # ------------------------------------------------------------
    if hasattr(shap_vals, "values"):              # Explanation object
        S = shap_vals.values
        if feature_names is None:
            feature_names = getattr(shap_vals, "feature_names", None)
    else:                                        # already ndarray
        S = np.asarray(shap_vals)

    if S.ndim != 2:
        raise ValueError("`shap_vals` must be 2-D (n_samples, n_features).")

    n_features = S.shape[1]

    # ------------------------------------------------------------
    # 2.  Compute global importance vector
    # ------------------------------------------------------------
    if abs_vals:
        importance = np.abs(S).mean(axis=0)
    else:
        importance = S.mean(axis=0)


    # pick top-k
    order = np.argsort(importance)[::-1][:max_display]

    # ------------------------------------------------------------
    # 3.  Make the plot
    # ------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(order))[::-1]          # largest at the top
    ax.barh(y_pos, importance[order][::-1])      # default matplotlib colour
    ax.set_yticks(y_pos)

    # graceful fallback if no names provided
    if feature_names is None:
        feature_names = [f"feat {i}" for i in range(n_features)]

    ax.set_yticklabels(np.array(feature_names)[order][::-1], fontsize=9)
    ax.set_xlabel("mean(|SHAP value|)" if abs_vals else "mean SHAP value")
    ax.set_title(title, weight="bold")
    ax.invert_yaxis()                            # highest at the top
    plt.tight_layout()
    return ax



import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_all_dependence(
        shap_values,               # (n_samples, n_features) – ndarray or Explanation.values
        X_values,                  # (n_samples, n_features) – ndarray or DataFrame
        feature_names=None,        # list/array of length n_features
        n_cols=4,                  # how many subplots per row
        point_size=8,
        alpha=0.8,
        cmap_name="tab20",
        figsize_per_col=4,
        figsize_per_row=3,
        zero_line=True):
    """
    Grid of dependence plots: one scatter per feature.

    Returns
    -------
    fig, axes : matplotlib Figure and ndarray of Axes
    """

    # --- 0. Normalise inputs -----------------------------------------------
    # shap_values might be a shap.Explanation
    if hasattr(shap_values, "values"):
        S = shap_values.values
        if feature_names is None:
            feature_names = getattr(shap_values, "feature_names", None)
    else:
        S = np.asarray(shap_values)

    Xv = X_values.values if hasattr(X_values, "values") else np.asarray(X_values)
    n_samples, n_features = S.shape

    if feature_names is None:
        feature_names = [f"feat {i}" for i in range(n_features)]

    # --- 1. Figure layout ---------------------------------------------------
    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * figsize_per_col, n_rows * figsize_per_row),
        squeeze=False
    )

    # --- 2. Colour palette --------------------------------------------------
    cmap   = cm.get_cmap(cmap_name, n_features)
    colours = [cmap(i) for i in range(n_features)]

    # --- 3. Draw each subplot ----------------------------------------------
    for i, (ax, name, colour) in enumerate(zip(axes.flat, feature_names, colours)):

        ax.scatter(Xv[:, i], S[:, i],
                   s=point_size, alpha=alpha,
                   color=colour, edgecolor="none")

        if zero_line:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")

        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Feature value", fontsize=8)
        ax.set_ylabel("SHAP value", fontsize=8)
        ax.tick_params(labelsize=8)

    # Hide any empty cells
    for ax in axes.flat[n_features:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes