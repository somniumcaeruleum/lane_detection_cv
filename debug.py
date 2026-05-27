import numpy as np
import matplotlib.pyplot as plt

def show_curves(*curves, labels=None, colors=None, styles=None,
                title='Curves', xlabel='x', ylabel='y',
                equal_aspect=True, grid=True, figsize=(100, 40),
                show_points=False, ax=None):
    """
    Plot one or more curves on a single figure.
    
    Parameters
    ----------
    *curves : tuples of (x, y)
        Each curve passed as a tuple/list of two array-likes.
        Example: show_curves((x1, y1), (x2, y2), (x3, y3))
    labels : list of str, optional
        Legend label for each curve. Defaults to 'Curve 1', 'Curve 2', ...
    colors : list of str, optional
        Color for each curve. Defaults to matplotlib's cycle.
    styles : list of str, optional
        Line style for each curve (e.g., '-', '--', ':', '-.'). Defaults to '-'.
    title, xlabel, ylabel : str
        Plot text.
    equal_aspect : bool
        Force equal x/y scaling (important for geometric curves).
    grid : bool
        Show grid.
    figsize : tuple
        Figure size if a new figure is created.
    show_points : bool
        If True, also draw the sample points as small dots.
    ax : matplotlib Axes, optional
        Plot on an existing axes instead of creating a new figure.
    
    Returns
    -------
    ax : matplotlib Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    n = len(curves)
    if labels is None:
        labels = [f'Curve {i + 1}' for i in range(n)]
    if colors is None:
        colors = [None] * n     # let matplotlib pick
    if styles is None:
        styles = ['-'] * n
    
    for i, curve in enumerate(curves):
        x, y = np.asarray(curve[0]), np.asarray(curve[1])
        if len(x) == 0 or len(y) == 0:
            continue
        ax.plot(x, y, linestyle=styles[i], color=colors[i],
                label=labels[i], linewidth=1.8)
        if show_points:
            ax.plot(x, y, 'o', color=colors[i], markersize=3, alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)
    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    
    plt.tight_layout()
    return ax