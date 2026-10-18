"""Alternative OverparametrisedNet-inspired styling for the PBE/HSE/QM9 plots.

The data loading and plotting primitives remain in ``plot-pbe-hse-qm9.py``.
This script applies the layout and typography hierarchy used by
``OverparametrisedNet/figs/Figure1.pdf`` without changing the original script.
"""

import importlib.util
from pathlib import Path

from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / 'plot-pbe-hse-qm9.py'


def load_base_module():
    spec = importlib.util.spec_from_file_location('plot_pbe_hse_qm9_base', BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load plotting module: {BASE_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()

PLOT_SELECTION = base.PLOT_SELECTION
PAPER_FIG_DIR = base.PAPER_FIG_DIR

# Figure 1 in OverparametrisedNet uses a 6.8 x 2.2 inch, three-panel row.
STYLE_REFERENCE_SIZE = (6.8, 2.2)
STYLE_LEFT = 0.08
STYLE_RIGHT = 0.985
STYLE_BOTTOM = 0.22
STYLE_TOP = 0.82
STYLE_WSPACE = 0.25

STYLE_LEFT_MARGIN = STYLE_REFERENCE_SIZE[0] * STYLE_LEFT
STYLE_RIGHT_MARGIN = STYLE_REFERENCE_SIZE[0] * (1.0 - STYLE_RIGHT)
STYLE_BOTTOM_MARGIN = STYLE_REFERENCE_SIZE[1] * STYLE_BOTTOM
STYLE_TOP_MARGIN = STYLE_REFERENCE_SIZE[1] * (1.0 - STYLE_TOP)
STYLE_PANEL_WIDTH = (
    STYLE_REFERENCE_SIZE[0] * (STYLE_RIGHT - STYLE_LEFT)
    / (3.0 + 2.0 * STYLE_WSPACE)
)
STYLE_PANEL_HEIGHT = STYLE_REFERENCE_SIZE[1] * (STYLE_TOP - STYLE_BOTTOM)
# Uniform 10 pt labels require slightly more room than the 8/7 pt reference.
STYLE_PANEL_GAP = STYLE_WSPACE * STYLE_PANEL_WIDTH + 0.15

STYLE_FIG1_SIZE = (
    STYLE_LEFT_MARGIN
    + 3.0 * STYLE_PANEL_WIDTH
    + 2.0 * STYLE_PANEL_GAP
    + STYLE_RIGHT_MARGIN,
    STYLE_REFERENCE_SIZE[1],
)

STYLE_FIG2_SIZE = (
    STYLE_LEFT_MARGIN
    + 4.0 * STYLE_PANEL_WIDTH
    + 3.0 * STYLE_PANEL_GAP
    + STYLE_RIGHT_MARGIN,
    STYLE_REFERENCE_SIZE[1],
)

FONT_SIZE = 10.0
SPINE_LW = 1.20
TICK_LW = 1.20
DATA_LW = 1.5

OVERPARAM_COLORS = [
    '#003f5c',
    '#2f4b7c',
    '#665191',
    '#a05195',
    '#d45087',
    '#f95d6a',
    '#ff7c43',
    '#ffa600',
]
MODEL_COLORS = ['#003f5c', '#40B0A7', '#FFBE6A']
L_COLORS = [
    OVERPARAM_COLORS[0],
    OVERPARAM_COLORS[2],
    OVERPARAM_COLORS[3],
    OVERPARAM_COLORS[5],
    OVERPARAM_COLORS[7],
]


def apply_overparam_style():
    base.FTSZFIG1 = FONT_SIZE
    base.FTSZ = FONT_SIZE
    base.LABELSZ = FONT_SIZE
    base.TICKSZ = FONT_SIZE
    base.LEGENDSZ = FONT_SIZE
    base.PANEL_FTSZ = FONT_SIZE
    base.AXIS_LW = SPINE_LW
    base.HIST_EDGE_LW = SPINE_LW

    base.L_COLORS = L_COLORS
    base.MODEL_COLORS = dict(zip(base.MODEL_ORDER, MODEL_COLORS))
    base.DIST_TRAIN_COLOR = '#003f5c'
    base.DIST_VAL_COLOR = '#40B0A7'
    base.CHARGE_TRAIN_COLOR = '#003f5c'
    base.CHARGE_VAL_COLOR = '#FFBE6A'
    base.HSE_ZERO_COLOR = '#000000'
    base.HSE_SCR_TRAIN_COLOR = '#40B0A7'
    base.HSE_FT_TRAIN_COLOR = '#FFBE6A'
    base.TIME_BATCH_COLOR = MODEL_COLORS[0]
    base.MODEL_PARAM_COLOR = MODEL_COLORS[2]

    base.mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        'axes.linewidth': SPINE_LW,
        'lines.linewidth': DATA_LW,
        'xtick.major.width': TICK_LW,
        'ytick.major.width': TICK_LW,
        'xtick.minor.width': TICK_LW,
        'ytick.minor.width': TICK_LW,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'text.usetex': False,
    })


def style_overparam_axis(ax):
    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        top=True,
        bottom=True,
        left=True,
        right=True,
        labeltop=False,
        labelright=False,
        width=TICK_LW,
        length=3.5,
        labelsize=FONT_SIZE,
        zorder=21,
    )
    ax.tick_params(
        axis='both',
        which='minor',
        direction='in',
        top=True,
        bottom=True,
        left=True,
        right=True,
        width=TICK_LW,
        length=2.0,
        zorder=21,
    )
    ax.xaxis.get_offset_text().set_fontsize(FONT_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)
    ax.xaxis.set_zorder(21)
    ax.yaxis.set_zorder(21)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LW)
        spine.set_zorder(20)


def create_style_row_axes(fig, ncols):
    fig_width, fig_height = fig.get_size_inches()
    panel_left = STYLE_LEFT_MARGIN / fig_width
    bottom = STYLE_BOTTOM_MARGIN / fig_height
    panel_w = STYLE_PANEL_WIDTH / fig_width
    panel_h = STYLE_PANEL_HEIGHT / fig_height
    gap = STYLE_PANEL_GAP / fig_width

    axes = []
    for _ in range(ncols):
        axes.append(fig.add_axes([panel_left, bottom, panel_w, panel_h]))
        panel_left += panel_w + gap
    return axes


def emphasize_data_lines(ax, linewidth=DATA_LW):
    for line in ax.get_lines():
        if str(line.get_linestyle()).lower() not in {'none', ' ', ''}:
            line.set_linewidth(linewidth)
    legend = ax.get_legend()
    if legend is not None:
        for line in legend.get_lines():
            line.set_linewidth(linewidth)


def add_panel_label(ax, label):
    ax.text(
        -0.15,
        1.035,
        f'{label})',
        transform=ax.transAxes,
        fontsize=FONT_SIZE,
        ha='left',
        va='bottom',
        clip_on=False,
    )


def plot_inference_time_panel(ax, inference_stats, show_xlabel=True):
    x = base.np.arange(len(base.L_TICK_LABELS), dtype=float)
    bar_width = 0.24
    all_means = inference_stats['mean_seconds'].to_numpy(dtype=float)
    all_stds = base.np.nan_to_num(
        inference_stats['std_seconds'].to_numpy(dtype=float),
        nan=0.0,
    )
    raw_lower_edges = all_means - all_stds
    # A logarithmic axis cannot show non-positive mean - std values. Display
    # those lower ends one decade below the mean and keep every cap in-frame.
    display_lower_edges = base.np.where(
        raw_lower_edges > 0.0,
        raw_lower_edges,
        all_means * 0.10,
    )
    y_floor = display_lower_edges.min() * 0.75
    y_ceiling = base.np.max(all_means + all_stds) * 1.35

    for i, model in enumerate(base.MODEL_ORDER):
        model_stats = (
            inference_stats[inference_stats['model'] == model]
            .set_index('max_angular_order')
            .reindex(range(len(base.L_TICK_LABELS)))
        )
        means = model_stats['mean_seconds'].to_numpy(dtype=float)
        stds = base.np.nan_to_num(
            model_stats['std_seconds'].to_numpy(dtype=float),
            nan=0.0,
        )
        raw_lower_edges = means - stds
        display_lower_edges = base.np.where(
            raw_lower_edges > 0.0,
            raw_lower_edges,
            means * 0.10,
        )
        asymmetric_yerr = base.np.vstack([
            means - display_lower_edges,
            stds,
        ])
        positions = x + (i - 1) * bar_width
        ax.bar(
            positions,
            means - y_floor,
            bottom=y_floor,
            width=bar_width,
            color=base.MODEL_COLORS[model],
            label=base.MODEL_DISPLAY_LABELS[model],
            yerr=asymmetric_yerr,
            error_kw={
                'ecolor': 'black',
                'elinewidth': SPINE_LW,
                'capsize': 2.5,
                'capthick': SPINE_LW,
            },
            zorder=3,
        )

    base.style_model_comparison_axis(
        ax,
        'Infer. time [s/sample]',
        show_xlabel,
    )
    ax.set_ylim(y_floor, y_ceiling)


def create_figure_1(inference_stats, parameter_df, hse_data):
    fig = base.plt.figure(figsize=STYLE_FIG1_SIZE, dpi=400)
    ax_inference, ax_parameters, ax_hse = create_style_row_axes(fig, 3)

    plot_inference_time_panel(ax_inference, inference_stats, show_xlabel=True)
    base.plot_parameter_panel(ax_parameters, parameter_df, show_xlabel=True)
    base.plot_hse_panel(ax_hse, hse_data)
    ax_inference.xaxis.set_minor_locator(base.mticker.NullLocator())
    ax_parameters.xaxis.set_minor_locator(base.mticker.NullLocator())
    ax_hse.yaxis.labelpad = 1.0

    for ax in (ax_inference, ax_parameters, ax_hse):
        style_overparam_axis(ax)
    emphasize_data_lines(ax_hse)

    old_hse_legend = ax_hse.get_legend()
    if old_hse_legend is not None:
        old_hse_legend.remove()
    hse_handles = [
        Line2D([0], [0], color=base.HSE_ZERO_COLOR, lw=DATA_LW, linestyle='--', label='Zero-shot'),
        Line2D([0], [0], color=base.HSE_SCR_TRAIN_COLOR, lw=DATA_LW, linestyle='-', label='Scratch'),
        Line2D([0], [0], color=base.HSE_FT_TRAIN_COLOR, lw=DATA_LW, linestyle='-', label='FT'),
    ]
    hse_legend_center = (
        ax_hse.get_position().x0 + ax_hse.get_position().x1
    ) / 2.0
    fig.legend(
        handles=hse_handles,
        loc='upper center',
        bbox_to_anchor=(hse_legend_center, 1.005),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE,
        handlelength=1.0,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.0,
    )

    add_panel_label(ax_inference, 'a')
    add_panel_label(ax_parameters, 'b')
    add_panel_label(ax_hse, 'c')

    comparison_handles, comparison_labels = ax_inference.get_legend_handles_labels()
    comparison_center = (
        ax_inference.get_position().x0 + ax_parameters.get_position().x1
    ) / 2.0
    fig.legend(
        comparison_handles,
        comparison_labels,
        loc='upper center',
        bbox_to_anchor=(comparison_center, 1.005),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE,
        handlelength=1.2,
        handletextpad=0.40,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    return fig


def create_figure_2(pbe_loss_data, qm_loss_data, mp_distribution_data, qm9_distribution_data):
    fig = base.plt.figure(figsize=STYLE_FIG2_SIZE, dpi=400)
    ax_pbe, ax_qm, ax_mp_violin, ax_qm9_violin = create_style_row_axes(fig, 4)

    base.plot_lmax_loss(ax_pbe, pbe_loss_data, base.MP_RUN_NAMES, show_xlabel=True, show_legend=False, show_ylabel=True)
    base.plot_lmax_loss(ax_qm, qm_loss_data, base.QM9_RUN_NAMES, show_xlabel=True, show_legend=False, show_ylabel=True)
    base.plot_violin_panel(ax_mp_violin, mp_distribution_data, ylim=(0, 5))
    base.plot_violin_panel(ax_qm9_violin, qm9_distribution_data, ylim=(0, 2))
    ax_pbe.xaxis.set_minor_locator(base.mticker.NullLocator())
    ax_qm.xaxis.set_minor_locator(base.mticker.NullLocator())

    for ax in (ax_pbe, ax_qm, ax_mp_violin, ax_qm9_violin):
        style_overparam_axis(ax)
        emphasize_data_lines(ax)

    l_handles = [
        Line2D([0], [0], color=color, lw=DATA_LW, linestyle='-', label=label)
        for color, label in zip(base.L_COLORS, base.LABELS)
    ]
    loss_legend_center = (
        ax_pbe.get_position().x0 + ax_qm.get_position().x1
    ) / 2.0
    fig.legend(
        handles=l_handles,
        loc='upper center',
        bbox_to_anchor=(loss_legend_center, 1.005),
        ncol=5,
        frameon=False,
        fontsize=FONT_SIZE,
        handlelength=1.0,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.0,
    )

    add_panel_label(ax_pbe, 'a')
    add_panel_label(ax_qm, 'b')
    add_panel_label(ax_mp_violin, 'c')
    add_panel_label(ax_qm9_violin, 'd')

    return fig


def main():
    apply_overparam_style()
    base.PLOT_SELECTION = PLOT_SELECTION
    base.PAPER_FIG_DIR = PAPER_FIG_DIR
    base.style_four_side_axis = style_overparam_axis
    base.create_figure_1 = create_figure_1
    base.create_figure_2 = create_figure_2

    original_save_figure = base.save_figure

    def save_new_figure(fig, stem, tight=False):
        original_save_figure(fig, f'new-{stem}', tight=True)

    base.save_figure = save_new_figure
    base.main()


if __name__ == '__main__':
    main()
