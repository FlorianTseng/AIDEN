import gzip
import os
import re
import json
import colorsys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

PLOT_SELECTION = 'both'

FIG1_RIGHT_MARGIN = 0.20

# Preserve the physical panel geometry of the original 2 x 2 Figure 2 when
# rearranging its panels into horizontal rows.
ORIGINAL_FIG2_SIZE = (4.4, 2.5)
ORIGINAL_FIG2_LEFT = 0.11
ORIGINAL_FIG2_RIGHT = 0.98
ORIGINAL_FIG2_BOTTOM = 0.16
ORIGINAL_FIG2_TOP = 0.88
ORIGINAL_FIG2_WSPACE = 0.42
ORIGINAL_FIG2_HSPACE = 0.10

ROW_LEFT_MARGIN = ORIGINAL_FIG2_SIZE[0] * ORIGINAL_FIG2_LEFT
ROW_RIGHT_MARGIN = ORIGINAL_FIG2_SIZE[0] * (1.0 - ORIGINAL_FIG2_RIGHT)
ROW_BOTTOM_MARGIN = ORIGINAL_FIG2_SIZE[1] * ORIGINAL_FIG2_BOTTOM
ROW_TOP_MARGIN = ORIGINAL_FIG2_SIZE[1] * (1.0 - ORIGINAL_FIG2_TOP)
ROW_PANEL_WIDTH = (
    ORIGINAL_FIG2_SIZE[0] * (ORIGINAL_FIG2_RIGHT - ORIGINAL_FIG2_LEFT)
    / (2.0 + ORIGINAL_FIG2_WSPACE)
)
ROW_PANEL_HEIGHT = (
    ORIGINAL_FIG2_SIZE[1] * (ORIGINAL_FIG2_TOP - ORIGINAL_FIG2_BOTTOM)
    / (2.0 + ORIGINAL_FIG2_HSPACE)
)
ROW_PANEL_GAP = ORIGINAL_FIG2_WSPACE * ROW_PANEL_WIDTH

FIG1_SIZE = (
    ROW_LEFT_MARGIN + 3.0 * ROW_PANEL_WIDTH + 2.0 * ROW_PANEL_GAP + FIG1_RIGHT_MARGIN,
    ROW_BOTTOM_MARGIN + ROW_PANEL_HEIGHT + ROW_TOP_MARGIN,
)
FIG2_SIZE = (
    ROW_LEFT_MARGIN + 4.0 * ROW_PANEL_WIDTH + 3.0 * ROW_PANEL_GAP + ROW_RIGHT_MARGIN,
    ROW_BOTTOM_MARGIN + ROW_PANEL_HEIGHT + ROW_TOP_MARGIN,
)
CHARGE_FIG_SIZE = ORIGINAL_FIG2_SIZE

BASE_DIR = '.'

MP_RUN_NAMES = ['aiden-pbe-l0', 'aiden-pbe-l1', 'aiden-pbe-l2', 'aiden-pbe-l3', 'aiden-pbe-l4']
QM9_RUN_NAMES = ['aiden-qm9-l0', 'aiden-qm9-l1', 'aiden-qm9-l2', 'aiden-qm9-l3', 'aiden-qm9-l4']

MP_NMAE_CSV = os.path.join(BASE_DIR, 'mp_train_val_lmax_nmae.csv')
QM9_NMAE_CSV = os.path.join(BASE_DIR, 'qm9_train_val_lmax_nmae.csv')
MP_CHARGE_CSV = os.path.join(BASE_DIR, 'mp_train_val_lmax_charge_5000.csv')
QM9_CHARGE_CSV = os.path.join(BASE_DIR, 'qm9_train_val_lmax_charge_5000.csv')
INFERENCE_TIMES_CSV = os.path.join(BASE_DIR, 'per_sample_inference_times.csv')
PARAM_EFFICIENCY_CSV = os.path.join(BASE_DIR, 'param_efficiency_comparison.csv')

HSE_BASE_DIR = BASE_DIR
HSE_ZERO_FILE = os.path.join(HSE_BASE_DIR, 'o-zero')
HSE_SCR_FILE = os.path.join(HSE_BASE_DIR, 'o-scr')
HSE_ALL_FILE = os.path.join(HSE_BASE_DIR, 'o-all')

PAPER_FIG_DIR = '/Users/yuxuanzeng/Desktop/iclr-temp/figs'

FTSZFIG1 = 9
FTSZ = 9
LABELSZ = FTSZ
TICKSZ = FTSZ
LEGENDSZ = 9
PANEL_FTSZ = FTSZ
AXIS_LW = 1.2

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = FTSZ
mpl.rcParams['axes.labelsize'] = LABELSZ
mpl.rcParams['xtick.labelsize'] = TICKSZ
mpl.rcParams['ytick.labelsize'] = TICKSZ
mpl.rcParams['legend.fontsize'] = LEGENDSZ
mpl.rcParams['axes.linewidth'] = AXIS_LW
mpl.rcParams['lines.linewidth'] = AXIS_LW
mpl.rcParams['xtick.major.width'] = AXIS_LW
mpl.rcParams['ytick.major.width'] = AXIS_LW
mpl.rcParams['xtick.minor.width'] = AXIS_LW
mpl.rcParams['ytick.minor.width'] = AXIS_LW
# mpl.rcParams['xtick.major.size'] = 3.0
# mpl.rcParams['ytick.major.size'] = 3.0
# mpl.rcParams['xtick.minor.size'] = 1.5
# mpl.rcParams['ytick.minor.size'] = 1.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['text.usetex'] = False

LABELS = [r'$L=0$', r'$L=1$', r'$L=2$', r'$L=3$', r'$L=4$']
L_TICK_LABELS = ['0', '1', '2', '3', '4']
PALETTE = ['#00487C', '#0CA4A5', '#98561F', '#FFBA08']
L_COLORS = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], '#000000']
MODEL_ORDER = ['AIDEN', 'Charge3Net', 'BOA']
MODEL_COLORS = dict(zip(MODEL_ORDER, PALETTE[:3]))
MODEL_DISPLAY_LABELS = {
    'AIDEN': 'AIDEN',
    'Charge3Net': 'ChargE3Net',
    'BOA': 'BOA',
}

DIST_TRAIN_COLOR = PALETTE[0]
DIST_VAL_COLOR = PALETTE[1]
CHARGE_TRAIN_COLOR = PALETTE[0]
CHARGE_VAL_COLOR = PALETTE[3]
HIST_EDGE_LW = AXIS_LW

HSE_ZERO_COLOR = PALETTE[2]
HSE_SCR_TRAIN_COLOR = PALETTE[0]
HSE_FT_TRAIN_COLOR = PALETTE[3]
TIME_BATCH_COLOR = PALETTE[0]
MODEL_PARAM_COLOR = PALETTE[3]

FLOAT = r'(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)'
RESUME_PATTERN = re.compile(r'Full resume at step=(\d+)', re.IGNORECASE)
TRAIN_PATTERN = re.compile(rf'Step\s+(\d+)\s+\|.*?epoch=\d+.*?train_mae=({FLOAT})\s+train_nmae=({FLOAT})', re.IGNORECASE)
VAL_PATTERN = re.compile(rf'Step\s+(\d+)\s+\|\s*train_mae=({FLOAT})\s+train_nmae=({FLOAT})\s+\|\s+val_mae=({FLOAT})\s+val_nmae=({FLOAT})', re.IGNORECASE)
TIMED_TRAIN_PATTERN = re.compile(rf'Step\s+(\d+)\s+\|.*?epoch=\d+.*?train_nmae=({FLOAT}).*?\|\s*([0-9]+(?:\.[0-9]+)?)s\s*$', re.IGNORECASE)
PARAM_PATTERN = re.compile(r'params_total=(\d+)', re.IGNORECASE)


def find_log(base_dir, run_name):
    """Return the run's train.log path, preferring plain text over .gz."""
    plain = os.path.join(base_dir, run_name, 'train.log')
    if os.path.exists(plain):
        return plain
    return os.path.join(base_dir, run_name, 'train.log.gz')


def open_log(log_path):
    """Open a training log, transparently handling gzip."""
    if log_path.endswith('.gz'):
        return gzip.open(log_path, 'rt', encoding='utf-8', errors='ignore')
    return open(log_path, 'r', encoding='utf-8', errors='ignore')

def desaturate(color, saturation_scale=0.42):
    rgb = mcolors.to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return colorsys.hsv_to_rgb(h, s * saturation_scale, v)

def valid_value(value):
    return np.isfinite(value) and value > 0.0

def parse_lmax_log(log_path):
    train_records = {}
    val_records = {}

    with open_log(log_path) as f:
        for line in f:
            resume_match = RESUME_PATTERN.search(line)
            if resume_match is not None:
                resume_step = int(resume_match.group(1))
                train_records = {step: value for step, value in train_records.items() if step <= resume_step}
                val_records = {step: value for step, value in val_records.items() if step <= resume_step}
                continue

            train_match = TRAIN_PATTERN.search(line)
            if train_match is not None:
                step = int(train_match.group(1))
                train_nmae = float(train_match.group(3))
                if step > 0 and valid_value(train_nmae):
                    train_records[step] = train_nmae
                continue

            val_match = VAL_PATTERN.search(line)
            if val_match is not None:
                step = int(val_match.group(1))
                train_nmae = float(val_match.group(3))
                val_nmae = float(val_match.group(5))
                if step > 0 and valid_value(train_nmae) and valid_value(val_nmae):
                    val_records[step] = (train_nmae, val_nmae)

    train_steps = np.array(sorted(train_records.keys()), dtype=float)
    train_nmae = np.array([train_records[int(step)] for step in train_steps], dtype=float)
    val_steps = np.array(sorted(val_records.keys()), dtype=float)
    val_train_nmae = np.array([val_records[int(step)][0] for step in val_steps], dtype=float)
    val_nmae = np.array([val_records[int(step)][1] for step in val_steps], dtype=float)

    if len(train_steps) == 0:
        raise RuntimeError(f'No valid training NMAE found in: {log_path}')
    if len(val_steps) == 0:
        raise RuntimeError(f'No valid validation NMAE found in: {log_path}')

    best_idx = np.argmin(val_nmae)
    best_step = val_steps[best_idx]
    best_train_nmae = val_train_nmae[best_idx]
    best_val_nmae = val_nmae[best_idx]

    return {'train_steps': train_steps, 'train_nmae': train_nmae * 100.0, 'val_steps': val_steps, 'val_nmae': val_nmae * 100.0, 'best_step': best_step, 'best_train_nmae': best_train_nmae * 100.0, 'best_val_nmae': best_val_nmae * 100.0}

def load_lmax_loss_data(base_dir, run_names):
    data = {}

    for run_name in run_names:
        log_path = find_log(base_dir, run_name)
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f'Cannot find log file: {log_path}')

        data[run_name] = parse_lmax_log(log_path)
        d = data[run_name]
        print(f'{run_name:16s} | train NMAE = {d["best_train_nmae"]:.6f}% | val NMAE = {d["best_val_nmae"]:.6f}% | step = {int(d["best_step"])}')

    return data

def load_efficiency_data(base_dir, run_names):
    data = {}

    for run_name in run_names:
        log_path = find_log(base_dir, run_name)
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f'Cannot find log file: {log_path}')

        time_records = {}
        params_total = None
        with open_log(log_path) as f:
            for line in f:
                param_match = PARAM_PATTERN.search(line)
                if param_match is not None:
                    value = int(param_match.group(1))
                    if params_total is not None and value != params_total:
                        raise RuntimeError(f'Inconsistent parameter counts found in: {log_path}')
                    params_total = value

                resume_match = RESUME_PATTERN.search(line)
                if resume_match is not None:
                    resume_step = int(resume_match.group(1))
                    time_records = {step: value for step, value in time_records.items() if step <= resume_step}
                    continue

                time_match = TIMED_TRAIN_PATTERN.search(line)
                if time_match is not None:
                    step = int(time_match.group(1))
                    duration = float(time_match.group(3))
                    if step > 0 and duration > 0.0:
                        time_records[step] = duration

        steps = np.array(sorted(time_records.keys()), dtype=float)
        durations = np.array([time_records[int(step)] for step in steps], dtype=float)
        if len(steps) < 2:
            raise RuntimeError(f'No valid timed training records found in: {log_path}')
        if params_total is None:
            raise RuntimeError(f'No model parameter count found in: {log_path}')

        batch_seconds = float(np.mean(durations))
        params_million = params_total / 1e6
        data[run_name] = {'batch_seconds': batch_seconds, 'params_total': params_total, 'params_million': params_million}
        print(f'{run_name:16s} | batch time = {batch_seconds:.4f}s | parameters = {params_million:.6f} million')

    return data

def validate_model_l_grid(df, csv_path):
    expected_pairs = {
        (model, l_value)
        for model in MODEL_ORDER
        for l_value in range(len(L_TICK_LABELS))
    }
    actual_pairs = set(zip(df['model'], df['max_angular_order']))
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise ValueError(f'Unexpected model/L grid in {csv_path}; missing={missing}, extra={extra}')

def load_inference_time_stats(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Cannot find CSV file: {csv_path}')

    df = pd.read_csv(csv_path)
    required = {'model', 'max_angular_order', 'inference_ms'}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f'Missing columns in {csv_path}: {sorted(missing_columns)}')

    df = df.copy()
    df['model'] = df['model'].astype(str).str.strip()
    df['max_angular_order'] = pd.to_numeric(df['max_angular_order'], errors='coerce')
    df['inference_seconds'] = pd.to_numeric(df['inference_ms'], errors='coerce') / 1000.0
    if df[['max_angular_order', 'inference_seconds']].isna().any().any():
        raise ValueError(f'Non-numeric inference data found in {csv_path}')
    if (df['inference_seconds'] <= 0.0).any():
        raise ValueError(f'Inference times must be positive in {csv_path}')

    df['max_angular_order'] = df['max_angular_order'].astype(int)
    validate_model_l_grid(df, csv_path)

    return (
        df.groupby(['model', 'max_angular_order'], as_index=False)
        .agg(
            mean_seconds=('inference_seconds', 'mean'),
            std_seconds=('inference_seconds', 'std'),
            sample_count=('inference_seconds', 'size'),
        )
    )

def load_parameter_comparison(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Cannot find CSV file: {csv_path}')

    df = pd.read_csv(csv_path)
    required = {'model', 'max_angular_order', 'parameter_count_total'}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f'Missing columns in {csv_path}: {sorted(missing_columns)}')

    df = df.copy()
    df['model'] = df['model'].astype(str).str.strip()
    df['max_angular_order'] = pd.to_numeric(df['max_angular_order'], errors='coerce')
    df['parameter_count_total'] = pd.to_numeric(df['parameter_count_total'], errors='coerce')
    if df[['max_angular_order', 'parameter_count_total']].isna().any().any():
        raise ValueError(f'Non-numeric parameter data found in {csv_path}')
    if (df['parameter_count_total'] <= 0.0).any():
        raise ValueError(f'Parameter counts must be positive in {csv_path}')

    df['max_angular_order'] = df['max_angular_order'].astype(int)
    validate_model_l_grid(df, csv_path)
    if df.duplicated(['model', 'max_angular_order']).any():
        raise ValueError(f'Duplicate model/L rows found in {csv_path}')
    return df

def load_distribution_data(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Cannot find CSV file: {csv_path}')

    df = pd.read_csv(csv_path, sep=None, engine='python')
    df['split'] = df['split'].astype(str).str.strip().str.lower()
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    distribution_data = {}

    for i in range(len(LABELS)):
        column = f'l{i}_nmae'
        train_values = pd.to_numeric(train_df[column], errors='coerce').to_numpy(dtype=float)
        val_values = pd.to_numeric(val_df[column], errors='coerce').to_numpy(dtype=float)
        train_values = train_values[np.isfinite(train_values) & (train_values >= 0.0)] * 100.0
        val_values = val_values[np.isfinite(val_values) & (val_values >= 0.0)] * 100.0

        if len(train_values) < 2 or len(val_values) < 2:
            raise RuntimeError(f'Not enough samples for {column} in {csv_path}')

        distribution_data[i] = {'train': train_values, 'val': val_values}

    return distribution_data

def load_charge_df(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Cannot find CSV file: {csv_path}')
    return pd.read_csv(csv_path, sep=None, engine='python')

def draw_half_violin_outline(ax, data, center, side, color, half_width=0.34, lw=1.0):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if len(data) < 2:
        return

    dmin = max(0.0, data.min())
    dmax = data.max()

    if dmax <= dmin:
        ax.plot([center, center], [dmin, dmax + 1e-6], color=color, lw=lw, zorder=2)
        return

    pad = 0.08 * (dmax - dmin)
    y = np.linspace(max(0.0, dmin - pad), dmax + pad, 256)
    kde = gaussian_kde(data)
    density = kde(y)

    if np.allclose(density.max(), 0.0):
        return

    density = density / density.max() * half_width
    x = center - density if side == 'left' else center + density
    ax.plot(x, y, color=color, lw=lw, zorder=2)

def calc_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if len(y_true) > 1 and ss_tot > 0 else np.nan

def style_four_side_axis(ax):
    ax.tick_params(axis='both', which='major', direction='in', top=True, bottom=True, left=True, right=True, labeltop=False, labelright=False, width=AXIS_LW, length=3.0)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, bottom=True, left=True, right=True, width=AXIS_LW)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LW)

def read_zero_val_nmae(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    objects = []

    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1

        if idx >= len(text):
            break

        obj, end = decoder.raw_decode(text, idx)
        objects.append(obj)
        idx = end

    for obj in objects:
        if obj.get('split') == 'val':
            return float(obj['nmae'])

    raise ValueError('Cannot find split="val" in o-zero')

def read_hse_loss_log(path):
    train_steps = []
    train_nmae = []
    val_steps = []
    val_nmae = []

    train_pattern = re.compile(r'Step\s+(\d+).*?train_nmae=([0-9.eE+-]+)')
    val_pattern = re.compile(r'Step\s+(\d+).*?val_nmae=([0-9.eE+-]+)')

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'Step' not in line:
                continue

            train_match = train_pattern.search(line)
            val_match = val_pattern.search(line)

            if train_match:
                step = int(train_match.group(1))
                value = float(train_match.group(2))
                if not train_steps or step != train_steps[-1]:
                    train_steps.append(step)
                    train_nmae.append(value)

            if val_match:
                step = int(val_match.group(1))
                value = float(val_match.group(2))
                if not val_steps or step != val_steps[-1]:
                    val_steps.append(step)
                    val_nmae.append(value)

    return np.asarray(train_steps), np.asarray(train_nmae), np.asarray(val_steps), np.asarray(val_nmae)

def load_hse_loss_data():
    for path in [HSE_ZERO_FILE, HSE_SCR_FILE, HSE_ALL_FILE]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Cannot find HSE loss file: {path}')

    zero_nmae = read_zero_val_nmae(HSE_ZERO_FILE)
    scr_train_x, scr_train_y, scr_val_x, scr_val_y = read_hse_loss_log(HSE_SCR_FILE)
    ft_train_x, ft_train_y, ft_val_x, ft_val_y = read_hse_loss_log(HSE_ALL_FILE)

    if len(scr_val_y) == 0 or len(ft_val_y) == 0:
        raise RuntimeError('Cannot find valid HSE validation loss records.')

    scr_best_idx = np.argmin(scr_val_y)
    ft_best_idx = np.argmin(ft_val_y)

    data = {
        'zero_nmae': zero_nmae,
        'scr_train_x': scr_train_x,
        'scr_train_y': scr_train_y,
        'scr_val_x': scr_val_x,
        'scr_val_y': scr_val_y,
        'scr_best_x': scr_val_x[scr_best_idx],
        'scr_best_y': scr_val_y[scr_best_idx],
        'ft_train_x': ft_train_x,
        'ft_train_y': ft_train_y,
        'ft_val_x': ft_val_x,
        'ft_val_y': ft_val_y,
        'ft_best_x': ft_val_x[ft_best_idx],
        'ft_best_y': ft_val_y[ft_best_idx],
    }

    data['max_step'] = max(scr_train_x.max(), scr_val_x.max(), ft_train_x.max(), ft_val_x.max())

    print(f'zero-shot val NMAE: {zero_nmae:.8f}')
    print(f'scratch best val NMAE: {data["scr_best_y"]:.8f} at step {data["scr_best_x"]}')
    print(f'fine-tuning best val NMAE: {data["ft_best_y"]:.8f} at step {data["ft_best_x"]}')

    return data

def plot_lmax_loss(ax, loss_data, run_names, show_xlabel=True, show_legend=True, show_ylabel=False):
    train_colors = L_COLORS
    val_colors = L_COLORS

    for i, run_name in enumerate(run_names):
        d = loss_data[run_name]
        ax.plot(d['train_steps'], d['train_nmae'], color=train_colors[i], linestyle='-', lw=AXIS_LW, alpha=0.5, zorder=2 + i)

    for i, run_name in enumerate(run_names):
        d = loss_data[run_name]
        ax.plot(d['val_steps'], d['val_nmae'], color=val_colors[i], linestyle='--', lw=AXIS_LW, alpha=1.0, zorder=20 + i)

    for i, run_name in enumerate(run_names):
        d = loss_data[run_name]
        ax.plot(d['best_step'], d['best_val_nmae'], marker='*', linestyle='None', markersize=5.0, markerfacecolor=val_colors[i], markeredgecolor='k', markeredgewidth=1.0, zorder=40 + i, clip_on=False)

    ax.set_yscale('log')
    ax.set_xlim(0, 2e6)

    if show_ylabel:
        ax.set_ylabel(r'NMAE $\varepsilon_{\rho}$ [%]', fontsize=FTSZFIG1)
    else:
        ax.set_ylabel('')

    if show_xlabel:
        ax.set_xlabel(r'Training Step', fontsize=FTSZFIG1)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', which='both', labelbottom=False)

    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.ticklabel_format(style='sci', axis='x', scilimits=(6, 6), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(FTSZFIG1)
    ax.xaxis.get_offset_text().set_visible(show_xlabel)
    ax.tick_params(axis='both', which='major', labelsize=FTSZFIG1)
    style_four_side_axis(ax)

    if show_legend:
        model_handles = [Line2D([0], [0], color=color, lw=AXIS_LW, linestyle='-', label=label) for color, label in zip(val_colors, LABELS)]
        ax.legend(handles=model_handles, loc='upper center', ncol=5, frameon=False, fontsize=LEGENDSZ, handlelength=0.70, handletextpad=0.15, columnspacing=0.60, labelspacing=0.10, borderaxespad=0.10)

def plot_efficiency_panel(ax, efficiency_data, run_names):
    x = np.arange(len(run_names))
    bar_width = 0.34
    batch_seconds = np.array([efficiency_data[run_name]['batch_seconds'] for run_name in run_names])
    params_million = np.array([efficiency_data[run_name]['params_million'] for run_name in run_names])

    ax.bar(x - bar_width / 2.0, batch_seconds, width=bar_width, color=TIME_BATCH_COLOR, zorder=3)
    ax_params = ax.twinx()
    ax_params.bar(x + bar_width / 2.0, params_million, width=bar_width, color=MODEL_PARAM_COLOR, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(L_TICK_LABELS, fontsize=FTSZFIG1)
    ax.set_xlim(-0.55, len(run_names) - 0.45)
    ax.set_ylim(0.0, batch_seconds.max() * 1.12)
    ax.set_xlabel(r'Max. angular order $L$', fontsize=FTSZFIG1)
    ax.set_ylabel('Batch time [s]', fontsize=FTSZFIG1, color=TIME_BATCH_COLOR)

    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.tick_params(axis='x', which='major', direction='in', top=True, bottom=True, width=AXIS_LW, labelsize=FTSZFIG1)
    ax.tick_params(axis='x', which='minor', top=False, bottom=False)
    ax.tick_params(axis='y', which='major', direction='in', left=True, right=False, width=AXIS_LW, labelsize=FTSZFIG1, colors=TIME_BATCH_COLOR)
    ax.tick_params(axis='y', which='minor', direction='in', left=True, right=False, width=AXIS_LW, colors=TIME_BATCH_COLOR)

    ax_params.set_ylim(0.0, params_million.max() * 1.12)
    ax_params.set_ylabel('Parameters [million]', fontsize=FTSZFIG1, color=MODEL_PARAM_COLOR)
    ax_params.xaxis.set_minor_locator(mticker.NullLocator())
    ax_params.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax_params.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_params.tick_params(axis='y', which='major', direction='in', left=False, right=True, width=AXIS_LW, labelsize=FTSZFIG1, colors=MODEL_PARAM_COLOR)
    ax_params.tick_params(axis='y', which='minor', direction='in', left=False, right=True, width=AXIS_LW, colors=MODEL_PARAM_COLOR)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(AXIS_LW)
    ax.spines['left'].set_color(TIME_BATCH_COLOR)
    ax.spines['bottom'].set_linewidth(AXIS_LW)
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_linewidth(AXIS_LW)
    ax_params.spines['left'].set_visible(False)
    ax_params.spines['bottom'].set_visible(False)
    ax_params.spines['top'].set_visible(False)
    ax_params.spines['right'].set_linewidth(AXIS_LW)
    ax_params.spines['right'].set_color(MODEL_PARAM_COLOR)

    return ax_params

def style_model_comparison_axis(ax, ylabel, show_xlabel):
    x = np.arange(len(L_TICK_LABELS))
    ax.set_xlim(-0.55, len(L_TICK_LABELS) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(L_TICK_LABELS, fontsize=FTSZFIG1)
    ax.set_ylabel(ylabel, fontsize=FTSZFIG1)
    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    if show_xlabel:
        ax.set_xlabel(r'Max. angular order $L$', fontsize=FTSZFIG1)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', which='both', labelbottom=False)

    style_four_side_axis(ax)

def plot_inference_time_panel(ax, inference_stats, show_xlabel=False):
    x = np.arange(len(L_TICK_LABELS), dtype=float)
    bar_width = 0.24
    all_means = inference_stats['mean_seconds'].to_numpy(dtype=float)
    all_stds = inference_stats['std_seconds'].to_numpy(dtype=float)
    y_floor = all_means.min() * 0.5

    for i, model in enumerate(MODEL_ORDER):
        model_stats = (
            inference_stats[inference_stats['model'] == model]
            .set_index('max_angular_order')
            .reindex(range(len(L_TICK_LABELS)))
        )
        means = model_stats['mean_seconds'].to_numpy(dtype=float)
        stds = model_stats['std_seconds'].to_numpy(dtype=float)
        positions = x + (i - 1) * bar_width
        ax.bar(
            positions,
            means - y_floor,
            bottom=y_floor,
            width=bar_width,
            color=MODEL_COLORS[model],
            label=MODEL_DISPLAY_LABELS[model],
            yerr=stds,
            error_kw={'ecolor': 'black', 'elinewidth': AXIS_LW, 'capsize': 2.5, 'capthick': AXIS_LW},
            zorder=3,
        )

    ax.set_ylim(y_floor, np.max(all_means + all_stds) * 2.0)
    style_model_comparison_axis(ax, 'Infer. time [s/sample]', show_xlabel)

def plot_parameter_panel(ax, parameter_df, show_xlabel=True):
    x = np.arange(len(L_TICK_LABELS), dtype=float)
    bar_width = 0.24
    all_parameters = parameter_df['parameter_count_total'].to_numpy(dtype=float)
    y_floor = all_parameters.min() * 0.5

    for i, model in enumerate(MODEL_ORDER):
        model_parameters = (
            parameter_df[parameter_df['model'] == model]
            .set_index('max_angular_order')['parameter_count_total']
            .reindex(range(len(L_TICK_LABELS)))
            .to_numpy(dtype=float)
        )
        positions = x + (i - 1) * bar_width
        ax.bar(
            positions,
            model_parameters - y_floor,
            bottom=y_floor,
            width=bar_width,
            color=MODEL_COLORS[model],
            label=MODEL_DISPLAY_LABELS[model],
            zorder=3,
        )

    ax.set_ylim(y_floor, all_parameters.max() * 1.5)
    style_model_comparison_axis(ax, 'Parameters', show_xlabel)

def plot_violin_panel(ax, distribution_data, ylim=(0, 5)):
    x = np.arange(len(LABELS))
    violin_half_width = 0.34
    box_offset = 0.095
    box_width = 0.10

    for i in range(len(LABELS)):
        train_values = distribution_data[i]['train'].copy()
        val_values = distribution_data[i]['val'].copy()

        if len(train_values) > 1500:
            train_values = np.delete(train_values, np.argpartition(train_values, -1500)[-1500:])
        if len(val_values) > 150:
            val_values = np.delete(val_values, np.argpartition(val_values, -150)[-150:])

        draw_half_violin_outline(ax, train_values, x[i], 'left', DIST_TRAIN_COLOR, half_width=violin_half_width, lw=AXIS_LW)
        draw_half_violin_outline(ax, val_values, x[i], 'right', DIST_VAL_COLOR, half_width=violin_half_width, lw=AXIS_LW)

        ax.boxplot(train_values, positions=[x[i] - box_offset], widths=box_width, patch_artist=True, showfliers=False, manage_ticks=False, zorder=4, whis=[0, 95], boxprops=dict(facecolor=DIST_TRAIN_COLOR, edgecolor=DIST_TRAIN_COLOR, linewidth=AXIS_LW), whiskerprops=dict(color=DIST_TRAIN_COLOR, linewidth=AXIS_LW), capprops=dict(color=DIST_TRAIN_COLOR, linewidth=AXIS_LW), medianprops=dict(color='#FFFFFF', linewidth=AXIS_LW))
        ax.boxplot(val_values, positions=[x[i] + box_offset], widths=box_width, patch_artist=True, showfliers=False, manage_ticks=False, zorder=4, whis=[0, 95], boxprops=dict(facecolor=DIST_VAL_COLOR, edgecolor=DIST_VAL_COLOR, linewidth=AXIS_LW), whiskerprops=dict(color=DIST_VAL_COLOR, linewidth=AXIS_LW), capprops=dict(color=DIST_VAL_COLOR, linewidth=AXIS_LW), medianprops=dict(color='#FFFFFF', linewidth=AXIS_LW))

    ax.set_xticks(x)
    ax.set_xticklabels(L_TICK_LABELS, fontsize=FTSZFIG1)
    ax.set_xlim(-0.50, len(LABELS) - 0.50)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'Max. angular order $L$', fontsize=FTSZFIG1)
    ax.set_ylabel(r'NMAE $\varepsilon_{\rho}$ [%]', fontsize=FTSZFIG1)

    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.tick_params(axis='x', which='major', direction='in', top=True, bottom=True, width=AXIS_LW, labelsize=FTSZFIG1)
    ax.tick_params(axis='x', which='minor', top=False, bottom=False)
    ax.tick_params(axis='y', which='major', direction='in', left=True, right=True, width=AXIS_LW, labelsize=FTSZFIG1)
    ax.tick_params(axis='y', which='minor', direction='in', left=True, right=True, width=AXIS_LW)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LW)

def plot_hse_panel(ax, hse_data, show_xlabel=True, show_ylabel=True):
    ax.axhline(hse_data['zero_nmae'], color=HSE_ZERO_COLOR, linewidth=AXIS_LW, linestyle='--', label='zero-shot', zorder=1)
    ax.plot(hse_data['scr_train_x'], hse_data['scr_train_y'], color=HSE_SCR_TRAIN_COLOR, linewidth=AXIS_LW, alpha=0.5, zorder=2)
    ax.plot(hse_data['scr_val_x'], hse_data['scr_val_y'], color=HSE_SCR_TRAIN_COLOR, linewidth=AXIS_LW, alpha=1.0, zorder=3)
    ax.scatter(hse_data['scr_best_x'], hse_data['scr_best_y'], s=45, marker='*', facecolors=HSE_SCR_TRAIN_COLOR, edgecolors='black', linewidths=AXIS_LW, zorder=6)
    ax.plot(hse_data['ft_train_x'], hse_data['ft_train_y'], color=HSE_FT_TRAIN_COLOR, linewidth=AXIS_LW, alpha=0.5, zorder=2)
    ax.plot(hse_data['ft_val_x'], hse_data['ft_val_y'], color=HSE_FT_TRAIN_COLOR, linewidth=AXIS_LW, alpha=1.0, zorder=3)
    ax.scatter(hse_data['ft_best_x'], hse_data['ft_best_y'], s=45, marker='*', facecolors=HSE_FT_TRAIN_COLOR, edgecolors='black', linewidths=AXIS_LW, zorder=6)

    ax.set_xlim(0, hse_data['max_step'])
    if show_xlabel:
        ax.set_xlabel('Step', fontsize=FTSZFIG1)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', which='both', labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(r'NMAE $\varepsilon_{\rho}$ [%]', fontsize=FTSZFIG1)
    else:
        ax.set_ylabel('')

    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.ticklabel_format(style='sci', axis='x', scilimits=(6, 6), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(FTSZFIG1)
    ax.tick_params(axis='both', which='major', labelsize=FTSZFIG1)
    style_four_side_axis(ax)

    legend_handles = [
        Line2D([0], [0], color=HSE_ZERO_COLOR, lw=AXIS_LW, linestyle='--', label='Zero-shot'),
        Line2D([0], [0], color=HSE_SCR_TRAIN_COLOR, lw=AXIS_LW, linestyle='-', label='Scratch'),
        Line2D([0], [0], color=HSE_FT_TRAIN_COLOR, lw=AXIS_LW, linestyle='-', label='FT')
    ]
    ax.legend(handles=legend_handles, loc='center right', ncol=1, frameon=False, fontsize=LEGENDSZ, handlelength=0.85, handletextpad=0.15, columnspacing=0.35, borderaxespad=0.10)

def create_original_fig2_row_axes(fig, ncols):
    fig_width, fig_height = fig.get_size_inches()
    panel_left = ROW_LEFT_MARGIN / fig_width
    bottom = ROW_BOTTOM_MARGIN / fig_height
    panel_w = ROW_PANEL_WIDTH / fig_width
    panel_h = ROW_PANEL_HEIGHT / fig_height
    gap = ROW_PANEL_GAP / fig_width

    axes = []
    for _ in range(ncols):
        axes.append(fig.add_axes([panel_left, bottom, panel_w, panel_h]))
        panel_left += panel_w + gap
    return axes

def add_inside_panel_label(ax, label):
    ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=PANEL_FTSZ, ha='left', va='top')

def create_figure_1(inference_stats, parameter_df, hse_data):
    fig = plt.figure(figsize=FIG1_SIZE)
    ax_inference, ax_parameters, ax_hse = create_original_fig2_row_axes(fig, 3)

    plot_inference_time_panel(ax_inference, inference_stats, show_xlabel=True)
    plot_parameter_panel(ax_parameters, parameter_df, show_xlabel=True)
    plot_hse_panel(ax_hse, hse_data)

    comparison_handles, comparison_labels = ax_inference.get_legend_handles_labels()
    comparison_center = (
        ax_inference.get_position().x0 + ax_parameters.get_position().x1
    ) / 2.0
    panel_top = ax_inference.get_position().y1
    fig.legend(
        comparison_handles,
        comparison_labels,
        loc='lower center',
        bbox_to_anchor=(comparison_center, panel_top + 0.11 / FIG1_SIZE[1]),
        ncol=3,
        frameon=False,
        fontsize=LEGENDSZ,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.80,
        borderaxespad=0.0,
    )

    add_inside_panel_label(ax_inference, 'a)')
    add_inside_panel_label(ax_parameters, 'b)')
    add_inside_panel_label(ax_hse, 'c)')

    return fig

def create_charge_panel_axes(fig, left, panel_w):
    scatter_left = left + 0.18 * panel_w
    scatter_bottom = 0.18
    scatter_w = 0.60 * panel_w
    scatter_h = 0.60

    histx_left = scatter_left
    histx_bottom = 0.805
    histx_w = 0.60 * panel_w
    histx_h = 0.15

    histy_left = left + 0.805 * panel_w
    histy_bottom = scatter_bottom
    histy_w = 0.15 * panel_w
    histy_h = 0.60

    ax_scatter = fig.add_axes([scatter_left, scatter_bottom, scatter_w, scatter_h])
    ax_histx = fig.add_axes([histx_left, histx_bottom, histx_w, histx_h], sharex=ax_scatter)
    ax_histy = fig.add_axes([histy_left, histy_bottom, histy_w, histy_h], sharey=ax_scatter)

    return ax_scatter, ax_histx, ax_histy

def prepare_charge_parity_data(charge_df):
    split_series = charge_df['split'].astype(str).str.strip().str.lower()
    pred_col = 'l4_charge_e'

    train_mask = split_series == 'train'
    val_mask = split_series == 'val'

    true_train = pd.to_numeric(charge_df.loc[train_mask, 'true_charge_e'], errors='coerce').to_numpy(dtype=float)
    pred_train = pd.to_numeric(charge_df.loc[train_mask, pred_col], errors='coerce').to_numpy(dtype=float)
    true_val = pd.to_numeric(charge_df.loc[val_mask, 'true_charge_e'], errors='coerce').to_numpy(dtype=float)
    pred_val = pd.to_numeric(charge_df.loc[val_mask, pred_col], errors='coerce').to_numpy(dtype=float)

    mask_train = np.isfinite(true_train) & np.isfinite(pred_train) & (true_train > 0) & (pred_train > 0)
    mask_val = np.isfinite(true_val) & np.isfinite(pred_val) & (true_val > 0) & (pred_val > 0)

    true_train = true_train[mask_train]
    pred_train = pred_train[mask_train]
    true_val = true_val[mask_val]
    pred_val = pred_val[mask_val]

    r2_trn = calc_r2(true_train, pred_train)
    r2_val = calc_r2(true_val, pred_val)

    all_charge = np.concatenate([true_train, pred_train, true_val, pred_val])
    charge_lo = all_charge.min() / 1.08
    charge_hi = all_charge.max() * 1.08
    charge_bins = np.logspace(np.log10(charge_lo), np.log10(charge_hi), 25)

    hist_true_train, _ = np.histogram(true_train, bins=charge_bins)
    hist_true_val, _ = np.histogram(true_val, bins=charge_bins)
    hist_pred_train, _ = np.histogram(pred_train, bins=charge_bins)
    hist_pred_val, _ = np.histogram(pred_val, bins=charge_bins)

    histx_max = max(10, hist_true_train.max() * 1.2, hist_true_val.max() * 1.2)
    histy_max = max(10, hist_pred_train.max() * 1.2, hist_pred_val.max() * 1.2)

    return {'true_train': true_train, 'pred_train': pred_train, 'true_val': true_val, 'pred_val': pred_val, 'r2_trn': r2_trn, 'r2_val': r2_val, 'charge_lo': charge_lo, 'charge_hi': charge_hi, 'charge_bins': charge_bins, 'histx_max': histx_max, 'histy_max': histy_max}

def style_full_parity_axis(ax):
    ax.tick_params(axis='both', which='major', direction='in', top=True, bottom=True, left=True, right=True, width=AXIS_LW, length=3.0)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, bottom=True, left=True, right=True, width=AXIS_LW)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LW)

def plot_charge_parity_panel(fig, left, panel_w, charge_df, panel_label):
    ax_charge, ax_histx, ax_histy = create_charge_panel_axes(fig, left, panel_w)
    d = prepare_charge_parity_data(charge_df)

    charge_lo = d['charge_lo']
    charge_hi = d['charge_hi']
    charge_bins = d['charge_bins']

    ax_charge.plot([charge_lo, charge_hi], [charge_lo, charge_hi], '--', lw=1.1, color='#888888', zorder=1)

    band_x = np.logspace(np.log10(charge_lo), np.log10(charge_hi), 400)
    ax_charge.fill_between(band_x, band_x * 0.995, band_x * 1.005, color='#DEDEDE', linewidth=0, zorder=0)

    ax_charge.scatter(d['true_train'], d['pred_train'], s=10, marker='o', facecolors=CHARGE_TRAIN_COLOR, edgecolors='k', linewidths=1.0, alpha=1.0, zorder=3)
    ax_charge.scatter(d['true_val'], d['pred_val'], s=10, marker='D', facecolors=CHARGE_VAL_COLOR, edgecolors='k', linewidths=1.0, alpha=1.0, zorder=4)

    r2_text = rf'$R^2_{{\rm TRN}}={d["r2_trn"]:.3f}$' + '\n' + rf'$R^2_{{\rm VAL}}={d["r2_val"]:.3f}$'
    ax_charge.text(0.05, 0.95, r2_text, transform=ax_charge.transAxes, fontsize=FTSZ, ha='left', va='top', linespacing=1.25, zorder=20)

    ax_charge.set_xscale('log')
    ax_charge.set_yscale('log')
    ax_charge.set_xlim(charge_lo, charge_hi)
    ax_charge.set_ylim(charge_lo, charge_hi)
    ax_charge.set_xlabel(r'$N_{\rm e}$ [Unitless]')
    ax_charge.set_ylabel(r'$\hat{N}_{\rm e}$ [Unitless]')
    ax_charge.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_charge.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_charge.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_charge.yaxis.set_minor_formatter(mticker.NullFormatter())
    style_full_parity_axis(ax_charge)

    ax_histx.hist(d['true_train'], bins=charge_bins, color=CHARGE_TRAIN_COLOR, edgecolor='k', linewidth=HIST_EDGE_LW, zorder=2)
    ax_histx.hist(d['true_val'], bins=charge_bins, color=CHARGE_VAL_COLOR, edgecolor='k', linewidth=HIST_EDGE_LW, zorder=3)

    ax_histx.set_xscale('log')
    ax_histx.set_yscale('log')
    ax_histx.set_xlim(charge_lo, charge_hi)
    ax_histx.set_ylim(0.8, d['histx_max'])
    ax_histx.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_histx.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_histx.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax_histx.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_histx.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax_histx.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_histx.tick_params(axis='x', which='both', labelbottom=False, labeltop=False)
    style_full_parity_axis(ax_histx)
    ax_histx.xaxis.set_zorder(40)
    ax_histx.yaxis.set_zorder(40)
    for spine in ax_histx.spines.values():
        spine.set_zorder(50)
    ax_histx.set_xlabel('')
    ax_histx.set_ylabel('')

    ax_histy.hist(d['pred_train'], bins=charge_bins, orientation='horizontal', color=CHARGE_TRAIN_COLOR, edgecolor='k', linewidth=HIST_EDGE_LW, zorder=2)
    ax_histy.hist(d['pred_val'], bins=charge_bins, orientation='horizontal', color=CHARGE_VAL_COLOR, edgecolor='k', linewidth=HIST_EDGE_LW, zorder=3)

    ax_histy.set_yscale('log')
    ax_histy.set_xscale('log')
    ax_histy.set_ylim(charge_lo, charge_hi)
    ax_histy.set_xlim(0.8, d['histy_max'])
    ax_histy.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_histy.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_histy.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax_histy.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_histy.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax_histy.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_histy.tick_params(axis='y', which='both', labelleft=False, labelright=False)
    style_full_parity_axis(ax_histy)
    ax_histy.xaxis.set_zorder(40)
    ax_histy.yaxis.set_zorder(40)
    for spine in ax_histy.spines.values():
        spine.set_zorder(50)
    ax_histy.set_xlabel('')
    ax_histy.set_ylabel('')

    charge_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=CHARGE_TRAIN_COLOR, markeredgecolor='k', markeredgewidth=0.4, markersize=4, label=r'$\mathrm{TRN}$'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor=CHARGE_VAL_COLOR, markeredgecolor='k', markeredgewidth=0.4, markersize=4, label=r'$\mathrm{VAL}$'),
    ]

    ax_charge.legend(handles=charge_handles, loc='lower right', frameon=False, fontsize=LEGENDSZ, handlelength=0.8, handletextpad=0.35, borderaxespad=0.20)
    ax_charge.text(-0.30, 1.15, panel_label, transform=ax_charge.transAxes, fontsize=PANEL_FTSZ, ha='left', va='top', clip_on=False)

def create_charge_figure(mp_charge_df, qm9_charge_df):
    fig = plt.figure(figsize=CHARGE_FIG_SIZE)
    panel_w = 0.5
    plot_charge_parity_panel(fig, left=0.0, panel_w=panel_w, charge_df=mp_charge_df, panel_label='a)')
    plot_charge_parity_panel(fig, left=0.5, panel_w=panel_w, charge_df=qm9_charge_df, panel_label='b)')
    return fig

def create_figure_2(pbe_loss_data, qm_loss_data, mp_distribution_data, qm9_distribution_data):
    fig = plt.figure(figsize=FIG2_SIZE)
    ax_pbe, ax_qm, ax_mp_violin, ax_qm9_violin = create_original_fig2_row_axes(fig, 4)

    plot_lmax_loss(ax_pbe, pbe_loss_data, MP_RUN_NAMES, show_xlabel=True, show_legend=False, show_ylabel=True)
    plot_lmax_loss(ax_qm, qm_loss_data, QM9_RUN_NAMES, show_xlabel=True, show_legend=False, show_ylabel=True)
    plot_violin_panel(ax_mp_violin, mp_distribution_data, ylim=(0, 5))
    plot_violin_panel(ax_qm9_violin, qm9_distribution_data, ylim=(0, 2))

    model_handles = [Line2D([0], [0], color=color, lw=AXIS_LW, linestyle='-', label=label) for color, label in zip(L_COLORS, LABELS)]
    legend_order = [0, 3, 1, 4, 2]
    ax_pbe.legend(handles=[model_handles[i] for i in legend_order], loc='upper right', ncol=3, frameon=False, fontsize=LEGENDSZ, handlelength=0.70, handletextpad=0.15, columnspacing=0.60, labelspacing=0.10, borderaxespad=0.25)

    add_inside_panel_label(ax_pbe, 'a)')
    add_inside_panel_label(ax_qm, 'b)')
    add_inside_panel_label(ax_mp_violin, 'c)')
    add_inside_panel_label(ax_qm9_violin, 'd)')

    return fig

def save_figure(fig, stem, tight=False):
    save_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.05} if tight else {}
    output_dirs = (BASE_DIR, PAPER_FIG_DIR)

    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f'{stem}.pdf')
        jpg_path = os.path.join(output_dir, f'{stem}.jpg')
        fig.savefig(pdf_path, **save_kwargs)
        fig.savefig(jpg_path, dpi=1200, **save_kwargs)
        print(f'Saved: {pdf_path}')
        print(f'Saved: {jpg_path}')

def main():
    plt.close('all')

    if PLOT_SELECTION not in {'fig1', 'fig2', 'both'}:
        raise ValueError("PLOT_SELECTION must be 'fig1', 'fig2', or 'both'.")

    if PLOT_SELECTION in {'fig1', 'both'}:
        hse_data = load_hse_loss_data()
        inference_stats = load_inference_time_stats(INFERENCE_TIMES_CSV)
        parameter_df = load_parameter_comparison(PARAM_EFFICIENCY_CSV)

        fig1 = create_figure_1(inference_stats, parameter_df, hse_data)
        save_figure(fig1, 'mp-qm-distribution-hse')

    if PLOT_SELECTION in {'fig2', 'both'}:
        pbe_loss_data = load_lmax_loss_data(BASE_DIR, MP_RUN_NAMES)
        qm_loss_data = load_lmax_loss_data(BASE_DIR, QM9_RUN_NAMES)
        mp_distribution_data = load_distribution_data(MP_NMAE_CSV)
        qm9_distribution_data = load_distribution_data(QM9_NMAE_CSV)

        fig2 = create_figure_2(pbe_loss_data, qm_loss_data, mp_distribution_data, qm9_distribution_data)
        save_figure(fig2, 'pbe-qm-loss', tight=True)

    plt.show()

if __name__ == '__main__':
    main()
