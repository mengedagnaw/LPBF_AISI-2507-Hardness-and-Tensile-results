#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# -------------------------------------------------
# 0. Read data from Excel (one sheet per condition)
# -------------------------------------------------
hra_file = "HRA_SDSS_3% Ni .xlsx"     # adjust path/name if needed
hv_file  = "HV0.1_SDSS_3% Ni.xlsx"

# Read all sheets into dicts: {sheet_name: DataFrame}
hra_dict = pd.read_excel(hra_file, sheet_name=None)
hv_dict  = pd.read_excel(hv_file, sheet_name=None)

def build_summary_df(hra_sheets, hv_sheets):
    """
    For each sheet (AS, SR400_1h, ...), compute:
      - mean/std Surface hardness  → HRA_Surface, HRA_Surface_Err
      - mean/std Core hardness     → HRA_Cross,   HRA_Cross_Err
      - mean/std HV0.1             → HV,          HV_Err
    And map sheet names like 'SR400_1h' → 'SR400' for plotting.
    """
    rows = []
    for sheet_name, hra_df in hra_sheets.items():
        hv_df = hv_sheets[sheet_name]

        cond_label = sheet_name.split("_")[0]  # "SR400_1h" → "SR400"

        # --- HRA data ---
        surface = hra_df["Surface hardness"].dropna()
        core    = hra_df["Core hardness"].dropna()

        # --- HV data ---
        hv_col_candidates = [c for c in hv_df.columns if str(c).startswith("HV0.1")]
        if not hv_col_candidates:
            raise ValueError(f"No HV0.1 column found in sheet '{sheet_name}' – check Excel.")
        hv_col = hv_col_candidates[0]
        hv_vals = hv_df[hv_col].dropna()

        rows.append({
            "Condition_raw": sheet_name,
            "Condition": cond_label,
            "HRA_Surface": surface.mean(),
            "HRA_Surface_Err": surface.std(ddof=1),
            "HRA_Cross": core.mean(),
            "HRA_Cross_Err": core.std(ddof=1),
            "HV": hv_vals.mean(),
            "HV_Err": hv_vals.std(ddof=1),
        })

    df = pd.DataFrame(rows)
    return df

df = build_summary_df(hra_dict, hv_dict)

# -------------------------------------------------
# 1. Global style
# -------------------------------------------------
mpl.rcParams['text.usetex'] = False
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 20,
    'font.weight': 'bold',
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 20,
    'ytick.labelsize': 22,
    'axes.linewidth': 2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'legend.fontsize': 20,
    'figure.dpi': 600,
    'grid.linestyle': '-',
    'grid.alpha': 0.3,
})

# -------------------------------------------------
# New publication-friendly, distinct palette (Tol "muted"-style)
# -------------------------------------------------
# Surface (HRA) → cyan
START_COLOR_SURFACE = '#88CCEE'   # light cyan
END_COLOR_SURFACE   = '#44AADD'   # deeper cyan

# Core (HRA, cross-section) → rose
START_COLOR_CROSS   = '#CC6677'   # muted rose
END_COLOR_CROSS     = '#AA4455'   # darker rose

# HV0.1 → forest green
START_COLOR_HV      = '#3cb371'   # Medium sea green
END_COLOR_HV        = '#0086A2'   # deep marine 

cmap_surface = LinearSegmentedColormap.from_list(
    "surface_grad", [START_COLOR_SURFACE, END_COLOR_SURFACE]
)
cmap_cross   = LinearSegmentedColormap.from_list(
    "cross_grad",   [START_COLOR_CROSS,   END_COLOR_CROSS]
)
cmap_hv      = LinearSegmentedColormap.from_list(
    "hv_grad",      [START_COLOR_HV,      END_COLOR_HV]
)

# -------------------------------------------------
# 2. Figure and axes
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
ax1, ax2 = axes
fig.patch.set_facecolor('white')

fig.suptitle(
    'Material Hardness Measurements by Sample Condition',
    fontsize=30, fontweight='bold', y=1.03
)

for ax in axes:
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)

# -------------------------------------------------
# 3. HRA plot (left)
# -------------------------------------------------
x = np.arange(len(df["Condition"]))
bar_width = 0.35

for i, row in df.iterrows():
    frac = i / max(len(df) - 1, 1)  # 0 → 1 across bars
    color_s = cmap_surface(0.2 + 0.3 * frac)
    color_c = cmap_cross(0.2 + 0.3 * frac)

    ax1.bar(
        x[i] - bar_width/2, row["HRA_Surface"], bar_width,
        yerr=row["HRA_Surface_Err"], capsize=8, ecolor='black',
        color=color_s, edgecolor='black', linewidth=1.5, zorder=3
    )
    ax1.bar(
        x[i] + bar_width/2, row["HRA_Cross"], bar_width,
        yerr=row["HRA_Cross_Err"], capsize=8, ecolor='black',
        color=color_c, edgecolor='black', linewidth=1.5, zorder=3
    )

ax1.set_ylabel('HRA', rotation=90, labelpad=18)
ax1.set_xticks(x)
labels1 = ax1.set_xticklabels(df["Condition"], rotation=15, ha='right')
for lbl in labels1:
    lbl.set_fontweight('bold')

ax1.set_ylim(50, 80)
ax1.set_yticks(np.arange(50, 81, 5))
ax1.grid(axis='y', linewidth=1.0, alpha=0.4, zorder=0)
ax1.set_axisbelow(True)

ax1.legend(
    handles=[
        Patch(facecolor=cmap_surface(0.5), edgecolor='black', label='Surface'),
        Patch(facecolor=cmap_cross(0.5),   edgecolor='black', label='Core (Cross-section)')
    ],
    loc='upper left',
    ncol=1,
    frameon=True,
    edgecolor='black',
    prop={'weight': 'bold', 'size': 18},
    bbox_to_anchor=(0.32, 0.98)
)

# -------------------------------------------------
# 4. HV plot (right)
# -------------------------------------------------
for i, row in df.iterrows():
    frac = i / max(len(df) - 1, 1)
    color_h = cmap_hv(0.2 + 0.3 * frac)
    ax2.bar(
        x[i], row["HV"], bar_width * 1.2,
        yerr=row["HV_Err"], capsize=8, ecolor='black',
        color=color_h, edgecolor='black', linewidth=1.5, zorder=3
    )

ax2.set_ylabel(r'HV$_{0.1}$', rotation=90, labelpad=18)
ax2.set_xticks(x)
labels2 = ax2.set_xticklabels(df["Condition"], rotation=15, ha='right')
for lbl in labels2:
    lbl.set_fontweight('bold')

ax2.set_ylim(250, 550)  # or (0, 600) if you prefer full scale
ax2.set_yticks(np.arange(250, 551, 50))
ax2.grid(axis='y', linewidth=1.0, alpha=0.4, zorder=0)
ax2.set_axisbelow(True)

# -------------------------------------------------
# 5. Layout & save
# -------------------------------------------------
plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14, wspace=0.25)
plt.savefig("hardness_gradient_tolmuted.png", dpi=600, bbox_inches="tight")
plt.show()


# In[ ]:




