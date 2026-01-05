#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from google.colab import files

# === Config: axis thickness + figure label ===
AXIS_LINE        = 5           # ↑ increase for thicker axes/ticks
FIG_LABEL_TEXT   = "(b)"         # figure label
FIG_LABEL_COLOR  = "#000080"     # vivid, distinct
FIG_LABEL_SIZE   = 60            # font size

# === 1) Upload & resolve actual saved filename (Colab may rename like "AS (4).xlsx") ===
print("📁 UPLOAD AS.xlsx NOW:")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No file uploaded.")
xlsx_candidates = sorted(Path('.').glob('*.xlsx'),
                         key=lambda p: p.stat().st_mtime, reverse=True)
filename = str(xlsx_candidates[0])
print(f"📦 Using file on disk: {filename}")

# === 2) Sheets & specimen areas (mm^2) ===
sheets = ['Test 1', 'Test 2', 'Test 3', 'Test 4']
areas_mm2 = {
    'Test 1': 32.7756,
    'Test 2': 33.1740,
    'Test 3': 32.5458,
    'Test 4': 32.8329
}

# === 3) Helper: find STRAIN (%) and LOAD (N) columns (accepts mislabel 'Stress (N)') ===
def find_strain_and_load_cols(df):
    cols = [c for c in df.columns if isinstance(c, str)]
    low  = [c.lower() for c in cols]

    # Strain
    cand_idx = [i for i, s in enumerate(low) if 'strain' in s]
    if not cand_idx:
        raise ValueError(f"Could not find a strain column. Columns: {cols}")
    strain_idx = next((i for i in cand_idx if ('%' in low[i]) or ('percent' in low[i])), cand_idx[0])
    strain_col = cols[strain_idx]

    # Load in Newtons (Load/Force) OR mislabel 'Stress (N)' (but not MPa/Pa/GPa)
    def is_newton_header(s):
        s = s.strip()
        if any(u in s for u in ['mpa', 'gpa', ' pa']):  # exclude pressure units
            return False
        return bool(re.search(r'(\(|\[)\s*n\s*(\)|\])', s) or s.endswith(' n') or s.endswith('/n'))

    load_idx = None
    for i, s in enumerate(low):
        if re.search(r'\b(load|force)\b', s) and is_newton_header(s):
            load_idx = i
            break
    if load_idx is None:
        for i, s in enumerate(low):
            if 'stress' in s and is_newton_header(s):  # e.g., "Stress (N)" mislabel
                load_idx = i
                break
    if load_idx is None:
        raise ValueError(f"Could not find a Load/Force (N) column. Columns: {cols}")

    return strain_col, cols[load_idx]

# === 4) Read workbook & build long dataframe (Load N / Area mm² → MPa) ===
xls = pd.ExcelFile(filename)
df_list = []
for sheet in sheets:
    df_raw = pd.read_excel(xls, sheet_name=sheet)
    print(f"\n🔍 COLUMNS IN {sheet}: {list(df_raw.columns)}")
    strain_col, load_col = find_strain_and_load_cols(df_raw)
    print(f"✅ Using Strain: '{strain_col}' | Load (N): '{load_col}' for {sheet}")

    area = areas_mm2[sheet]
    stress_mpa = pd.to_numeric(df_raw[load_col], errors='coerce') / area
    s = pd.to_numeric(df_raw[strain_col], errors='coerce')
    s_percent = 100.0 * s if s.max(skipna=True) is not None and s.max(skipna=True) <= 1.1 else s

    df_list.append(pd.DataFrame({
        'Strain(%)': s_percent,
        'Stress (MPa)': stress_mpa,
        'Sheet': sheet
    }))

df = pd.concat(df_list, ignore_index=True)
print(f"\n✅ LOADED: {filename} ({len(sheets)} sheets)")
print(f"📊 TOTAL ROWS: {len(df)}")
print(f"🔍 FINAL COLUMNS: {list(df.columns)}")

# === 5) Styling ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': AXIS_LINE,
    'mathtext.default': 'regular',
    'lines.linewidth': 7,
    'lines.solid_capstyle': 'round'
})
plt.rcParams['font.serif'] = ["Palatino Linotype", "Palatino", "URW Palladio L"] + plt.rcParams.get('font.serif', [])

# === 6) Plot ===
plt.figure(figsize=(10.5, 7.8), dpi=1200)
colors = ['#D55E00', '#E69F00', '#56B4E9', '#009E73']

for i, sheet in enumerate(sheets):
    d = df[df['Sheet'] == sheet].dropna(subset=['Strain(%)', 'Stress (MPa)']).sort_values('Strain(%)')
    plt.plot(d['Strain(%)'], d['Stress (MPa)'], label=f'AS_Test {i+1}', color=colors[i])

# Axis labels (bold)
plt.xlabel(r'$\epsilon$ (%)', fontsize=40, fontweight='bold')
plt.ylabel(r'$\sigma$ (MPa)', fontsize=40, fontweight='bold')

ax = plt.gca()
# Ensure all spines honor AXIS_LINE
for side in ('left', 'bottom', 'right', 'top'):
    ax.spines[side].set_linewidth(AXIS_LINE)
# Minimal look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# === Fixed ticks & limits ===
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([0, 400, 800, 1200, 1600])
ax.set_xlim(0, 8)
ax.set_ylim(0, 1600)
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Tick params + bold tick values
ax.tick_params(axis='both', which='major',
               labelsize=38, width=AXIS_LINE, length=10, direction='out', pad=8)
for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
    ticklabel.set_fontweight('bold')

# === Figure label centered at the top edge ===
ax.text(0.5, 1.02, FIG_LABEL_TEXT,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=FIG_LABEL_SIZE,
        color=FIG_LABEL_COLOR,
        fontweight='bold')

# === Legend with BOLD text like "AS_Test 1", "AS_Test 2", ... ===
leg = plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.02),
                 fontsize=30, framealpha=0.85, borderpad=0.6,
                 labelspacing=0.6, fancybox=True)
for txt in leg.get_texts():
    txt.set_fontweight('bold')   # <- make legend entries bold
if leg.get_title():
    leg.get_title().set_fontweight('bold')

plt.tight_layout()
plt.subplots_adjust(left=0.14, bottom=0.12)

# === 7) Save THEN show ===
outname = 'LPBF_SDSS2507_AS_Stress_Strain.png'
plt.savefig(outname, format='png', dpi=1200, bbox_inches='tight')
plt.show()

print("🎉 ULTRA-HIGH-DPI PNG SAVED!")
print(f"📄 Filename: {outname}")
print(f"🔧 Axis & tick thickness set via AXIS_LINE = {AXIS_LINE}")
print(f"🏷️ Figure label: {FIG_LABEL_TEXT} (size {FIG_LABEL_SIZE}, color {FIG_LABEL_COLOR})")




# In[ ]:


# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from google.colab import files

# ==== CONFIG ====
SERIES = 'SR400_1h'                               # legend prefix
AXIS_LINE = 5                                    # axis & tick thickness
OUTPUT_NAME = f'{SERIES}_StressStrain_NtoMPa.png'  # output filename

# Figure label config
FIG_LABEL_TEXT  = "(c)"       # figure label
FIG_LABEL_COLOR = "#000080"   # vivid, distinct
FIG_LABEL_SIZE  = 60          # font size

print(f"📁 UPLOAD your {SERIES}.xlsx NOW:")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No file uploaded.")

# Use the most recently saved .xlsx on disk (handles "X (n).xlsx" renames)
xlsx_candidates = sorted(Path('.').glob('*.xlsx'),
                         key=lambda p: p.stat().st_mtime, reverse=True)
filename = str(xlsx_candidates[0])
print(f"📦 Using file on disk: {filename}")

# ==== SHEETS & SPECIMEN AREAS (mm²) ====
sheets = ['Test 1', 'Test 2', 'Test 3', 'Test 4']
areas_mm2 = {
    'Test 1': 32.3102,
    'Test 2': 32.8208,
    'Test 3': 33.0616,
    'Test 4': 32.8320,
}

# ==== COLUMN FINDERS ====
def find_strain_and_load_cols(df):
    cols = [c for c in df.columns if isinstance(c, str)]
    low  = [c.lower() for c in cols]

    # Strain: prefer % headers, else any containing 'strain'
    cand_idx = [i for i, s in enumerate(low) if 'strain' in s]
    if not cand_idx:
        raise ValueError(f"Could not find a strain column. Columns: {cols}")
    strain_idx = next((i for i in cand_idx if ('%' in low[i]) or ('percent' in low[i])), cand_idx[0])
    strain_col = cols[strain_idx]

    # Load in N (or mislabel 'Stress (N)'), exclude pressure units
    def is_newton_header(s):
        if any(u in s for u in ['mpa', 'gpa', ' pa']):  # exclude pressure units
            return False
        return bool(re.search(r'(\(|\[)\s*n\s*(\)|\])', s) or s.endswith(' n') or s.endswith('/n'))

    load_idx = None
    for i, s in enumerate(low):
        if re.search(r'\b(load|force)\b', s) and is_newton_header(s):
            load_idx = i
            break
    if load_idx is None:
        for i, s in enumerate(low):
            if 'stress' in s and is_newton_header(s):  # e.g., "Stress (N)" mislabel
                load_idx = i
                break
    if load_idx is None:
        raise ValueError(f"Could not find a Load/Force (N) column. Columns: {cols}")

    return strain_col, cols[load_idx]

# ==== READ & CONVERT (Load N / Area mm² → Stress MPa) ====
xls = pd.ExcelFile(filename)
df_list = []
for sheet in sheets:
    df_raw = pd.read_excel(xls, sheet_name=sheet)
    print(f"\n🔍 COLUMNS IN {sheet}: {list(df_raw.columns)}")
    strain_col, load_col = find_strain_and_load_cols(df_raw)
    print(f"✅ Using Strain: '{strain_col}' | Load (N): '{load_col}' | Area: {areas_mm2[sheet]:.4f} mm²")

    area = areas_mm2[sheet]
    stress_mpa = pd.to_numeric(df_raw[load_col], errors='coerce') / area  # 1 N/mm² = 1 MPa
    s = pd.to_numeric(df_raw[strain_col], errors='coerce')
    s_percent = 100.0 * s if s.max(skipna=True) is not None and s.max(skipna=True) <= 1.1 else s

    df_list.append(pd.DataFrame({
        'Strain(%)': s_percent,
        'Stress (MPa)': stress_mpa,
        'Sheet': sheet
    }))

df = pd.concat(df_list, ignore_index=True)
print(f"\n✅ LOADED: {filename} ({len(sheets)} sheets)")
print(f"📊 TOTAL ROWS: {len(df)}")
print(f"🔍 FINAL COLUMNS: {list(df.columns)}")

# ==== STYLING ====
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': AXIS_LINE,     # thick axes
    'mathtext.default': 'regular',
    'lines.linewidth': 7,            # bold curves
    'lines.solid_capstyle': 'round'
})
plt.rcParams['font.serif'] = ["Palatino Linotype", "Palatino", "URW Palladio L"] + plt.rcParams.get('font.serif', [])

# ==== PLOT ====
plt.figure(figsize=(10.5, 7.8), dpi=1200)

# Distinct, high-contrast palette (different from earlier AS colors)
colors = ['#3B528B', '#21918C', '#5EC962', '#440154']

for i, sheet in enumerate(sheets):
    d = df[df['Sheet'] == sheet].dropna(subset=['Strain(%)', 'Stress (MPa)']).sort_values('Strain(%)')
    if d.empty:
        continue
    plt.plot(d['Strain(%)'], d['Stress (MPa)'],
             label=f'{SERIES}_Test {i+1}', color=colors[i])

# Axis labels (bold)
plt.xlabel(r'$\epsilon$ (%)', fontsize=40, fontweight='bold')
plt.ylabel(r'$\sigma$ (MPa)', fontsize=40, fontweight='bold')

ax = plt.gca()
# Spines
for side in ('left', 'bottom', 'right', 'top'):
    ax.spines[side].set_linewidth(AXIS_LINE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Fixed ticks & limits
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([0, 400, 800, 1200, 1600])
ax.set_xlim(0, 8)
ax.set_ylim(0, 1600)
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Thick tick marks + bold tick values
ax.tick_params(axis='both', which='major',
               labelsize=38, width=AXIS_LINE, length=10, direction='out', pad=8)
for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
    ticklabel.set_fontweight('bold')

# ---- Figure label centered at the very top (fix for SR400_1h) ----
ax.text(0.5, 1.02, FIG_LABEL_TEXT,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=FIG_LABEL_SIZE,
        color=FIG_LABEL_COLOR,
        fontweight='bold')

# Legend (bold labels like SR400_1h_Test 1, etc.)
leg = plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.02),
                 fontsize=30, framealpha=0.85, borderpad=0.6, labelspacing=0.6, fancybox=True)
for txt in leg.get_texts():
    txt.set_fontweight('bold')

plt.tight_layout()
plt.subplots_adjust(left=0.14, bottom=0.12)

# ==== SAVE THEN SHOW ====
plt.savefig(OUTPUT_NAME, format='png', dpi=1200, bbox_inches='tight')
plt.show()

print("🎉 ULTRA-HIGH-DPI PNG SAVED!")
print(f"📄 Filename: {OUTPUT_NAME}")
print("✅ Stress (MPa) = Load (N) / Area (mm²) using your SR400_1h areas.")
print(f"🏷️ Figure label placed: {FIG_LABEL_TEXT} (size {FIG_LABEL_SIZE}, color {FIG_LABEL_COLOR})")


# In[ ]:


# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from google.colab import files

# ==== CONFIG ====
SERIES = 'SR450_1h'                               # legend prefix
AXIS_LINE = 5                                    # axis & tick thickness
OUTPUT_NAME = f'{SERIES}_StressStrain_NtoMPa.png'  # output filename

# Figure label config (top-center)
FIG_LABEL_TEXT  = "(d)"       # figure label
FIG_LABEL_COLOR = "#000080"   # vivid, distinct
FIG_LABEL_SIZE  = 60          # font size

print(f"📁 UPLOAD your {SERIES}.xlsx NOW:")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No file uploaded.")

# Use the most recently saved .xlsx on disk (handles Colab renames like 'X (n).xlsx')
xlsx_candidates = sorted(Path('.').glob('*.xlsx'),
                         key=lambda p: p.stat().st_mtime, reverse=True)
filename = str(xlsx_candidates[0])
print(f"📦 Using file on disk: {filename}")

# ==== SHEETS & SPECIMEN AREAS (mm²) — YOUR VALUES ====
sheets = ['Test 1', 'Test 2', 'Test 3', 'Test 4']
areas_mm2 = {
    'Test 1': 34.8096,
    'Test 2': 34.6320,
    'Test 3': 34.5719,
    'Test 4': 34.8634,
}

# ==== COLUMN FINDERS ====
# Find STRAIN (%) and LOAD (N). Accepts 'Load (N)', 'Force (N)', and mislabels like 'Stress (N)'.
def find_strain_and_load_cols(df):
    cols = [c for c in df.columns if isinstance(c, str)]
    low  = [c.lower() for c in cols]

    # Strain: prefer headers with %; else any containing 'strain'
    cand_idx = [i for i, s in enumerate(low) if 'strain' in s]
    if not cand_idx:
        raise ValueError(f"Could not find a strain column. Columns: {cols}")
    strain_idx = next((i for i in cand_idx if ('%' in low[i]) or ('percent' in low[i])), cand_idx[0])
    strain_col = cols[strain_idx]

    # Load in Newtons (or mislabel 'Stress (N)'), but exclude pressure units (MPa/Pa/GPa)
    def looks_like_newtons(s):
        if any(u in s for u in ['mpa', 'gpa', ' pa']):  # exclude pressure units
            return False
        return bool(re.search(r'(\(|\[)\s*n\s*(\)|\])', s) or s.endswith(' n') or s.endswith('/n'))

    load_idx = None
    for i, s in enumerate(low):
        if re.search(r'\b(load|force)\b', s) and looks_like_newtons(s):
            load_idx = i
            break
    if load_idx is None:
        for i, s in enumerate(low):
            if 'stress' in s and looks_like_newtons(s):  # e.g., "Stress (N)" mislabel
                load_idx = i
                break
    if load_idx is None:
        raise ValueError(f"Could not find a Load/Force (N) column. Columns: {cols}")

    return strain_col, cols[load_idx]

# ==== READ & CONVERT (Load N / Area mm² → Stress MPa) ====
xls = pd.ExcelFile(filename)
df_list = []
for sheet in sheets:
    df_raw = pd.read_excel(xls, sheet_name=sheet)
    print(f"\n🔍 COLUMNS IN {sheet}: {list(df_raw.columns)}")
    strain_col, load_col = find_strain_and_load_cols(df_raw)
    print(f"✅ Using Strain: '{strain_col}' | Load (N): '{load_col}' | Area: {areas_mm2[sheet]:.4f} mm²")

    area = areas_mm2[sheet]
    stress_mpa = pd.to_numeric(df_raw[load_col], errors='coerce') / area  # 1 N/mm² = 1 MPa
    s = pd.to_numeric(df_raw[strain_col], errors='coerce')
    s_percent = 100.0 * s if s.max(skipna=True) is not None and s.max(skipna=True) <= 1.1 else s

    df_list.append(pd.DataFrame({
        'Strain(%)': s_percent,
        'Stress (MPa)': stress_mpa,
        'Sheet': sheet
    }))

df = pd.concat(df_list, ignore_index=True)
print(f"\n✅ LOADED: {filename} ({len(sheets)} sheets)")
print(f"📊 TOTAL ROWS: {len(df)}")
print(f"🔍 FINAL COLUMNS: {list(df.columns)}")

# ==== STYLING ====
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': AXIS_LINE,     # thick axes
    'mathtext.default': 'regular',
    'lines.linewidth': 7,            # bold curves
    'lines.solid_capstyle': 'round'
})
plt.rcParams['font.serif'] = ["Palatino Linotype", "Palatino", "URW Palladio L"] + plt.rcParams.get('font.serif', [])

# ==== PLOT ====
plt.figure(figsize=(10.5, 7.8), dpi=1200)

# Distinct, high-contrast palette (different from your other series)
colors = ['#8E0152', '#5C53A5', '#1A9641', '#F46D43']

for i, sheet in enumerate(sheets):
    d = df[df['Sheet'] == sheet].dropna(subset=['Strain(%)', 'Stress (MPa)']).sort_values('Strain(%)')
    if d.empty:
        continue
    plt.plot(d['Strain(%)'], d['Stress (MPa)'],
             label=f'{SERIES}_Test {i+1}', color=colors[i])

# Axis labels (bold)
plt.xlabel(r'$\epsilon$ (%)', fontsize=40, fontweight='bold')
plt.ylabel(r'$\sigma$ (MPa)', fontsize=40, fontweight='bold')

ax = plt.gca()
# Spines
for side in ('left', 'bottom', 'right', 'top'):
    ax.spines[side].set_linewidth(AXIS_LINE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Fixed ticks & limits
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([0, 400, 800, 1200, 1600])
ax.set_xlim(0, 8)
ax.set_ylim(0, 1600)
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Thick tick marks + bold tick values
ax.tick_params(axis='both', which='major',
               labelsize=38, width=AXIS_LINE, length=10, direction='out', pad=8)
for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
    ticklabel.set_fontweight('bold')

# ---- Figure label centered at the very top ----
ax.text(0.5, 1.02, FIG_LABEL_TEXT,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=FIG_LABEL_SIZE,
        color=FIG_LABEL_COLOR,
        fontweight='bold')

# Legend (bold labels like SR450_1h_Test 1, etc.)
leg = plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.02),
                 fontsize=30, framealpha=0.85, borderpad=0.6, labelspacing=0.6, fancybox=True)
for txt in leg.get_texts():
    txt.set_fontweight('bold')

plt.tight_layout()
plt.subplots_adjust(left=0.14, bottom=0.12)

# ==== SAVE THEN SHOW ====
plt.savefig(OUTPUT_NAME, format='png', dpi=1200, bbox_inches='tight')
plt.show()

print("🎉 ULTRA-HIGH-DPI PNG SAVED!")
print(f"📄 Filename: {OUTPUT_NAME}")
print("✅ Stress (MPa) = Load (N) / Area (mm²) using your SR450_1h areas.")
print(f"🏷️ Figure label placed: {FIG_LABEL_TEXT} (size {FIG_LABEL_SIZE}, color {FIG_LABEL_COLOR})")



# In[ ]:


# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from google.colab import files

# ==== CONFIG ====
SERIES = 'SR500_1h'                               # legend prefix
AXIS_LINE = 5                                    # axis & tick thickness
OUTPUT_NAME = f'{SERIES}_StressStrain_NtoMPa.png' # output filename

# Figure label config (top-center)
FIG_LABEL_TEXT  = "(e)"       # figure label
FIG_LABEL_COLOR = "#000080"   # vivid, distinct
FIG_LABEL_SIZE  = 60          # font size

print(f"📁 UPLOAD your {SERIES}.xlsx NOW:")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No file uploaded.")

# Use the most recently saved .xlsx on disk (handles Colab renames like 'X (n).xlsx')
xlsx_candidates = sorted(Path('.').glob('*.xlsx'),
                         key=lambda p: p.stat().st_mtime, reverse=True)
filename = str(xlsx_candidates[0])
print(f"📦 Using file on disk: {filename}")

# ==== SHEETS & SPECIMEN AREAS (mm²) — YOUR VALUES ====
sheets = ['Test 1', 'Test 2', 'Test 3', 'Test 4']
areas_mm2 = {
    'Test 1': 34.5644,
    'Test 2': 34.4533,
    'Test 3': 34.0992,
    'Test 4': 34.5136,
}

# ==== COLUMN FINDERS ====
def find_strain_and_load_cols(df):
    cols = [c for c in df.columns if isinstance(c, str)]
    low  = [c.lower() for c in cols]

    # Strain: prefer headers with %; else any containing 'strain'
    cand_idx = [i for i, s in enumerate(low) if 'strain' in s]
    if not cand_idx:
        raise ValueError(f"Could not find a strain column. Columns: {cols}")
    strain_idx = next((i for i in cand_idx if ('%' in low[i]) or ('percent' in low[i])), cand_idx[0])
    strain_col = cols[strain_idx]

    # Load in Newtons (or mislabel 'Stress (N)'), but exclude pressure units
    def looks_like_newtons(s):
        if any(u in s for u in ['mpa', 'gpa', ' pa']):
            return False
        return bool(re.search(r'(\(|\[)\s*n\s*(\)|\])', s) or s.endswith(' n') or s.endswith('/n'))

    load_idx = None
    for i, s in enumerate(low):
        if re.search(r'\b(load|force)\b', s) and looks_like_newtons(s):
            load_idx = i
            break
    if load_idx is None:
        for i, s in enumerate(low):
            if 'stress' in s and looks_like_newtons(s):
                load_idx = i
                break
    if load_idx is None:
        raise ValueError(f"Could not find a Load/Force (N) column. Columns: {cols}")

    return strain_col, cols[load_idx]

# ==== READ & CONVERT (Load N / Area mm² → Stress MPa) ====
xls = pd.ExcelFile(filename)
df_list = []
for sheet in sheets:
    df_raw = pd.read_excel(xls, sheet_name=sheet)
    print(f"\n🔍 COLUMNS IN {sheet}: {list(df_raw.columns)}")
    strain_col, load_col = find_strain_and_load_cols(df_raw)
    print(f"✅ Using Strain: '{strain_col}' | Load (N): '{load_col}' | Area: {areas_mm2[sheet]:.4f} mm²")

    area = areas_mm2[sheet]
    stress_mpa = pd.to_numeric(df_raw[load_col], errors='coerce') / area  # 1 N/mm² = 1 MPa
    s = pd.to_numeric(df_raw[strain_col], errors='coerce')
    s_percent = 100.0 * s if s.max(skipna=True) is not None and s.max(skipna=True) <= 1.1 else s

    df_list.append(pd.DataFrame({
        'Strain(%)': s_percent,
        'Stress (MPa)': stress_mpa,
        'Sheet': sheet
    }))

df = pd.concat(df_list, ignore_index=True)
print(f"\n✅ LOADED: {filename} ({len(sheets)} sheets)")
print(f"📊 TOTAL ROWS: {len(df)}")
print(f"🔍 FINAL COLUMNS: {list(df.columns)}")

# ==== STYLING ====
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': AXIS_LINE,
    'mathtext.default': 'regular',
    'lines.linewidth': 7,
    'lines.solid_capstyle': 'round'
})
plt.rcParams['font.serif'] = ["Palatino Linotype", "Palatino", "URW Palladio L"] + plt.rcParams.get('font.serif', [])

# ==== PLOT ====
plt.figure(figsize=(10.5, 7.8), dpi=1200)

# Distinct palette
colors = ['#0D0887', '#6A00A8', '#B12A90', '#E16462']

for i, sheet in enumerate(sheets):
    d = df[df['Sheet'] == sheet].dropna(subset=['Strain(%)', 'Stress (MPa)']).sort_values('Strain(%)')
    plt.plot(d['Strain(%)'], d['Stress (MPa)'], label=f'{SERIES}_Test {i+1}', color=colors[i])

# Axis labels (bold)
plt.xlabel(r'$\epsilon$ (%)', fontsize=40, fontweight='bold')
plt.ylabel(r'$\sigma$ (MPa)', fontsize=40, fontweight='bold')

ax = plt.gca()
# Spines
for side in ('left', 'bottom', 'right', 'top'):
    ax.spines[side].set_linewidth(AXIS_LINE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Fixed ticks & limits
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([0, 400, 800, 1200, 1600])
ax.set_xlim(0, 8)
ax.set_ylim(0, 1600)
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Thick tick marks + **bold tick values (x & y)**
ax.tick_params(axis='both', which='major',
               labelsize=38, width=AXIS_LINE, length=10, direction='out', pad=8)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight('bold')

# ---- Figure label centered at the very top ----
ax.text(0.5, 1.02, FIG_LABEL_TEXT,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=FIG_LABEL_SIZE,
        color=FIG_LABEL_COLOR,
        fontweight='bold')

# Legend with **bold** entries like "SR500_1h_Test 1", etc.
leg = plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.02),
                 fontsize=30, framealpha=0.85, borderpad=0.6, labelspacing=0.6, fancybox=True)
for txt in leg.get_texts():
    txt.set_fontweight('bold')

plt.tight_layout()
plt.subplots_adjust(left=0.14, bottom=0.12)

# ==== SAVE THEN SHOW ====
plt.savefig(OUTPUT_NAME, format='png', dpi=1200, bbox_inches='tight')
plt.show()

print("🎉 ULTRA-HIGH-DPI PNG SAVED!")
print(f"📄 Filename: {OUTPUT_NAME}")
print("✅ Stress (MPa) = Load (N) / Area (mm²) using your SR500_1h areas.")
print(f"🏷️ Figure label placed: {FIG_LABEL_TEXT} (size {FIG_LABEL_SIZE}, color {FIG_LABEL_COLOR})")




# In[ ]:


# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from google.colab import files

# ==== CONFIG ====
SERIES = 'SR550_1h'                               # legend prefix
AXIS_LINE = 5                                    # axis & tick thickness
OUTPUT_NAME = f'{SERIES}_StressStrain_NtoMPa.png' # output filename

# Figure label config (top-center)
FIG_LABEL_TEXT  = "(f)"       # figure label
FIG_LABEL_COLOR = "#000080"   # vivid, distinct
FIG_LABEL_SIZE  = 60          # font size

print(f"📁 UPLOAD your {SERIES}.xlsx NOW:")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No file uploaded.")

# Use the most recently saved .xlsx on disk (handles Colab renames like 'X (n).xlsx')
xlsx_candidates = sorted(Path('.').glob('*.xlsx'),
                         key=lambda p: p.stat().st_mtime, reverse=True)
filename = str(xlsx_candidates[0])
print(f"📦 Using file on disk: {filename}")

# ==== SHEETS & SPECIMEN AREAS (mm²) ====
# Provide areas for the tests you want to plot; sheets without an area are skipped.
sheets = ['Test 1', 'Test 2', 'Test 3', 'Test 4']
areas_mm2 = {
    'Test 1': 34.0472,
    'Test 2': 33.9888,
    'Test 3': 33.0030,
    # 'Test 4': <add value if available>
}

# ==== COLUMN FINDERS ====
# Find STRAIN (%) and LOAD (N). Accepts 'Load (N)', 'Force (N)', and mislabels like 'Stress (N)'.
def find_strain_and_load_cols(df):
    cols = [c for c in df.columns if isinstance(c, str)]
    low  = [c.lower() for c in cols]

    # Strain: prefer headers with %; else any containing 'strain'
    cand_idx = [i for i, s in enumerate(low) if 'strain' in s]
    if not cand_idx:
        raise ValueError(f"Could not find a strain column. Columns: {cols}")
    strain_idx = next((i for i in cand_idx if ('%' in low[i]) or ('percent' in low[i])), cand_idx[0])
    strain_col = cols[strain_idx]

    # Load in Newtons (or mislabel 'Stress (N)'), but exclude pressure units (MPa/Pa/GPa)
    def looks_like_newtons(s):
        s = s.strip()
        if any(u in s for u in ['mpa', 'gpa', ' pa']):  # exclude pressure units
            return False
        return bool(re.search(r'(\(|\[)\s*n\s*(\)|\])', s) or s.endswith(' n') or s.endswith('/n'))

    load_idx = None
    for i, s in enumerate(low):
        if re.search(r'\b(load|force)\b', s) and looks_like_newtons(s):
            load_idx = i
            break
    if load_idx is None:
        for i, s in enumerate(low):
            if 'stress' in s and looks_like_newtons(s):  # e.g., "Stress (N)" mislabel
                load_idx = i
                break
    if load_idx is None:
        raise ValueError(f"Could not find a Load/Force (N) column. Columns: {cols}")

    return strain_col, cols[load_idx]

# ==== READ & CONVERT (Load N / Area mm² → Stress MPa) ====
xls = pd.ExcelFile(filename)
df_list = []
skipped = []

for sheet in sheets:
    if sheet not in xls.sheet_names:
        print(f"ℹ️ Sheet '{sheet}' not found in workbook; skipping.")
        continue

    if sheet not in areas_mm2:
        print(f"ℹ️ No area provided for '{sheet}'; skipping this sheet.")
        skipped.append(sheet)
        continue

    df_raw = pd.read_excel(xls, sheet_name=sheet)
    print(f"\n🔍 COLUMNS IN {sheet}: {list(df_raw.columns)}")
    strain_col, load_col = find_strain_and_load_cols(df_raw)
    area = areas_mm2[sheet]  # mm²
    print(f"✅ Using Strain: '{strain_col}' | Load (N): '{load_col}' | Area: {area:.4f} mm²")

    # Stress (MPa) = Load (N) / Area (mm²)
    stress_mpa = pd.to_numeric(df_raw[load_col], errors='coerce') / area
    s = pd.to_numeric(df_raw[strain_col], errors='coerce')
    s_percent = 100.0 * s if s.max(skipna=True) is not None and s.max(skipna=True) <= 1.1 else s

    df_list.append(pd.DataFrame({
        'Strain(%)': s_percent,
        'Stress (MPa)': stress_mpa,
        'Sheet': sheet
    }))

if not df_list:
    raise RuntimeError("No usable data found. Ensure sheet names and areas are correct.")

df = pd.concat(df_list, ignore_index=True)
print(f"\n✅ LOADED: {filename} | Sheets plotted: {sorted(df['Sheet'].unique())}")
if skipped:
    print(f"ℹ️ Skipped (no area defined): {skipped}")
print(f"📊 TOTAL ROWS: {len(df)}")
print(f"🔍 FINAL COLUMNS: {list(df.columns)}")

# ==== STYLING ====
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': AXIS_LINE,     # thick axes
    'mathtext.default': 'regular',
    'lines.linewidth': 7,            # bold curves
    'lines.solid_capstyle': 'round'
})
plt.rcParams['font.serif'] = ["Palatino Linotype", "Palatino", "URW Palladio L"] + plt.rcParams.get('font.serif', [])

# ==== PLOT ====
plt.figure(figsize=(10.5, 7.8), dpi=1200)

# Distinct, colorblind-friendly palette
palette = {
    'Test 1': '#CC79A7',  # reddish purple
    'Test 2': '#0072B2',  # blue
    'Test 3': '#009E73',  # bluish green
    'Test 4': '#E69F00',  # orange (only if included)
}

for i, sheet in enumerate(sheets):
    d = df[df['Sheet'] == sheet].dropna(subset=['Strain(%)', 'Stress (MPa)']).sort_values('Strain(%)')
    if d.empty:
        continue
    plt.plot(d['Strain(%)'], d['Stress (MPa)'],
             label=f'{SERIES}_Test {i+1}',
             color=palette.get(sheet, None))

# Axis labels (bold)
plt.xlabel(r'$\epsilon$ (%)', fontsize=40, fontweight='bold')
plt.ylabel(r'$\sigma$ (MPa)', fontsize=40, fontweight='bold')

ax = plt.gca()
# Spines
for side in ('left', 'bottom', 'right', 'top'):
    ax.spines[side].set_linewidth(AXIS_LINE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Fixed ticks & limits
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([0, 400, 800, 1200, 1600])
ax.set_xlim(0, 8)
ax.set_ylim(0, 1600)
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Thick tick marks + bold tick values
ax.tick_params(axis='both', which='major',
               labelsize=38, width=AXIS_LINE, length=10, direction='out', pad=8)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight('bold')

# ---- Figure label centered at the very top ----
ax.text(0.5, 1.02, FIG_LABEL_TEXT,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=FIG_LABEL_SIZE,
        color=FIG_LABEL_COLOR,
        fontweight='bold')

# Legend with bold entries like "SR550_1h_Test 1", etc.
leg = plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.02),
                 fontsize=30, framealpha=0.85, borderpad=0.6, labelspacing=0.6, fancybox=True)
for txt in leg.get_texts():
    txt.set_fontweight('bold')

plt.tight_layout()
plt.subplots_adjust(left=0.14, bottom=0.12)

# ==== SAVE THEN SHOW ====
plt.savefig(OUTPUT_NAME, format='png', dpi=1200, bbox_inches='tight')
plt.show()

print("🎉 ULTRA-HIGH-DPI PNG SAVED!")
print(f"📄 Filename: {OUTPUT_NAME}")
print("✅ Stress (MPa) = Load (N) / Area (mm²) using your SR550_1h areas.")
print(f"🏷️ Figure label placed: {FIG_LABEL_TEXT} (size {FIG_LABEL_SIZE}, color {FIG_LABEL_COLOR})")


# In[ ]:


# -*- coding: utf-8 -*-
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from google.colab import files

# =========================
# CONFIG
# =========================
SERIES = 'SA1100_15min'                             # legend prefix
AXIS_LINE = 5                                     # axis & tick thickness
OUTPUT_NAME = f'{SERIES}_StressStrain_NtoMPa.png'   # output filename

# Figure label (top-center)
FIG_LABEL_TEXT  = "(g)"
FIG_LABEL_COLOR = "#000080"
FIG_LABEL_SIZE  = 60

# Ticks & limits
X_TICKS  = [0, 2, 4, 6, 8]
Y_TICKS  = [0, 400, 800, 1200, 1600]
X_MIN, X_MAX = 0, 8.10   # small cushion right of 8 so curves don't touch the label
Y_MIN, Y_MAX = 0, 1600

# Distinct, colorblind-friendly palette
PALETTE = {
    'Test 1': '#1b9e77',  # teal
    'Test 2': '#d95f02',  # orange
    'Test 3': '#7570b3',  # purple
    'Test 4': '#e7298a',  # magenta
}

# Sheets and cross-sectional areas (mm²) — provided by you
SHEETS = ['Test 1', 'Test 2', 'Test 3', 'Test 4']
AREAS_MM2 = {
    'Test 1': 33.1751,
    'Test 2': 33.2913,
    'Test 3': 33.0576,
    'Test 4': 33.2904,
}

# =========================
# Upload workbook and pick newest .xlsx
# =========================
print(f"📁 UPLOAD your {SERIES}.xlsx NOW:")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No file uploaded.")

xlsx_candidates = sorted(Path('.').glob('*.xlsx'),
                         key=lambda p: p.stat().st_mtime, reverse=True)
if not xlsx_candidates:
    raise RuntimeError("No .xlsx file found after upload.")
filename = str(xlsx_candidates[0])
print(f"📦 Using file on disk: {filename}")

# =========================
# Helpers
# =========================
def find_strain_and_load_cols(df: pd.DataFrame):
    """Find columns for Strain (%) and Load (N). Accepts 'Load'/'Force' in N,
    and tolerates mislabel 'Stress (N)'; excludes pressure unit labels like MPa/Pa/GPa."""
    cols = [c for c in df.columns if isinstance(c, str)]
    low  = [c.lower() for c in cols]

    # Strain: prefer % headers, else any containing 'strain'
    cand_idx = [i for i, s in enumerate(low) if 'strain' in s]
    if not cand_idx:
        raise ValueError(f"Could not find a strain column. Columns: {cols}")
    strain_idx = next((i for i in cand_idx if ('%' in low[i]) or ('percent' in low[i])), cand_idx[0])
    strain_col = cols[strain_idx]

    # Load in Newtons (or mislabel 'Stress (N)'), exclude pressure units
    def looks_like_newtons(s):
        s = s.strip()
        if any(u in s for u in ['mpa', 'gpa', ' pa']):
            return False
        return bool(re.search(r'(\(|\[)\s*n\s*(\)|\])', s) or s.endswith(' n') or s.endswith('/n'))

    load_idx = None
    for i, s in enumerate(low):
        if re.search(r'\b(load|force)\b', s) and looks_like_newtons(s):
            load_idx = i
            break
    if load_idx is None:
        for i, s in enumerate(low):
            if 'stress' in s and looks_like_newtons(s):  # e.g., "Stress (N)" mislabel
                load_idx = i
                break
    if load_idx is None:
        raise ValueError(f"Could not find a Load/Force (N) column. Columns: {cols}")

    return strain_col, cols[load_idx]

def clean_series(x: pd.Series):
    return pd.to_numeric(x, errors='coerce')

# =========================
# Read sheets, convert N→MPa via area, assemble tidy DataFrame
# =========================
xls = pd.ExcelFile(filename)
df_list, skipped = [], []

for sheet in SHEETS:
    if sheet not in xls.sheet_names:
        print(f"ℹ️ Sheet '{sheet}' not found; skipping.")
        continue

    df_raw = pd.read_excel(xls, sheet_name=sheet)
    print(f"\n🔍 COLUMNS IN {sheet}: {list(df_raw.columns)}")
    strain_col, load_col = find_strain_and_load_cols(df_raw)

    area = AREAS_MM2.get(sheet, None)
    if area is None:
        print(f"ℹ️ No area provided for '{sheet}'; skipping.")
        skipped.append(sheet)
        continue

    print(f"✅ Using Strain: '{strain_col}' | Load (N): '{load_col}' | Area: {area:.4f} mm²")

    strain = clean_series(df_raw[strain_col])
    loadN  = clean_series(df_raw[load_col])

    # Convert strain to percent if it appears fractional (<=1.1)
    if strain.max(skipna=True) is not None and strain.max(skipna=True) <= 1.1:
        strain = 100.0 * strain

    stress_mpa = loadN / area  # 1 N/mm² = 1 MPa
    tmp = pd.DataFrame({'Strain(%)': strain, 'Stress (MPa)': stress_mpa, 'Sheet': sheet})
    tmp = tmp.dropna(subset=['Strain(%)', 'Stress (MPa)']).sort_values('Strain(%)')
    tmp = tmp.drop_duplicates(subset=['Strain(%)'], keep='last')  # if duplicate x, keep last
    df_list.append(tmp)

if not df_list:
    raise RuntimeError("No usable data found. Check sheet names and your areas.")

df = pd.concat(df_list, ignore_index=True)
print(f"\n✅ LOADED: {filename} | Sheets plotted: {sorted(df['Sheet'].unique())}")
if skipped:
    print(f"ℹ️ Skipped (no area defined): {skipped}")
print(f"📊 TOTAL ROWS: {len(df)} | Columns: {list(df.columns)}")

# =========================
# Styling
# =========================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': AXIS_LINE,
    'mathtext.default': 'regular',
    'lines.linewidth': 7,
    'lines.solid_capstyle': 'round'
})
plt.rcParams['font.serif'] = ["Palatino Linotype", "Palatino", "URW Palladio L"] + plt.rcParams.get('font.serif', [])

# =========================
# Plot
# =========================
plt.figure(figsize=(10.5, 7.8), dpi=1200)

for i, sheet in enumerate(SHEETS):
    d = df[df['Sheet'] == sheet]
    if d.empty:
        continue
    plt.plot(d['Strain(%)'], d['Stress (MPa)'],
             label=f'{SERIES}_Test {i+1}',
             color=PALETTE.get(sheet, None),
             zorder=2)

# Axis labels (bold)
plt.xlabel(r'$\epsilon$ (%)', fontsize=40, fontweight='bold')
plt.ylabel(r'$\sigma$ (MPa)', fontsize=40, fontweight='bold')

ax = plt.gca()
# Spines
for side in ('left', 'bottom', 'right', 'top'):
    ax.spines[side].set_linewidth(AXIS_LINE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Ticks & limits (keep 0,2,4,6,8 labels; add small right cushion)
ax.set_xticks(X_TICKS)
ax.set_yticks(Y_TICKS)
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Bold, thick ticks
ax.tick_params(axis='both', which='major',
               labelsize=38, width=AXIS_LINE, length=10, direction='out', pad=8)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight('bold')

# Figure label at top-center
ax.text(0.5, 1.02, FIG_LABEL_TEXT,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=FIG_LABEL_SIZE,
        color=FIG_LABEL_COLOR,
        fontweight='bold')

# Legend (bold entries, ordered Test 1..4 if present)
handles, labels = ax.get_legend_handles_labels()
desired = [f'{SERIES}_Test 1', f'{SERIES}_Test 2', f'{SERIES}_Test 3', f'{SERIES}_Test 4']
order = [labels.index(x) for x in desired if x in labels]
leg = ax.legend([handles[i] for i in order],
                [labels[i] for i in order],
                loc='lower right', bbox_to_anchor=(0.975, 0),
                fontsize=30, framealpha=0.85, borderpad=0.2,
                labelspacing=0.4, fancybox=True)

for txt in leg.get_texts():
    txt.set_fontweight('bold')

plt.tight_layout()
plt.subplots_adjust(left=0.14, bottom=0.12)

# =========================
# Save & show
# =========================
plt.savefig(OUTPUT_NAME, format='png', dpi=1200, bbox_inches='tight')
plt.show()

print("🎉 ULTRA-HIGH-DPI PNG SAVED!")
print(f"📄 Filename: {OUTPUT_NAME}")
print("✅ Stress (MPa) = Load (N) / Area (mm²) using your SA1100_15min areas.")
print(f"🏷️ Figure label placed: {FIG_LABEL_TEXT} (size {FIG_LABEL_SIZE}, color {FIG_LABEL_COLOR})")


