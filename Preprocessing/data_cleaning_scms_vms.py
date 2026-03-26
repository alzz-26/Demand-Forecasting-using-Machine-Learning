"""
fill_scms_vms.py
----------------
Fills missing values in SCMS.xlsx and VMS.xlsx using gap-length + anchor-value logic.

DECISION RULES
══════════════

Special override (checked first):
  row total = 0  OR  all NaN  →  zero fill entire row

Leading gap (NaNs before first real value):
  anchor = first known value
  gap <= THRESHOLD (4)         →  backward fill all  (too short to declare absent)
  gap >  THRESHOLD, anchor >10 →  backward fill all  (established product, data missing)
  gap >  THRESHOLD, anchor <=10 →  first 4 cells = 0, remaining = first val  (new entry)

Trailing gap (NaNs after last real value):
  anchor = last known value
  gap <= THRESHOLD (4)         →  forward fill all
  gap >  THRESHOLD, anchor >10 →  forward fill all
  gap >  THRESHOLD, anchor <=10 →  first 4 cells = last val, remaining = 0  (fading/exited)

Middle gap (NaNs between two real values):
  NO zeros ever inserted — product was selling on both sides
  anchor = min(before_val, after_val)
  gap <= 4                     →  linear interpolation
  gap >  4, anchor >10         →  segment mean fill
  gap >  4, anchor <=10        →  segment mean fill, capped at min(seg_mean, before_val)

Usage
-----
  pip install pandas openpyxl numpy
  python fill_scms_vms.py

Input  : SCMS.xlsx, VMS.xlsx  (same folder as this script)
Output : SCMS_Filled.xlsx, VMS_Filled.xlsx  (same folder)
"""

import numpy as np
import pandas as pd

QUARTERS = [
    '2023Q1','2023Q2','2023Q3','2023Q4',
    '2024Q1','2024Q2','2024Q3','2024Q4',
    '2025Q1','2025Q2','2025Q3','2025Q4',
    '2026Q1',
]

GAP_THRESHOLD       = 4   # gaps <= this: pure fill without splitting
ZERO_QUARTERS       = 4   # for long gaps with anchor<=10: how many cells to zero/fwd-fill
ANCHOR_THRESHOLD    = 10  # first/last value <= this triggers the zero-split logic


def _contiguous_blocks(indices):
    """Split a list of indices into contiguous runs."""
    if not indices:
        return []
    blocks, cur = [], [indices[0]]
    for i in indices[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            blocks.append(cur); cur = [i]
    blocks.append(cur)
    return blocks


def fill_row(row, seg_col, df):
    """
    Return dict of {quarter: (filled_value, strategy_tag)} for every NaN cell.
    Real values are not touched.
    """
    vals = row[QUARTERS].astype(float)
    nm   = vals.isna()
    if not nm.any():
        return {}

    # ── Override: zero total or all NaN ─────────────────────────────────────
    if nm.all() or vals.sum() == 0:
        return {q: (0.0, 'zero_fill') for q in QUARTERS if nm[q]}

    nan_idx   = [i for i in range(len(QUARTERS)) if nm.iloc[i]]
    valid_idx = [i for i in range(len(QUARTERS)) if not nm.iloc[i]]
    first_v   = min(valid_idx)
    last_v    = max(valid_idx)
    first_val = float(vals.iloc[first_v])
    last_val  = float(vals.iloc[last_v])

    results = {}

    leading  = [i for i in nan_idx if i < first_v]
    trailing = [i for i in nan_idx if i > last_v]
    inner    = [i for i in nan_idx if first_v < i < last_v]

    # ── LEADING ─────────────────────────────────────────────────────────────
    if leading:
        n = len(leading)
        if n <= GAP_THRESHOLD or first_val > ANCHOR_THRESHOLD:
            # short gap OR established product → backward fill all
            for i in leading:
                results[QUARTERS[i]] = (round(first_val, 4), 'backward_fill')
        else:
            # long gap AND new entry (anchor <=10) → first 4 zeros, rest backfill
            for j, i in enumerate(leading):
                if j < ZERO_QUARTERS:
                    results[QUARTERS[i]] = (0.0, 'zero_fill')
                else:
                    results[QUARTERS[i]] = (round(first_val, 4), 'backward_fill')

    # ── TRAILING ────────────────────────────────────────────────────────────
    if trailing:
        n = len(trailing)
        if n <= GAP_THRESHOLD or last_val > ANCHOR_THRESHOLD:
            # short gap OR established product → forward fill all
            for i in trailing:
                results[QUARTERS[i]] = (round(last_val, 4), 'forward_fill')
        else:
            # long gap AND low last value → first 4 forward fill, rest zeros
            for j, i in enumerate(trailing):
                if j < ZERO_QUARTERS:
                    results[QUARTERS[i]] = (round(last_val, 4), 'forward_fill')
                else:
                    results[QUARTERS[i]] = (0.0, 'zero_fill')

    # ── MIDDLE ──────────────────────────────────────────────────────────────
    if inner:
        for block in _contiguous_blocks(inner):
            before_val = float(vals.iloc[block[0] - 1])  if block[0] > 0              else 0.0
            after_val  = float(vals.iloc[block[-1] + 1]) if block[-1] < len(QUARTERS)-1 else 0.0
            anchor     = min(before_val, after_val)

            if len(block) <= GAP_THRESHOLD:
                # linear interpolation regardless of anchor
                filled = vals.interpolate(method='linear', limit_direction='both')
                for i in block:
                    results[QUARTERS[i]] = (round(float(filled.iloc[i]), 4), 'interpolate')
            else:
                # long middle gap → segment mean
                seg   = row[seg_col]
                prod  = row['Product_Name']
                peers = df[
                    (df['Product_Name'] == prod) & (df[seg_col] == seg)
                ].drop(index=row.name)

                for i in block:
                    q  = QUARTERS[i]
                    pv = peers[q].dropna()
                    if len(pv) > 0:
                        seg_mean = float(pv.mean())
                    else:
                        fb = df[df[seg_col] == seg][q].dropna()
                        seg_mean = float(fb.mean()) if len(fb) > 0 else 0.0

                    if anchor <= ANCHOR_THRESHOLD:
                        # low-volume context: cap fill so we don't overestimate
                        v = min(seg_mean, before_val) if before_val > 0 else seg_mean
                    else:
                        v = seg_mean

                    results[q] = (round(v, 4), 'segment_mean')

    return results


def impute(df, seg_col):
    df  = df.copy()
    log = []
    for idx, row in df.iterrows():
        row_copy      = row.copy()
        row_copy.name = idx
        filled = fill_row(row_copy, seg_col, df)
        for q, (val, strategy) in filled.items():
            df.at[idx, q] = val
            log.append({
                'Product_Name': row['Product_Name'],
                seg_col:        row[seg_col],
                'Quarter':      q,
                'Filled_Value': val,
                'Strategy':     strategy,
            })
    return df, log


def process(input_file, output_file, seg_col):
    from collections import Counter
    print(f"\nProcessing {input_file} ...")
    df = pd.read_excel(input_file)

    before      = df[QUARTERS].isna().sum().sum()
    df_out, log = impute(df, seg_col)
    after       = df_out[QUARTERS].isna().sum().sum()

    print(f"  Missing before : {before}")
    print(f"  Missing after  : {after}")
    print(f"  Cells filled   : {len(log)}")
    for strat, n in sorted(Counter(e['Strategy'] for e in log).items()):
        print(f"    {strat:<20s} : {n} cells")

    df_out.to_excel(output_file, index=False)
    print(f"  Saved -> {output_file}")
    return df_out, log


if __name__ == '__main__':
    process('SCMS.xlsx', 'SCMS_Filled.xlsx', 'Segment')
    process('VMS.xlsx',  'VMS_Filled.xlsx',  'Vertical')