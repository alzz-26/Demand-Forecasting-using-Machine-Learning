"""
fill_bookings.py
----------------
Fills missing actuals in Bookings.xlsx using the following rules:

  1. NPI-Ramp products           → 0  (product not yet launched)
  2. Sustaining, first value ≤10 → 0  (effectively a new launch despite label)
  3. Sustaining, first value >10 → backward fill with first known quarter value
                                    then apply rolling mean (window=3) ONLY if
                                    the product is NOT on an active ramp-up trend

Usage
-----
  pip install pandas openpyxl numpy
  python fill_bookings.py

Input  : Bookings.xlsx  (same folder)
Output : Bookings_Filled.xlsx (same folder)
"""

import numpy as np
import pandas as pd

INPUT  = "Bookings.xlsx"
OUTPUT = "Bookings_Filled.xlsx"

ACTUALS = [
    "FY23_Q2", "FY23_Q3", "FY23_Q4",
    "FY24_Q1", "FY24_Q2", "FY24_Q3", "FY24_Q4",
    "FY25_Q1", "FY25_Q2", "FY25_Q3", "FY25_Q4",
    "FY26_Q1",
]

# Products confirmed to be on active ramp — skip smoothing for these
# even if they pass the ramp-detection heuristic
NO_SMOOTH = {
    "WIRELESS ACCESS POINT WiFi6E (External Antenna) Outdoor",
    "SWITCH Data Center 25G/100G Leaf",
}


def is_active_ramp(available_vals):
    """Return True if the first 4 known values are strictly increasing."""
    return len(available_vals) >= 4 and all(
        available_vals[i] < available_vals[i + 1] for i in range(3)
    )


def fill_row(row):
    lc   = row["Product_Life_Cycle"]
    vals = row[ACTUALS].astype(float).copy()

    missing_mask   = vals.isna()
    if not missing_mask.any():
        return vals  # nothing to do

    available      = vals[~missing_mask].values
    first_valid_i  = missing_mask.values.argmin()   # index of first non-NaN
    first_val      = float(vals.iloc[first_valid_i])

    # ── Rule 1 & 2: zero fill ────────────────────────────────────────────────
    if lc == "NPI-Ramp" or (lc == "Sustaining" and first_val <= 10):
        vals[missing_mask] = 0.0
        return vals

    # ── Rule 3: backward fill + optional smoothing ───────────────────────────
    vals[missing_mask] = first_val          # backward fill

    skip_smooth = (
        row["Product_Name"] in NO_SMOOTH
        or is_active_ramp(available.tolist())
    )

    if not skip_smooth:
        rolled = vals.rolling(window=3, min_periods=1).mean()
        # Only overwrite the originally-missing positions
        vals[missing_mask] = rolled[missing_mask].round(1)

    return vals


def main():
    df = pd.read_excel(INPUT)
    print(f"Loaded {INPUT}  —  {df.shape[0]} products, {df.shape[1]} columns")

    before_nulls = df[ACTUALS].isna().sum().sum()
    print(f"Missing actuals before fill: {before_nulls}")

    # Apply fill row-by-row
    filled_actuals = df.apply(fill_row, axis=1)
    df[ACTUALS]    = filled_actuals[ACTUALS]

    after_nulls = df[ACTUALS].isna().sum().sum()
    print(f"Missing actuals after  fill: {after_nulls}")

    # Print a summary of what changed
    print("\nImputation summary:")
    orig = pd.read_excel(INPUT)[ACTUALS]
    for idx in df.index:
        for q in ACTUALS:
            if pd.isna(orig.loc[idx, q]):
                new_val = df.loc[idx, q]
                lc      = df.loc[idx, "Product_Life_Cycle"]
                name    = df.loc[idx, "Product_Name"]
                print(f"  [{lc:12s}]  {name[:50]:50s}  {q}  →  {new_val}")

    df.to_excel(OUTPUT, index=False)
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()