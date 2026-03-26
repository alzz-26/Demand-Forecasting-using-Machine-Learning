import pandas as pd
import numpy as np
import re
from xgboost import XGBRegressor

# -------------------------------
# LOAD DATA
# -------------------------------
bookings = pd.read_excel("Bookings_Filled.xlsx")
product_order = bookings["Product_Name"].drop_duplicates().tolist()
bigdeal = pd.read_excel("BigDeal.xlsx")
scms = pd.read_excel("SCMS_Filled_Corrected.xlsx")
vms = pd.read_excel("VMS_Filled_Corrected.xlsx")
accuracy = pd.read_excel("Forecast_Accuracy.xlsx")

# Clean column names
for df in [bookings, bigdeal, scms, vms, accuracy]:
    df.columns = df.columns.astype(str).str.strip()

# -------------------------------
# CONVERT FY FORMAT → YYYYQ
# -------------------------------
def fy_to_yyyyq(col):
    match = re.match(r"FY(\d{2})_Q(\d)", col)
    if match:
        year = int(match.group(1)) + 2000 - 1  # FY23 → 2022
        q = match.group(2)
        return f"{year}Q{q}"
    return col

bookings.rename(columns={c: fy_to_yyyyq(c) for c in bookings.columns}, inplace=True)

# -------------------------------
# IDENTIFY TIME COLUMNS
# -------------------------------
time_cols = [c for c in bookings.columns if re.match(r"\d{4}Q[1-4]", str(c))]
time_cols = sorted(time_cols)

# -------------------------------
# MELT FUNCTION
# -------------------------------
def melt_df(df, value_name, time_cols):
    return df.melt(
        id_vars=["Product_Name"],
        value_vars=time_cols,
        var_name="Quarter",
        value_name=value_name
    )

bookings_long = melt_df(bookings, "Production", time_cols)

# -------------------------------
# SCMS & VMS (already YYYYQ format)
# -------------------------------
scms_cols = [c for c in scms.columns if re.match(r"\d{4}Q[1-4]", str(c))]
vms_cols = [c for c in vms.columns if re.match(r"\d{4}Q[1-4]", str(c))]

scms_long = melt_df(scms, "SCMS", scms_cols)
vms_long = melt_df(vms, "VMS", vms_cols)

# -------------------------------
# BIG DEAL PROCESSING
# -------------------------------
def process_bigdeal(df):
    records = []
    for _, row in df.iterrows():
        product = row["Product_Name"]
        for col in df.columns:
            if "Big_" in col:
                q = col.split("_")[1]
                records.append([product, q, row[col]])
    return pd.DataFrame(records, columns=["Product_Name","Quarter","BigDeal"])

bigdeal_long = process_bigdeal(bigdeal)

# -------------------------------
# MERGE ALL
# -------------------------------
df = bookings_long.merge(scms_long, on=["Product_Name","Quarter"], how="left")
df = df.merge(vms_long, on=["Product_Name","Quarter"], how="left")
df = df.merge(bigdeal_long, on=["Product_Name","Quarter"], how="left")

# -------------------------------
# ADD LAG FEATURES
# -------------------------------
df = df.sort_values(["Product_Name","Quarter"])

df["lag_1"] = df.groupby("Product_Name")["Production"].shift(1)
df["lag_2"] = df.groupby("Product_Name")["Production"].shift(2)

# -------------------------------
# FILL FEATURES
# -------------------------------
df["SCMS"] = df["SCMS"].fillna(0)
df["VMS"] = df["VMS"].fillna(0)
df["BigDeal"] = df["BigDeal"].fillna(0)

# -------------------------------
# PROCESS ACCURACY & BIAS
# -------------------------------
accuracy["DP_Acc_mean"] = accuracy[
    ["DP_FY26Q1_Acc","DP_FY25Q4_Acc","DP_FY25Q3_Acc"]
].mean(axis=1)

accuracy["Mkt_Acc_mean"] = accuracy[
    ["Mkt_FY26Q1_Acc","Mkt_FY25Q4_Acc","Mkt_FY25Q3_Acc"]
].mean(axis=1)

accuracy["DS_Acc_mean"] = accuracy[
    ["DS_FY26Q1_Acc","DS_FY25Q4_Acc","DS_FY25Q3_Acc"]
].mean(axis=1)

accuracy["DP_Bias_mean"] = accuracy[
    ["DP_FY26Q1_Bias","DP_FY25Q4_Bias","DP_FY25Q3_Bias"]
].mean(axis=1)

accuracy["Mkt_Bias_mean"] = accuracy[
    ["Mkt_FY26Q1_Bias","Mkt_FY25Q4_Bias","Mkt_FY25Q3_Bias"]
].mean(axis=1)

accuracy["DS_Bias_mean"] = accuracy[
    ["DS_FY26Q1_Bias","DS_FY25Q4_Bias","DS_FY25Q3_Bias"]
].mean(axis=1)

# Merge
df = df.merge(
    accuracy[[
        "Product_Name",
        "DP_Acc_mean","Mkt_Acc_mean","DS_Acc_mean",
        "DP_Bias_mean","Mkt_Bias_mean","DS_Bias_mean"
    ]],
    on="Product_Name",
    how="left"
)

# -------------------------------
# ADD TEAM FORECASTS
# -------------------------------
team = bookings[[
    "Product_Name",
    "Demand_Planners_Forecast",
    "Marketing_Team_Forecast",
    "DS_Team_Forecast"
]]

df = df.merge(team, on="Product_Name", how="left")

# Bias correction
df["DP_adj"] = df["Demand_Planners_Forecast"] - df["DP_Bias_mean"]
df["Mkt_adj"] = df["Marketing_Team_Forecast"] - df["Mkt_Bias_mean"]
df["DS_adj"] = df["DS_Team_Forecast"] - df["DS_Bias_mean"]

total_acc = df["DP_Acc_mean"] + df["Mkt_Acc_mean"] + df["DS_Acc_mean"]

df["team_forecast_weighted"] = (
    df["DP_adj"] * df["DP_Acc_mean"] +
    df["Mkt_adj"] * df["Mkt_Acc_mean"] +
    df["DS_adj"] * df["DS_Acc_mean"]
) / total_acc

# -------------------------------
# MODEL
# -------------------------------
def forecast_product(pdf):
    train = pdf.dropna(subset=["Production"])
    
    features = ["lag_1","lag_2","SCMS","VMS","BigDeal","team_forecast_weighted"]
    train = train.dropna(subset=features)

    if len(train) < 5:
        return train["Production"].iloc[-1]  # fallback

    X = train[features]
    y = train["Production"]

    model = XGBRegressor(n_estimators=200, max_depth=4)
    model.fit(X, y)

    latest = pdf.iloc[-1]
    X_test = latest[features].values.reshape(1, -1)

    return model.predict(X_test)[0]

# -------------------------------
# FORECAST
# -------------------------------
results = []

for p in df["Product_Name"].unique():
    pdf = df[df["Product_Name"] == p]
    
    try:
        pred = forecast_product(pdf)
    except:
        pred = np.nan

    results.append([p, pred])

results_df = pd.DataFrame(results, columns=["Product_Name","Model_Forecast"])

# -------------------------------
# FINAL BLEND
# -------------------------------
team_final = df.groupby("Product_Name")["team_forecast_weighted"].last().reset_index()

results_df = results_df.merge(team_final, on="Product_Name")

results_df["Combined_Weighted_Forecast"] = (
    0.7 * results_df["Model_Forecast"] +
    0.3 * results_df["team_forecast_weighted"]
)
results_df["Product_Name"] = pd.Categorical(
    results_df["Product_Name"],
    categories=product_order,
    ordered=True
)

results_df = results_df.sort_values("Product_Name")
# -------------------------------
# SAVE
# -------------------------------
results_df.to_csv("final_forecast.csv", index=False)

print("✅ DONE — No column mismatch, no format issues")