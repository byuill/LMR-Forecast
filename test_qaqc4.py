import pandas as pd
import numpy as np

# Let's inspect the predicted values, raw values, and corrected values to see why NaNs remain
df_raw = pd.read_csv("models/qaqc_outputs/raw_stage_network.csv", parse_dates=["date"]).set_index("date")
df_pred = pd.read_csv("models/qaqc_outputs/predicted_stage_network.csv", parse_dates=["date"]).set_index("date")
df_corr = pd.read_csv("models/qaqc_outputs/corrected_stage_network.csv", parse_dates=["date"]).set_index("date")

print("Raw:")
print(df_raw['stage_01545'].loc['2025-05-20':'2025-05-30'])
print("\nPred:")
print(df_pred['stage_01545'].loc['2025-05-20':'2025-05-30'])
print("\nCorr:")
print(df_corr['stage_01545'].loc['2025-05-20':'2025-05-30'])
