import pandas as pd
import numpy as np

# Let's inspect the data around 2025-05-24
df = pd.read_csv("models/qaqc_outputs/raw_stage_network.csv", parse_dates=["date"]).set_index("date")
print(df.loc['2025-05-20':'2025-05-30'])
