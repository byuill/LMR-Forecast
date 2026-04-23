import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

df = pd.DataFrame({
    'y': [1, 2, 3, 4, 5],
    'f1': [1.1, 2.1, 2.9, 4.0, 5.0],
    'f2': [1.0, 2.0, 3.0, 4.0, 5.0]
})

model = HistGradientBoostingRegressor()
model.fit(df[['f1', 'f2']], df['y'])

test_df = pd.DataFrame({
    'f1': [np.nan, 1.5],
    'f2': [np.nan, np.nan]
})

print(model.predict(test_df))
