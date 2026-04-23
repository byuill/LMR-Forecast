with open("Stage_Values_For_You.py", "r") as f:
    text = f.read()

text = text.replace('all_outliers = pd.Series(False, index=usable.index)', 'all_outliers = pd.Series(False, index=usable.index)\n    pred_all = pd.Series(np.nan, index=usable.index)')

with open("Stage_Values_For_You.py", "w") as f:
    f.write(text)
