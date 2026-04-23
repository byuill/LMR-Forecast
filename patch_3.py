with open("Stage_Values_For_You.py", "r") as f:
    text = f.read()
text = text.replace("sigma = robust_sigma(s.values)", "sigma = robust_sigma(s.values.astype(float))")
text = text.replace("neighbours = np.concatenate(\n            [vals[i - half_window: i], vals[i + 1: i + half_window + 1]])", "neighbours = np.concatenate(\n            [vals[i - half_window: i].astype(float), vals[i + 1: i + half_window + 1].astype(float)])")
text = text.replace("sigma = robust_sigma((obs_final - pred_final).values)", "sigma = robust_sigma((obs_final - pred_final).values.astype(float))")
text = text.replace("sigma = robust_sigma(resid.values)", "sigma = robust_sigma(resid.values.astype(float))")
with open("Stage_Values_For_You.py", "w") as f:
    f.write(text)
