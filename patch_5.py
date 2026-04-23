with open("Stage_Values_For_You.py", "r") as f:
    text = f.read()

text = text.replace('pred_final = pd.Series(model.predict(\n        usable[feature_columns]), index=usable.index)', 'pred_final = pd.Series(model.predict(usable[feature_columns]), index=usable.index)\n    pred_all_imputed = pd.Series(model.predict(table[feature_columns]), index=table.index)')
text = text.replace('result = pd.DataFrame(index=usable.index)', 'result = pd.DataFrame(index=table.index)')
text = text.replace('result["observed"] = obs_final', 'result["observed"] = table[target_col]')
text = text.replace('result["predicted"] = pred_final', 'result["predicted"] = pred_all_imputed')
text = text.replace('result["residual"] = obs_final - pred_final', 'result["residual"] = table[target_col] - pred_all_imputed')
text = text.replace('result["is_outlier"] = flagged', 'flagged_all = pd.Series(False, index=table.index)\n    flagged_all.loc[usable.index] = flagged\n    result["is_outlier"] = flagged_all')
text = text.replace('result["corrected"] = np.where(flagged, pred_final, obs_final)', 'result["corrected"] = np.where(flagged_all | table[target_col].isna(), pred_all_imputed, table[target_col])')

with open("Stage_Values_For_You.py", "w") as f:
    f.write(text)
