with open("Stage_Values_For_You.py", "r") as f:
    text = f.read()

text = text.replace("def robust_sigma(values: np.ndarray) -> float:", "def robust_sigma(values) -> float:")

with open("Stage_Values_For_You.py", "w") as f:
    f.write(text)
