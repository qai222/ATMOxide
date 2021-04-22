import pandas as pd
from collections import Counter
df = pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")
c = Counter(df["dimension"])
total = sum(c.values(), 0.0)
for key in c:
    c[key] /= total
print(c)

df = pd.read_csv("../../DataGeneration/3_SDes/3_dim.csv")
c = Counter(df["dimension"])
total = sum(c.values(), 0.0)
for key in c:
    c[key] /= total
print(c)
