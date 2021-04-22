import pandas as pd

df_check1 = pd.read_csv("check.2020Nov040022", sep=" ")
df_check2 = pd.read_csv("check.2020Nov032233", sep=" ")
df_check = pd.concat([df_check1, df_check2], ignore_index=True)


checked = dict()
for r in df_check.to_dict("records"):
    try:
        checked[r["identifier"]] = int(r["check"])
    except ValueError:
        checked[r["identifier"]] = None

dimrs = []
for r in pd.read_csv("2_sdes.csv").to_dict(orient="records"):
    i = r["identifier"]
    dimr = dict(identifier=i)
    if i in checked.keys():
        if checked[i] is None:
            continue
        dimr["dimension"] = int(checked[i])
    else:
        dimr["dimension"] = int(r["SDES_Dimension"])
    dimrs.append(dimr)

df = pd.DataFrame.from_records(dimrs)
df.to_csv("3_dim.csv", index=False)
df[["identifier"]].to_csv("3_dim.gcd", index=False, header=False)
