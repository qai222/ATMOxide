import neptune
import pandas as pd
from MLModule.utils import neptune_proj_name, neptune_api_token

proj = neptune.init(api_token=neptune_api_token, project_qualified_name=neptune_proj_name)
expts = proj.get_experiments(tag=["tuned", "2080"])
records = []
for expt in expts:
    r = dict(name=expt.name)
    log = expt.get_logs()
    for k in log.keys():
        if k.startswith("test"):
            v = log[k]
            r[k] = v["y"]
    records.append(r)
df = pd.DataFrame.from_records(records)
df.to_csv("compile_opt_results.csv", index=False)
