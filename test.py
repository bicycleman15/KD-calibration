import pandas as pd
import os

df = {
        "model" : [12],
        "dataset" : ["mm"],
        "folder_path" : ["adasd"]
    }

df = pd.DataFrame(df)

headers = [x for x in df.keys()]
print(headers)

save_path = 'teacher_stats.csv'
df.to_csv(save_path, mode='a', index=False, header=(not os.path.exists(save_path)))