import os
import pandas as pd

folder_path = 'TCGA'
subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]

for dir in subfolders:
    tempfolder = os.path.join(folder_path, dir)
    pd.read_csv()